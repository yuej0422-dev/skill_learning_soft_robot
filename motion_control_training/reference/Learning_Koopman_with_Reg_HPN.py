from ntpath import join
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
# physics engine
from scipy.io import loadmat, savemat


# define network
def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.num_real = int(np.mod(self.Nkoopman, 2))
        self.num_complex_pair = int(self.Nkoopman / 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wb = torch.Tensor(self.u_dim, self.Nkoopman)
        wb = torch.nn.init.normal_(wb, 0, 0.1)
        self.B = nn.Parameter(wb, requires_grad=True)

        init_A = torch.squeeze(torch.normal(0., 0.01, size=(1, self.Nkoopman)))
        self.A = nn.Parameter(init_A,
                              requires_grad=True)  # initialize the weight matrix self.A as a trainable parameter

    def encode_only(self, x):
        return self.encode_net(x)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)

    def form_A_from_eigenvalues(self):
        """
        for time-invariant systems
        """
        assert self.Nkoopman == self.num_real + 2 * self.num_complex_pair, "the sum of all eigenvalues must equal the Nkoopman"
        idx = 0
        temp_A = torch.zeros([self.Nkoopman, self.Nkoopman])
        for i in range(self.num_complex_pair):
            idx = 2 * i
            temp_A[idx:idx + 2, idx:idx + 2] = form_complex_conjugate_block(self.A[idx],
                                                                            self.A[idx + 1])  # temp_A[0:2, 0:2]
        for i in range(self.num_real):
            idx = 2 * self.num_complex_pair + i
            temp_A[idx, idx] = self.A[idx]
        return temp_A


    def forward(self, x, u):
        # Get the Koopman matrix from eigenvalues
        temp_A = self.form_A_from_eigenvalues().to(self.device)

        # Compute the new state using the Koopman matrix A and control input
        x_new = torch.matmul(x, temp_A)  # Apply the Koopman matrix to the state
        Bu = torch.mm(u, self.B)
        return x_new + Bu  # Combine the linear state transition and control effect


def form_complex_conjugate_block(real, imaginary):
    """
    structure the block for system: x(k+1) = Ax(k)
    :param real:
    :param imaginary:
    :return:
    """
    if real.size() == torch.Size([]):  # for time-invariant cases
        block = torch.zeros([2, 2])
        block[0, 0] = real
        block[0, 1] = imaginary
        block[1, 0] = -imaginary
        block[1, 1] = real
    elif list(real.size())[0] >= 1:  # for time-variant situations
        batch_size = list(real.size())[0]
        block = torch.zeros([batch_size, 2, 2])
        # print(block[:, 0, 0].size())
        block[:, 0, 0] = real
        block[:, 0, 1] = imaginary
        block[:, 1, 0] = -imaginary
        block[:, 1, 1] = real
    return block


def collect_data(filename, Ksteps):
    """
    读取mat文件，提取X和U数组并按照Ksteps拆分成三维数组，使用滑动窗口最大化数据利用。

    参数:
    - filename: .mat文件路径
    - Ksteps: 每组数据包含的预测步数

    返回:
    - Kdata: 拆分后的三维数组，形状为 [样本数, Ksteps+1, 状态数+控制量数]
    """
    # 加载mat文件
    mat_data = loadmat(filename)

    # 提取 X 和 U 数据
    X = mat_data['X']
    U = mat_data['U']

    # 检查X和U行数是否一致
    if X.shape[0] != U.shape[0]:
        raise ValueError("X and U arrays must have the same number of rows.")

    # 合并X和U，得到状态+控制量的组合
    data = np.hstack((U, X))  # 形状变为 (行数, 状态数 + 控制量数)

    # 每个样本的步数是 Ksteps + 1 (包括当前时间步和未来Ksteps)
    total_rows = data.shape[0]

    # 最大化数据利用，使用滑动窗口提取样本
    num_samples = total_rows - Ksteps  # 样本数等于总行数 - Ksteps

    # 重新调整数据的形状为三维数组
    Kdata = np.zeros((num_samples, Ksteps + 1, data.shape[1]))  # 三维数组 [样本数, Ksteps+1, 状态数+控制量数]

    for i in range(num_samples):
        # 取每个样本的Ksteps+1步的数据
        start_idx = i
        end_idx = start_idx + (Ksteps + 1)
        Kdata[i] = data[start_idx:end_idx]

    return Kdata


# loss function
def define_loss(data, net, mse_loss, u_dim=1, gamma=0.99, Nstate=12):

    koopman_lam = 10  # 10
    A_eig_lam = 0.003  # 0.003
    svd_lam = 0.003  # 0.003
    Aug_lam = 1
    pred_lam_C = 1  #

    train_traj_num, steps, _ = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_current = net.encode(data[:, 0, u_dim:])
    beta = 1.0
    beta_sum = 0.0

    linear_loss = torch.zeros(1, dtype=torch.float32).to(device)
    A_eig_loss = torch.zeros(1, dtype=torch.float32).to(device)
    svd_loss = torch.zeros(1, dtype=torch.float32).to(device)
    Aug_loss = torch.zeros(1, dtype=torch.float32).to(device)
    pred_loss_C = torch.zeros(1, dtype=torch.float32).to(device)

    for i in range(steps - 1):
        X_current = net.forward(X_current, data[:, i, :u_dim])
        beta_sum += beta

        pred_loss_C += beta * mse_loss(X_current[:, :Nstate], data[:, i + 1, u_dim:])

        X_next_encode = net.encode(data[:, i + 1, u_dim:])
        linear_loss += beta * mse_loss(X_current, X_next_encode)

        beta *= gamma

        X_current_encoded = net.encode(X_current[:, :Nstate])
        Aug_loss += mse_loss(X_current_encoded, X_current)

    pred_loss_C = pred_loss_C / beta_sum
    linear_loss = linear_loss / beta_sum

    Am = net.form_A_from_eigenvalues().T.to(device)
    # A_eig,_=torch.linalg.eig(Am)
    A_dim = net.Nkoopman
    temp = net.B.T.to(device)
    controllability_matrix = torch.zeros((A_dim, A_dim * u_dim))
    for i in range(A_dim):
        controllability_matrix[:, i * u_dim:(i + 1) * u_dim] = temp
        temp = torch.mm(Am, temp)
    U, S, Vh = torch.linalg.svd(controllability_matrix)
    c = S.abs() - 0.2 * torch.ones(1)  # 0.992
    # print("S", S)
    # print("S.abs()",S.abs())
    mask = c < 0
    for item in c[mask]:
        # A_eig_loss += 2*torch.norm(item, p=2)
        svd_loss += 1 * torch.norm(item, p=2)

    for i in range(net.num_complex_pair):
        idx = 2 * i
        c1 = net.A[idx].abs() - 0.5 * torch.ones(1).to(device)  # 0.992
        A_eig_loss += 1*torch.norm(c1, p=2)  #>0.97
        c2 = net.A[idx+1].abs() - 0 * torch.ones(1).to(device)  # 0.992
        A_eig_loss += torch.norm(c2, p=2)
    for i in range(net.num_real):
        idx = 2 * net.num_complex_pair + i
        c = net.A[idx].abs() - 0.5 * torch.ones(1).to(device)  # 0.992
        A_eig_loss += 1*torch.norm(c, p=2)

    loss = koopman_lam*linear_loss + A_eig_lam*A_eig_loss + svd_lam*svd_loss + Aug_lam*Aug_loss + pred_lam_C*pred_loss_C

    return loss, linear_loss, A_eig_loss, svd_loss, Aug_loss, pred_loss_C


def train(epoch=700, suffix="", encode_dim=12, layer_depth=3, gamma=0.5, batch_size=4096):
    """
    训练网络，使用epoch替代train_steps，并随机选取文件组成批次进行训练。

    参数:
    - epoch: 训练的轮数
    - suffix: 训练数据的后缀
    - encode_dim: 编码维度
    - layer_depth: 网络层数
    - gamma: 损失函数中的超参数
    - batch_size_file: 每个批次选取的文件数
    - Ksteps: 每个样本的预测步数
    """
    train_file_num = 540  # 训练文件的数量
    test_file_num = 60  # 验证文件的数量
    Ksteps = 50
    u_dim = 12
    in_dim = 12
    Nstate = in_dim
    layer_width = 128
    # layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    layers = [12, 64, 128, 64, 12]
    Nkoopman = in_dim + encode_dim
    print("layers:", layers)
    net = Network(layers, Nkoopman, u_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 3e-4
    if torch.cuda.is_available():
        net.cuda()
    net.float()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 训练日志
    subsuffix = suffix + "_" + "layer{}_edim{}".format(layer_depth, encode_dim)
    logdir = "Data/" + suffix + "/" + subsuffix
    if not os.path.exists("Data/" + suffix):
        os.makedirs("Data/" + suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)

    # 存储所有文件的训练数据
    all_train_files = list(range(train_file_num))  # 训练文件的索引
    all_data = []

    for file_idx in all_train_files:
        filename = f"data_SoftRobot_50/BL-RS-WL_koopman-data_mix_v3/soft_robot_{file_idx}"
        Kdata = collect_data(filename, Ksteps)  # 获取当前文件的数据
        all_data.append(Kdata)

    # 将所有数据拼接成一个大的数组
    all_data = np.concatenate(all_data, axis=0)
    print("Training Data Shape：", all_data.shape)

    num_samples = len(all_data)
    num_batches = num_samples // batch_size

    # 准备验证集数据
    batch_data_val = []
    for test_file_idx in range(test_file_num):  # 验证文件的数量是22
        test_filename = f"data_SoftRobot_50/BL-RS-WL_koopman-data_mix_v3/soft_robot_test_{test_file_idx}"
        Kdata_val = collect_data(test_filename, Ksteps)  # 获取验证文件的数据
        batch_data_val.append(Kdata_val)

    batch_data_val = np.concatenate(batch_data_val, axis=0)  # 拼接成一个大的验证集
    # print('batch_data_val:', batch_data_val.shape)

    X_val = torch.tensor(batch_data_val, dtype=torch.float32).to(device)  # 转换为tensor并加载到GPU
    print("Valuation Data Shape：", batch_data_val.shape)

    best_loss = 1000.0
    # 训练过程中，按epoch进行迭代
    for e in range(epoch):
        # 每个epoch内，生成随机索引
        indices = np.arange(num_samples)  # 所有数据的索引
        np.random.shuffle(indices)  # 随机打乱索引

        for batch_idx in range(num_batches):
            # 获取当前batch的随机索引
            batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # 根据batch_indices从all_data中取出数据
            batch_data = all_data[batch_indices]

            # 转换为tensor并加载到GPU
            X = torch.tensor(batch_data, dtype=torch.float32).to(device)

            # 计算损失
            loss, linear_loss, A_eig_loss, svd_loss, Aug_loss, pred_loss_C = define_loss(X, net, mse_loss, u_dim, gamma, Nstate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/loss', loss.item(), e * num_batches + batch_idx)
            writer.add_scalar('Train/linear_loss', linear_loss.item(), e * num_batches + batch_idx)
            writer.add_scalar('Train/A_eig_loss', A_eig_loss.item(), e * num_batches + batch_idx)
            writer.add_scalar('Train/svd_loss', svd_loss.item(), e * num_batches + batch_idx)
            writer.add_scalar('Train/Aug_loss', Aug_loss.item(), e * num_batches + batch_idx)
            writer.add_scalar('Train/pred_loss_C', pred_loss_C.item(), e * num_batches + batch_idx)

        # 每个epoch结束后进行一次验证评估
        # 计算验证损失
        loss_val, linear_loss, A_eig_loss, svd_loss, Aug_loss, pred_loss_C = define_loss(X_val, net, mse_loss, u_dim, gamma, Nstate)
        loss_val = loss_val.detach().cpu().numpy()

        writer.add_scalar('Eval/loss', loss_val, e)
        writer.add_scalar('Eval/linear_loss', linear_loss.item(), e)
        writer.add_scalar('Eval/A_eig_loss', A_eig_loss.item(), e)
        writer.add_scalar('Eval/svd_loss', svd_loss.item(), e)
        writer.add_scalar('Eval/Aug_loss', Aug_loss.item(), e)
        writer.add_scalar('Eval/pred_loss_C', pred_loss_C.item(), e)

        if loss_val < best_loss:
            best_loss = copy(loss_val)
            best_state_dict = copy(net.state_dict())
            Saved_dict = {'model': best_state_dict, 'layer': layers, 'optimizer': optimizer.state_dict()}
            torch.save(Saved_dict, "Data/" + suffix + "/" + subsuffix + ".pth")

        print(f"Epoch:{e + 1}/{epoch} Eval-loss:{loss_val}")

    print(f"Training finished. Best Eval-loss: {best_loss}")


def main():
    train(suffix=args.suffix, encode_dim=args.encode_dim, layer_depth=args.layer_depth,
          gamma=args.gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="DeepKoopman_BL-RS-WL_runtime2")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--encode_dim", type=int, default=12)
    parser.add_argument("--layer_depth", type=int, default=3)
    args = parser.parse_args()
    main()

