import torch
import numpy as np
import torch.nn as nn
import math
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
import scipy.io
from torch.utils.tensorboard import SummaryWriter



class Network(nn.Module):
    def __init__(self, encode_layers):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)


def U_loss(data,net,u_dim,mse_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    U_hat = net.encode_net(data[:, u_dim:])
    loss = mse_loss(U_hat, data[:, :u_dim])
    return loss

def quaternions_to_euler_angles(quaternions, rotation_sequence='sxyz'):
    euler_angles = []

    for quaternion in quaternions:
        q0, q1, q2, q3 = quaternion
        R = np.array([
            [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
            [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
            [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
        ])

        if rotation_sequence == 'sxyz':
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            euler_x = math.atan2(R[2, 1], R[2, 2])
            euler_y = math.atan2(-R[2, 0], sy)
            euler_z = math.atan2(R[1, 0], R[0, 0])
        elif rotation_sequence == 'szyx':
            asiny = -2*q1*q3 + 2*q0*q2
            asiny = np.clip(asiny, -1, 1)
            euler_x = math.atan2(R[1, 0], q0**2+q1**2-q2**2-q3**2)
            euler_y = math.asin(asiny)
            euler_z = math.atan2(R[2, 1], q0**2-q1**2-q2**2+q3**2)
        else:
            raise ValueError("Unsupported rotation sequence.")

        euler_angles.append([euler_x, euler_y, euler_z])

    return np.array(euler_angles)


def normalization_state(original_data, min_vals, max_vals):
    num_rows = original_data.shape[0]
    normalized_data = np.zeros_like(original_data)
    for i in range(num_rows):
        if (max_vals[i] - min_vals[i]) == 0:
            normalized_data[i, :] = 0
        else:
            normalized_data[i, :] = 2 * (original_data[i, :] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1
    return normalized_data


def train(train_steps=5000, suffix="", layer_depth=2):

    Kbatch_size = 64
    u_dim = 12

    # 数据准备
    folder_path = "Data/collect_data_KORL"
    file_name = "static_data_BL-BL-BL.mat"
    file_path = os.path.join(folder_path, file_name)
    Total_data = scipy.io.loadmat(file_path)
    # Total_data = np.concatenate((Total_data['u'], Total_data['x']), axis=1)
    X_data = Total_data['X_data']
    U_data = Total_data['U_data']
    Total_data = np.concatenate((U_data, X_data[:, :6]), axis=1)
    # Total_data = Total_data[:, 0:15]


    # 划分训练集和测试集
    row_num = Total_data.shape[0]
    Test_num = int(row_num * 0.2)
    indices = np.random.permutation(row_num)
    Total_data = Total_data[indices, :]
    Ktrain_data0 = Total_data[Test_num:row_num, :]
    Ktest_data0 = Total_data[:Test_num, :]
    print('Ktrain_data0.shape:{}'.format(Ktrain_data0.shape))
    print('Ktest_data0.shape:{}'.format(Ktest_data0.shape))
    # print(Total_data[:10, :])

    # min_vals = np.concatenate([-np.ones((1, u_dim)), np.min(Total_data[:, u_dim:], axis=0).reshape(1, -1)], axis=1).ravel()
    # max_vals = np.concatenate([np.ones((1, u_dim)), np.max(Total_data[:, u_dim:], axis=0).reshape(1, -1)], axis=1).ravel()
    min_vals = np.concatenate([-np.ones((1, u_dim)), -np.ones((1, 6))], axis=1).ravel()
    max_vals = np.concatenate([np.ones((1, u_dim)), np.ones((1, 6))], axis=1).ravel()
    # min_vals = np.min(Total_data[:, :], axis=0).ravel()
    # max_vals = np.max(Total_data[:, :], axis=0).ravel()
    print('min_vals:{}'.format(min_vals))
    print('max_vals:{}'.format(max_vals))

    # 归一化
    Ktrain_data = normalization_state(Ktrain_data0.T, min_vals, max_vals).T
    Ktest_data = normalization_state(Ktest_data0.T, min_vals, max_vals).T
    Ktrain_samples = Ktrain_data.shape[0]

    in_dim = Ktest_data.shape[-1] - u_dim
    layer_width = 32
    layers = [in_dim] + [32] + [64] + [32] + [u_dim]
    print("layers:", layers)
    net = Network(layers)
    # print(net.named_modules())
    learning_rate = 5e-4
    if torch.cuda.is_available():
        net.cuda()
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)
    # train
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    subsuffix = suffix + "_with_Regularization_" + "layer{}".format(layer_depth)
    logdir = "Data/" + suffix + "/" + subsuffix
    if not os.path.exists("Data/" + suffix):
        os.makedirs("Data/" + suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in range(train_steps):
        # K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[Kindex[:Kbatch_size], :]
        loss = U_loss(X, net, u_dim, mse_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('Train/Kloss', Kloss, i)
        # writer.add_scalar('Train/Eloss', Eloss, i)
        writer.add_scalar('Train/loss', loss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i + 1) % eval_step == 0:
            # K loss
            loss = U_loss(Ktest_data, net, u_dim, mse_loss)
            loss = loss.detach().cpu().numpy()
            # writer.add_scalar('Eval/Kloss', Kloss, i)
            # writer.add_scalar('Eval/Eloss', Eloss, i)
            writer.add_scalar('Eval/loss', loss, i)
            if loss < best_loss:
                best_loss = copy(loss)
                best_state_dict = copy(net.state_dict())
                Saved_dict = {'model': best_state_dict, 'layer': layers}
                file_path = os.path.join(f"Data/{suffix}", subsuffix + ".pth")
                torch.save(Saved_dict, file_path)
            print("Step:{} Eval-loss{}".format(i, loss))
        writer.add_scalar('Eval/best_loss', best_loss, i)
            # print("-------------END-------------")
    print("END-best_loss{}".format(best_loss))


def main():
    train(suffix=args.suffix,  layer_depth=args.layer_depth)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="HPN_three_KORL_task")
    parser.add_argument("--layer_depth", type=int, default=4)
    args = parser.parse_args()
    main()

