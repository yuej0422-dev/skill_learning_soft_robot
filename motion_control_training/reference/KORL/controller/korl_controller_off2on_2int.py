import torch

import numpy as np
import os
import scipy.io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
# import mujoco
# import mujoco_viewer
import time
import yaml

import LuMoSDKClient
import serial
import struct

# 导入 PolicyNet 类
# from cql_kop_Async_Qlinear import (KoopmanFeedforwardPolicy, KoopmanEncoder,
#                                    KoopmanInformedQFunction, Scalar, ReparameterizedSigmGaussian)
from cql_kop_Async_Qlinear_int import (KoopmanFeedforwardPolicy, KoopmanEncoder,
                                   KoopmanInformedQFunction, Scalar, ReparameterizedSigmGaussian)

def vec_z(a):  # vectorize vector for iteration
    size_a = a.shape[0]
    newsize = int(size_a * (size_a + 1) / 2)
    v = np.zeros(newsize)
    n = 0
    for i in range(size_a):
        for j in range(size_a):
            if j >= i:
                v[n] = a[i] * a[j]
                n = n + 1
    return v.reshape(-1, 1)


def vec_H(A):  # vectorize H matrix for iteration
    size_A = A.shape[0]
    newsize = int(size_A * (size_A + 1) / 2)
    a = np.zeros(newsize)
    # print(a.shape)
    n = 0
    for i in range(size_A):
        for j in range(size_A):
            if j >= i:
                if i == j:
                    a[n] = A[i, j]
                else:
                    a[n] = 2 * A[i, j]
                n = n + 1
    return a.reshape(-1, 1)


def vec_H_inv(a):  # recover H matrix
    a_size = a.shape[0]
    p = np.poly1d([0.5, 0.5, -a_size])
    r = np.roots(p)
    # print(r)
    X = np.max(r)
    # print(X)
    if X - int(X) > 0.5:
        A_size = math.ceil(X)
    else:
        A_size = int(X)
    # print(A_size)
    A = np.zeros((A_size, A_size))
    n = 0
    for i in range(A_size):
        for j in range(A_size):
            if j >= i:
                if i == j:
                    A[i, j] = a[n]
                else:
                    A[i, j] = a[n] / 2
                n = n + 1
    # print('A',A)
    A = A + A.T - np.diag(np.diag(A))
    return A

def normalization_state(original_data, min_vals, max_vals):
    num_rows = original_data.shape[0]
    normalized_data = np.zeros_like(original_data)
    for i in range(num_rows):
        normalized_data[i, :] = 2 * (original_data[i, :] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1
    return normalized_data


def main():
    # ===========================
    # 串口初始化
    # ===========================
    ser = serial.Serial("COM3", 115200)  # 打开COM17，将波特率配置为115200，其余参数使用默认值
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
    else:
        print("打开串口失败。")
    ip = "192.168.140.1"
    # ip = "169.254.51.8"
    # 初始化
    LuMoSDKClient.Init()
    # 设置Ip
    LuMoSDKClient.Connnect(ip)

    # ===========================
    # 部署策略网络
    # ===========================
    MODEL_DIR = "D:/controldesk/zxl/pycharm_project/CORL-main/checkpoints/0702/KORL_cql_HPN_test_v1/KORL_cql-HPN_test-aa02e449"  # 替换为您的模型目录
    ACTOR_MODEL_NAME = "checkpoint_1000.pt"
    model_path = os.path.join(MODEL_DIR, ACTOR_MODEL_NAME)
    configs_path = os.path.join(MODEL_DIR, 'config.yaml')

    # ===========================
    # 加载配置文件 config.yaml
    # ===========================
    with open(configs_path, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = 'cpu'

    # 提取配置文件中的参数
    state_dim = 24
    action_dim = 12
    max_action = 1  # 最大动作值，根据具体需求调整

    # ===========================
    # 初始化网络模型
    # ===========================
    Koopman_encoder = KoopmanEncoder(
        encode_layers=config['koopman_encode_layers'],  # e.g., [12, 256, 256, 24]
        u_dim=action_dim
    ).to(config['device'])

    actor = KoopmanFeedforwardPolicy(
        state_dim,
        action_dim,
        max_action,
        Koopman_encoder=Koopman_encoder,  # ✅ 注入 encoder
        log_std_offset=0,
        log_std_multiplier=config['policy_log_std_multiplier'],
    ).to(config['device'])

    critic_1_kop = KoopmanInformedQFunction(
        Koopman_encoder,
        actor,
        state_dim,
        action_dim,
    ).to(config['device'])
    # ===========================
    # 加载训练好的模型
    # ===========================
    # 加载保存的模型权重
    checkpoint = torch.load(model_path, map_location=config['device'])

    # 提取 checkpoint 中的 actor 权重并加载
    Koopman_encoder.load_state_dict(checkpoint['koopman_encoder'])
    actor.load_state_dict(checkpoint['actor'])  # 假设模型文件中保存了 'actor' 的权重
    critic_1_kop.load_state_dict(checkpoint['critic1_kop'])

    # 确保模型设置为评估模式
    actor.eval()

    H, _ = critic_1_kop.compute_H_and_K()
    H = H.detach().numpy()

    # ===========================
    # 仿真参数和数据初始化
    # ===========================
    # 状态空间维度
    n_x = 12
    n_u = 12
    sim_step = 500
    t_s = 0.02

    n_lift = 24
    space_dim = n_x + n_lift
    mr = n_u
    ny = 6
    nr = space_dim + ny
    n_learning_step = 0
    n_explore_step = 0  # 1000
    learning_rate = 5e-6  # 0.0005
    z_size = nr + mr  # Number of states in z-vector
    L = int(z_size * (z_size + 1) / 2)  # /Size of vectorized z
    N = L + 5  # 1* L  Number of measurements before evaluation % 10* L;
    # K =  zeros(m, n) # the starting point of the learning process
    # beta = 0.1  # parameter for RLS
    # Pk = beta * np.eye(L)  # parameter for RLS
    gamma = 0.99

    # 定义性能指标权重矩阵
    Q_temp_1 = 1 * np.array([[1, 1, 1]])  # 2.5
    Q_temp_2 = 1 * np.ones((1, 3))  # 2.5
    Q_temp_3 = 1 * np.ones((1, 6))  # 2.5
    Q_temp_4 = 1 * np.ones((1, 24))  # 2.5
    Q_temp_5 = 0.5 * np.ones((1, ny))  # 0.2

    Q_temp = np.concatenate((Q_temp_1, Q_temp_2, Q_temp_3, Q_temp_4, Q_temp_5), axis=1)
    # print('Q_temp',Q_temp)
    Q = np.diag(Q_temp.flatten())
    R = (2000 * np.eye(n_u, n_u))
    # 12 lift_dim
    # H_extended = np.zeros((42, 42))
    # # 拷贝原 H 到新位置（状态误差 + 控制增量）
    # H_extended[0:24, 0:24] = H[0:24, 0:24]  # 状态误差块
    # H_extended[0:24, 30:42] = H[0:24, 24:36]  # 状态对控制耦合
    # H_extended[30:42, 0:24] = H[24:36, 0:24]  # 控制对状态耦合
    # H_extended[30:42, 30:42] = H[24:36, 24:36]  # 控制增量块
    # # 添加积分误差的初始化项（对角线小正数，其他为0）
    # H_extended[24:30, 24:30] = 0.01 * np.eye(6)  # 积分误差自身加权
    # H = H_extended
    # 24 lift_dim
    # H_extended = np.zeros((54, 54))
    # # 拷贝原 H 到新位置（状态误差 + 控制增量）
    # H_extended[0:36, 0:36] = H[0:36, 0:36]  # 状态误差块
    # H_extended[0:36, 42:54] = H[0:36, 36:48]  # 状态对控制耦合
    # H_extended[42:54, 0:36] = H[36:48, 0:36]  # 控制对状态耦合
    # H_extended[42:54, 42:54] = H[36:48, 36:48]  # 控制增量块
    # # 添加积分误差的初始化项（对角线小正数，其他为0）
    # H_extended[36:42, 36:42] = 1 * np.eye(6)  # 积分误差自身加权
    # H = H_extended
    # print(H.shape)

    H_vec = vec_H(H)  # Vectorized value function matrix
    Hyy1 = H[nr:nr + mr, nr:nr + mr]  # 2*2
    Hyx1 = H[nr:nr + mr, 0:nr]  # 2*27
    Lf = -np.dot(np.linalg.inv(Hyy1), Hyx1)  # 2*27
    K = -Lf
    # print(K.shape)

    phi = np.zeros((L, N - 1))
    upsilon = np.zeros(N - 1)
    Z = np.zeros((nr + mr, sim_step))
    update_j = 1  # 5
    alpha = 127  # 50
    K_error = np.zeros(int((sim_step - alpha) / update_j))
    Kerroriter = 0
    batch_size = 128  # 5
    epsilon = 1e-8
    beta1 = 0.9  # 0.9
    beta2 = 0.999  # 0.999
    m_hat = np.zeros(H_vec.shape[0]).reshape(-1, 1)
    v_hat = np.zeros(H_vec.shape[0]).reshape(-1, 1)

    min_vals = np.array(
        [-0.209232955932617, 0.259572998046875, -0.205349105834961, -1.56803422927032, -3.15484933686399, -5.11506080141743, -1.16360193252563, -0.581957397460938, -0.773994607925415, -2.97411167492716, -27.0047700366781, -29.4927647021136])
    max_vals = np.array(
        [0.352538726806641, 0.595491577148438, 0.347042083740234, 1.56078949899892, 5.67225835022331, 3.38901336834750, 1.05944358825684, 0.429165954589844, 1.15636062622070, 4.90902231433771, 29.1294288855800, 18.7859489820633])

    # save data
    data = {
        'delta_u': np.zeros((sim_step, n_u)),
        'x_lift': np.zeros((sim_step, space_dim)),
        'e_extend': np.zeros((sim_step, nr)),
        'e': np.zeros((sim_step, n_x)),  # 时间
        'state': np.zeros((sim_step, n_x)),  # 状态 (位置和速度)
        'action': np.zeros((sim_step, n_u)),  # 控制输入
        'K0': K,
        'K': np.zeros((K.shape[0], K.shape[1])),
        'H': np.zeros((H.shape[0], H.shape[1]))
    }

    # 期望状态（根据需要修改）
    x_d = np.concatenate((
        np.array([[2.66038971e-01,  4.42966125e-01,  7.44434280e-02, -8.78455233e-03, 1.93662619e-02, -1.17662705e+00]]),
        # np.array([[-1.49389633e-01,  4.12539124e-01,  6.62979965e-02, -9.00200713e-02, -2.63820611e-01,  1.63228153e+00]]),
        np.zeros((1, 6))
    ), axis=1).T  # 形状应为 (12, 1)
    u_ref = np.array([0.1, 0.1, 0.45, 0.45, 0.08, 0.08, 0.45, 0.45,0.08, 0.08, 0.35, 0.35])

    x_d_normalization = normalization_state(x_d, min_vals, max_vals)
    x_d_lift = Koopman_encoder.encode(torch.tensor(x_d_normalization.T, dtype=torch.float32)).detach().numpy().reshape(-1, 1)

    sensor_data = []
    frame = LuMoSDKClient.ReceiveData(0)  # 0 :阻塞接收 1：非阻塞接收
    # if frame is None:
    #     continue
    for rigid in frame.rigidBodys:
        if rigid.Id == 1:
            sensor_data = [0.001 * rigid.X, 0.001 * rigid.Y, 0.001 * rigid.Z, rigid.speeds.XfSpeed,
                           rigid.speeds.YfSpeed,
                           rigid.speeds.ZfSpeed,
                           np.pi * rigid.eulerAngle.X / 180.00, np.pi * rigid.eulerAngle.Y / 180.00,
                           np.pi * rigid.eulerAngle.Z / 180.00,
                           rigid.palstance.fXPalstance, rigid.palstance.fYPalstance, rigid.palstance.fZPalstance]
    sensor_data = np.array(sensor_data)
    x_pos = sensor_data[0:3].reshape(3, 1)
    x_ang_radian = sensor_data[6:9].reshape(3, 1)
    x_pos_list = x_pos
    x_ang_radian_list = x_ang_radian
    x_pos_vel = sensor_data[3:6].reshape(3, 1)
    x_ang_radian_vel = np.array([0, 0, 0]).reshape(3, 1)
    x0 = np.concatenate((
        x_pos.reshape(1, 3),
        x_ang_radian.reshape(1, 3),
        x_pos_vel.reshape(1, 3),
        x_ang_radian_vel.reshape(1, 3)
    ), axis=1).T  # 形状为 (12, 1)
    x0_normalization = normalization_state(x0, min_vals, max_vals)
    x_lift = Koopman_encoder.encode(torch.tensor(x0_normalization.T, dtype=torch.float32)).detach().numpy().reshape(-1, 1)

    q = np.zeros((ny, 1))
    e_lift = x_lift - x_d_lift
    e_extend = np.concatenate((e_lift, q))
    delta_u = np.zeros((n_u, 1))
    e_extend_former = e_extend
    u_former = delta_u
    q = q + t_s * e_lift[:ny, :]

    # 导入策略
    # mat = scipy.io.loadmat("control_data/result.mat")
    # K = mat['K']
    # H = mat['H']
    # H_vec = vec_H(H)

    # ===========================
    # 控制循环
    # ===========================
    for iter in range(sim_step):
        tic = time.time()

        frame = LuMoSDKClient.ReceiveData(0)  # 0 :阻塞接收 1：非阻塞接收
        # if frame is None:
        #     continue
        for rigid in frame.rigidBodys:
            if rigid.Id == 1:
                sensor_data = [0.001 * rigid.X, 0.001 * rigid.Y, 0.001 * rigid.Z, rigid.speeds.XfSpeed,
                               rigid.speeds.YfSpeed,
                               rigid.speeds.ZfSpeed,
                               np.pi * rigid.eulerAngle.X / 180.00, np.pi * rigid.eulerAngle.Y / 180.00,
                               np.pi * rigid.eulerAngle.Z / 180.00,
                               rigid.palstance.fXPalstance, rigid.palstance.fYPalstance, rigid.palstance.fZPalstance]
        sensor_data = np.array(sensor_data)
        x_pos = sensor_data[0:3].reshape(3, 1)
        x_ang_radian = sensor_data[6:9].reshape(3, 1)
        x_pos_list = np.concatenate((x_pos_list, x_pos), axis=1)  # 按列连接
        x_ang_radian_list = np.concatenate((x_ang_radian_list, x_ang_radian), axis=1)  # 按列连接
        x_pos_vel = sensor_data[3:6].reshape(3, 1)
        # x_ang_radian_vel = sensor_data[9:12].reshape(3, 1)
        x_ang_radian_vel = (x_ang_radian_list[:, -1] - x_ang_radian_list[:, -2]) / t_s

        x = np.concatenate((
            x_pos.reshape(1, 3),
            x_ang_radian.reshape(1, 3),
            x_pos_vel.reshape(1, 3),
            x_ang_radian_vel.reshape(1, 3)
        ), axis=1).T  # 形状为 (12, 1)
        x_normalization = normalization_state(x, min_vals, max_vals)
        # x_normalization = (x.flatten() - mean) / std
        x_lift = Koopman_encoder.encode(torch.tensor(x_normalization.T, dtype=torch.float32)).detach().numpy().reshape(-1, 1)
        state = np.concatenate((x_normalization, x_d_normalization)).flatten()
        e_lift = x_lift - x_d_lift
        e_extend = np.concatenate((e_lift, q))

        if iter % 50 == 0:

            # enforce control limit
            a1 = 0.001 * np.ones((mr, 1))
            a2 = 0.001 * np.ones((mr, 1))
            sim_noise = 0.0001 * np.random.normal(loc=0, scale=1000, size=(mr, 1))
            n_explor = sim_noise + a1 * (
                    0.5 * np.sin(2.0 * iter) ** 2 * np.cos(10.1 * iter) + 0.9 * np.sin(1.102 * iter) ** 2 * np.cos(
                4.001 * iter) + 0.3 * np.sin(1.99 * iter) ** 2 * np.cos(7 * iter) +
                    0.3 * np.sin(10.0 * iter) ** 3 + 0.7 * np.sin(3.0 * iter) ** 2 * np.cos(
                4.0 * iter) + 0.3 * np.sin(3.00 * iter) * np.cos(1.2 * iter) ** 2 +
                    0.4 * np.sin(1.12 * iter) ** 2 + 0.5 * np.cos(2.4 * iter) * np.sin(
                8 * iter) ** 2 + 0.3 * np.sin(1.000 * iter) * np.cos(0.799999 * iter) ** 2 +
                    0.3 * np.sin(4 * iter) ** 3 + 0.4 * np.cos(2 * iter) * np.sin(5 * iter) ** 4 + 0.3 * np.sin(
                10.00 * iter) ** 3) + a2 * (0.1 * np.sin(2 * iter) ** 3 * np.cos(9 * iter) +
                                            0.37 * np.sin(1.1 * iter) ** 2 * np.cos(4.00 * iter) + 0.3 * np.sin(
                        2.2 * iter) ** 4 * np.cos(7. * iter) + 0.3 * np.sin(10.3 * iter) ** 2 +
                                            0.7 * np.sin(3 * iter) ** 2 * np.cos(4 * iter) + 0.3 * np.sin(
                        3 * iter) * np.cos(
                        1.2 * iter) ** 2 + 0.4 * np.sin(1.12 * iter) ** 3 + 0.5 * np.cos(2.4 * iter) * np.sin(
                        8 * iter) ** 2 +
                                            0.3 * np.sin(1 * iter) * np.cos(0.8 * iter) ** 2 + 0.3 * np.sin(
                        4 * iter) ** 3 + 0.4 * np.cos(2 * iter) * np.sin(5 * iter) ** 4 + 0.3 * np.sin(5 * iter) ** 5)

        # if iter % 2:
        if iter < n_explore_step:
            delta_u = -np.dot(K, e_extend) + 0.3 * n_explor
        else:
            delta_u = -np.dot(K, e_extend)

        # 使用策略网络选择动作
        u_ff = 1*actor.act(state, device=config['device'])  # 确定性策略
        u_fb = delta_u.T
        # print(u_fb)
        u = 1*u_ff+0*u_fb
        u = np.clip(u, 0, 1).flatten()

        u_input = np.clip(3.00 * u, 0, 2)
        buffer = struct.pack("dddddddddddd", u_input[0], u_input[1], u_input[2], u_input[3],
                             u_input[4], u_input[5], u_input[6], u_input[7],
                             u_input[8], u_input[9], u_input[10], u_input[11])  # 设置控制器输入
        write_len = ser.write(buffer)

        q = q + t_s * e_lift[:ny, :]
        Z[0:nr, iter] = e_extend_former.flatten()
        Z[nr:nr + mr, iter] = u_former.flatten()
        e_extend_next = e_extend

        # for iteration in range(N - 2):
        #     upsilon[iteration] = upsilon[iteration + 1]
        #     phi[:, iteration] = phi[:, iteration + 1]
        upsilon[:-1] = upsilon[1:]
        phi[:, :-1] = phi[:, 1:]
        z_j = vec_z(Z[:, iter])
        phi[:, -1] = z_j[:, 0]
        xa_k_1 = e_extend_next
        delta_u_k_1 = -np.dot(K, xa_k_1)  # 2*1
        z_k_1 = vec_z(np.concatenate((xa_k_1, delta_u_k_1)))  # xxx*1

        reward = (np.dot(np.dot(Z[0:nr, iter].T, Q), Z[0:nr, iter]) +
                  np.dot(np.dot(Z[nr:nr + mr, iter].T, R), Z[nr:nr + mr, iter]))
        upsilon[-1] = reward + gamma * np.dot(H_vec.T, z_k_1)
        #
        if (iter > alpha) & (np.mod(iter, update_j) == 0) & (iter < n_learning_step):  # 40
            X_batch = upsilon[-batch_size:].reshape(-1, 1)
            U_batch = phi[:, -batch_size:]
            U_pred_batch = np.dot(U_batch.T, H_vec)  # batch_size*1
            loss = (1 / (2 * batch_size)) * np.linalg.norm(U_pred_batch - X_batch)
            gradiant = (1 / batch_size) * np.dot(U_batch, U_pred_batch - X_batch)  # size(H_vec)*1
            m_hat = beta1 * m_hat + (1 - beta1) * gradiant
            v_hat = beta2 * v_hat + (1 - beta2) * np.square(gradiant)
            m_hat_bias_corrected = m_hat / (1 - beta1 ** (Kerroriter + 1))
            v_hat_bias_corrected = v_hat / (1 - beta2 ** (Kerroriter + 1))
            H_vec = H_vec - learning_rate * m_hat_bias_corrected / (np.sqrt(v_hat_bias_corrected) + epsilon)
            # H, H_vec, K_new, Pk = estimate_policy(phi, upsilon, nr, mr, L, Pk, H_vec)
            H = vec_H_inv(H_vec)
            H_uu = H[nr:mr + nr, nr:mr + nr]
            H_ux = H[nr:mr + nr, 0:nr]
            K_new = np.dot(np.linalg.inv(H_uu), H_ux)
            print('K_error at', iter, 'step is', np.linalg.norm(K - K_new))
            K_error[Kerroriter] = np.linalg.norm(K - K_new)
            Kerroriter = Kerroriter + 1
            # K_error = np.concatenate((K_error,np.linalg.norm(K-K_new)))
            K = K_new

        e_extend_former = e_extend
        u_former = delta_u.reshape(-1, 1)
        e = x - x_d

        data['delta_u'][iter] = delta_u.flatten()
        data['x_lift'][iter] = x_lift.flatten()
        data['e_extend'][iter] = e_extend.flatten()
        data['e'][iter] = e.flatten()
        data['state'][iter] = x.T.flatten()
        data['action'][iter] = u.flatten()

        toc = time.time()
        # print(toc - tic)
        time.sleep(np.clip(t_s - (toc - tic), 0, t_s))

    # ===========================
    # 保存数据
    # ===========================
    data['K'] = K
    data['H'] = H
    save_path = 'control_data/result.mat'
    scipy.io.savemat(save_path, data)
    # load data and plot
    result = scipy.io.loadmat('control_data/result.mat')
    e = result['e']
    u = result['action']
    x = result['state']
    buffer = struct.pack("dddddddddddd", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # 设置控制器输入置零
    write_len = ser.write(buffer)
    ser.close()
    if ser.isOpen():  # 判断串口是否关闭
        print("串口未关闭。")
    else:
        print("串口已关闭。")
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    axs[0].plot(e[:, :3])
    axs[0].set_title("ex-ey-ez")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("e")
    eytick1 = axs[0].get_ylim()
    axs[0].text(sim_step / 2, 0.6 * eytick1[0], 'e_x:%f' % (e[-1, 0]), fontdict={'color': 'b', 'size': '12'})
    axs[0].text(sim_step / 2, 0.8 * eytick1[0], 'e_y:%f' % (e[-1, 1]), fontdict={'color': 'b', 'size': '12'})
    axs[0].text(sim_step / 2, 1 * eytick1[0], 'e_z:%f' % (e[-1, 2]), fontdict={'color': 'b', 'size': '12'})
    print('ex', e[-1, 0])
    print('ey', e[-1, 1])
    print('ez', e[-1, 2])

    axs[1].plot(e[:, 3:6])
    axs[1].set_title("e-angle")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("e")
    eytick2 = axs[1].get_ylim()
    axs[1].text(sim_step / 2, 0.6 * eytick2[0], 'e_anx:%f' % (e[-1, 3]), fontdict={'color': 'b', 'size': '12'})
    axs[1].text(sim_step / 2, 0.8 * eytick2[0], 'e_any:%f' % (e[-1, 4]), fontdict={'color': 'b', 'size': '12'})
    axs[1].text(sim_step / 2, 1 * eytick2[0], 'e_anz:%f' % (e[-1, 5]), fontdict={'color': 'b', 'size': '12'})
    print('etheta1', e[-1, 3])
    print('etheta2', e[-1, 4])
    print('etheta3', e[-1, 5])

    axs[2].plot(u[:, :])
    axs[2].set_title("u1-u8")
    axs[2].set_xlabel("step")
    axs[2].set_ylabel("u")
    print('u: ', u[-1, :])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
