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
from cql_kop_Async_Qlinear import (KoopmanFeedforwardPolicy, KoopmanEncoder,
                                   KoopmanInformedQFunction, Scalar, ReparameterizedSigmGaussian)


# ===========================
# 仿真和控制逻辑
# ===========================

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
    # MODEL_DIR = "D:/controldesk/zxl/pycharm_project/CORL-main/checkpoints/0612/KORL_cql-HPN_test-4b116b4f"
    MODEL_DIR = "D:/controldesk/zxl/pycharm_project/CORL-main/checkpoints/0716/BL-RS-WL/KORL_cql_HPN_v1/mixed/KORL_CQL-HPN-20976e3a"  # 替换为您的模型目录
    ACTOR_MODEL_NAME = "checkpoint_30000.pt"
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

    # ===========================
    # 仿真参数和数据初始化
    # ===========================
    # 状态空间维度
    n_x = 12
    n_u = 12
    sim_step = 1000
    t_s = 0.02

    min_vals = np.array(
        [-0.218786819458008,    0.287727661132813,  -0.206123138427734, -1.56453377033352,  -12.2726242790186,  -2.16422318617507,  -0.918451709747314, -0.531738281250000  ,-0.708585796356201,    -3.10487565164087,  -27.7607455963523,  -19.7275968039865])
    max_vals = np.array(
        [0.336474945068359, 0.593327209472656,  0.325578460693359,  1.47627275022471,   2.59086185912188,   12.3175993107442,   0.976459083557129,  0.410773010253906,  0.980089492797852   ,4.03057679785183,  19.6928578649373,   27.6077448279089])

    # 保存数据
    data = {
        'e': np.zeros((sim_step, n_x)),  # 误差
        'state': np.zeros((sim_step, n_x)),  # 状态 (位置和速度)
        'action': np.zeros((sim_step, n_u)),  # 控制输入
    }

    # 期望状态（根据需要修改）
    x_d = np.concatenate((
        # np.array([[2.55555145e-01,  4.40762573e-01,  7.09538574e-02, -4.05738043e-02, 3.03838197e-02, -1.20259953e+00]]),
        # np.array([[-1.47283371e-01,  4.32574036e-01,  5.64879761e-02, -1.35493938e-01, -2.57229686e-01,  1.46016009e+00]]),
        # np.array([[4.67764473e-02,  4.34208649e-01, -1.33101974e-01, -1.45836925e+00, -7.62699109e-01,  9.62717288e-01]]),
        np.array([[4.96281433e-02,  4.62189026e-01,  2.54127350e-01,  1.00733096e+00, -6.72313249e-03, -4.49814316e-02]]),
        np.zeros((1, 6))
    ), axis=1).T
    # u_ref = np.array([0.1, 0.1, 0.45, 0.45, 0.08, 0.08, 0.45, 0.45,0.08, 0.08, 0.35, 0.35])
    # u_ref = np.array([0.45, 0.45, 0.1, 0.1, 0.45, 0.45, 0.08, 0.08, 0.35, 0.35, 0.08, 0.08])
    # u_ref = np.array([0.1, 0.45, 0.45, 0.1, 0.05, 0.45, 0.38, 0.08, 0.05, 0.35,  0.48, 0.08])
    u_ref = np.array([0.45, 0.1, 0.1, 0.45, 0.45, 0.05, 0.08, 0.38, 0.35, 0.05, 0.08, 0.38])

    x_d_normalization = normalization_state(x_d, min_vals, max_vals)

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
        state = np.concatenate((x_normalization, x_d_normalization)).flatten()

        # 使用策略网络选择动作
        u_ff = 1*actor.act(state, device=config['device'])  # 确定性策略
        u_ff = np.array([0.43166626, 0.054645,         0.061438,         0.44595022, 0.47039063, 0.061128,
                         0.028792,         0.41395989, 0.40207461, 0.06218 ,        0.05212,         0.43271229])

        u_fb = 1 * critic_1_kop.act(state, device=config['device'])  # 确定性策略
        u = 1*u_ff+1*u_fb
        u = np.clip(u, 0, 1).flatten()

        u_input = np.clip(3.0 * u, 0, 3)
        buffer = struct.pack("dddddddddddd", u_input[0], u_input[1], u_input[2], u_input[3],
                             u_input[4], u_input[5], u_input[6], u_input[7],
                             u_input[8], u_input[9], u_input[10], u_input[11])  # 设置控制器输入
        write_len = ser.write(buffer)

        # 计算误差
        e = x - x_d

        data['e'][iter] = e.flatten()
        data['state'][iter] = x.flatten()
        data['action'][iter] = u.flatten()

        toc = time.time()
        time.sleep(np.clip(t_s - (toc - tic), 0, t_s))

    # ===========================
    # 保存数据
    # ===========================
    save_path = 'control_data/result.mat'
    scipy.io.savemat(save_path, data)
    scipy.io.savemat(r'D:\controldesk\zxl\pycharm_project\CORL-main\record-0706\BLRSWL-mix-sp\result.mat', data)
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
