"""
The learned A is contructed from eigenvalues
"""
import argparse
from pprint import pprint
# from scipy.linalg import matrix_rank
import scipy.io as scio
import control
import torch
import pickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import scipy.io
import time
import LuMoSDKClient
import serial
import struct

import sys
sys.path.append('Training')
import Learning_Koopman_with_Reg_HPN as LK


class LQR:
    def __init__(self, A, B, Q, R, space_dim):
        self.A = A  # 12*1 # just eigenvalue
        self.B = B  # 24*8
        self.Q = Q
        self.R = R

    def lqr(self):
        K, S, E = control.dlqr(self.A, self.B, self.Q,
                               self.R)  # K (2D array (or matrix)) – State feedback gains;K (2D array (or matrix)) – State feedback gains;E (1D array) – Eigenvalues of the closed loop system
        print('K', K.shape)
        return K


def form_A_from_eigenvalues(space_dim, num_real, num_complex_pair, A):
    idx = 0
    temp_A = np.zeros([space_dim, space_dim])
    for i in range(num_complex_pair):
        idx = 2 * i
        temp_A[idx:idx + 2, idx:idx + 2] = form_complex_conjugate_block(A[idx], A[idx + 1])
    for i in range(num_real):
        idx = 2 * num_complex_pair + i
        temp_A[idx, idx] = A[idx]
    # print('temp_A',temp_A)
    return temp_A


def form_complex_conjugate_block(real, imaginary):
    block = np.zeros([2, 2])
    block[0, 0] = real
    block[0, 1] = imaginary
    block[1, 0] = -imaginary
    block[1, 1] = real
    return block


def is_controllable(A, B):
    n = A.shape[0]
    m = B.shape[1]
    # 构造可控性矩阵
    controllability_matrix = np.zeros((n, n * m))
    # print('controllability_matrix_shape',controllability_matrix.shape)
    for i in range(n):
        temp = np.linalg.matrix_power(A, i) @ B
        # print('np.linalg.matrix_power(A, i) @ B',temp.shape)
        controllability_matrix[:, i * m:(i + 1) * m] = np.linalg.matrix_power(A, i + 1) @ B
    # 判断可控性矩阵的秩
    rank = np.linalg.matrix_rank(controllability_matrix)
    # 判断系统是否可控
    if rank == n:
        return True
    else:
        return False


def normalization_state(original_data, min_vals, max_vals):
    num_rows = original_data.shape[0]
    normalized_data = np.zeros_like(original_data)
    for i in range(num_rows):
        normalized_data[i, :] = 2 * (original_data[i, :] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1
    return normalized_data


def plot_unit_circle(A):
    eigenvalues, _ = np.linalg.eig(A)  # 计算 A 矩阵的特征值
    # 绘制单位圆
    unit_circle = plt.Circle((0, 0), 1, color='k', fill=False)
    plt.gca().add_patch(unit_circle)
    # 绘制特征值
    for eigenvalue in eigenvalues:
        plt.plot(np.real(eigenvalue), np.imag(eigenvalue), 'ro')
    # 设置坐标轴范围
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    # 设置网格线
    plt.grid(True)
    # 显示图形
    plt.show()


static_koopman = scipy.io.loadmat('Static_Koopman_G/G_matrix.mat')
static_koopman_G = static_koopman['G']
# static_koopman_G = np.zeros((12, 24))
# print('static_koopman_G:', static_koopman_G)

Model_NAME = "Train_Koopman_ES4-ES3-ES2"
MODEL_DIR = os.path.join("Training/Models", Model_NAME)

# 状态空间维度
n_x = 12
n_u = 12
n_lift = 12
space_dim = n_x + n_lift
nr = space_dim
t_s = 0.02  # 0.01 0.03
sim_step = 500  # 3000

# load network
suffix = MODEL_DIR + "/" + Model_NAME
subsuffix = suffix + "_" + "layer{}_edim{}".format(3, 12)
# print(subsuffix)
dicts = torch.load(subsuffix + ".pth", map_location=torch.device('cpu'))
state_dict = dicts["model"]
Elayer = dicts["layer"]
# print(layers)
net = LK.Network(Elayer, space_dim, n_u)
net.load_state_dict(state_dict)
device = torch.device("cpu")
net.cpu()
net.float()

A = net.A.detach().numpy()
B = net.B.detach().numpy()

# 定义性能指标权重矩阵
Q_temp_1 = 2 * np.array([[2, 2, 2]])
Q_temp_2 = 1 * np.ones((1, 3))
Q_temp_3 = 0.5 * np.ones((1, 3))
Q_temp_4 = 0.5 * np.ones((1, 3))
Q_temp_5 = 1 * np.ones((1, 12))
Q_temp = np.concatenate((Q_temp_1, Q_temp_2, Q_temp_3, Q_temp_4, Q_temp_5), axis=1)
Q = np.diag(Q_temp.flatten())
R = (10 * np.eye(n_u, n_u))  # 20

num_real = int(np.mod(space_dim, 2))
num_complex_pair = int(space_dim / 2)
temp_A = form_A_from_eigenvalues(space_dim, num_real, num_complex_pair, A)
A = temp_A.T
B = B.T
controller = LQR(A, B, Q, R, space_dim=space_dim)
K = controller.lqr()

# initial state
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
# 1
# u_ref = np.array([[0.1, 0.1, 0.45, 0.45, 0.08, 0.08, 0.35, 0.35,0.08, 0.08, 0.45, 0.45]])
# x_d = np.array([[0.15860529,  0.31405063,  0.0853791,   0.03018992,  0.02252583, -0.94518057, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)
# 2
u_ref = np.array([[0.38, 0.45, 0.05, 0.08, 0.48, 0.28, 0.15, 0.15,0.48, 0.48, 0.1, 0.1]])
x_d = np.array([[-8.93930511e-02,  3.00755615e-01,  8.93038254e-02,  8.90345103e-02,  -2.49624019e-02,  1.30454680e+00, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)
# # 3
# u_ref = np.array([[0.08, 0.45, 0.36, 0.08, 0.08, 0.28, 0.35, 0.08,0.1, 0.35, 0.25, 0.05]])
# x_d = np.array([[1.51007633e-02,  3.20245819e-01, -7.51049280e-03, -7.31499045e-01, -1.52485430e-01,  5.06724421e-01, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)
# # 4
# u_ref = np.array([[0.38, 0.15, 0.16, 0.38, 0.48, 0.18, 0.15, 0.48,0.45, 0.06, 0.04, 0.45]])
# x_d = np.array([[2.14985409e-02,  3.29598022e-01,  1.66807404e-01,  8.78652344e-01, 1.41118476e-01,  3.69111730e-01, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)
# # 5
# u_ref = np.array([[0.2, 0.05, 0.2, 0.4, 0.18, 0.08, 0.15, 0.4, 0.3, 0.06, 0.15, 0.45]])
# x_d = np.array([[0.0952269,   0.32751242,  0.1475923,   0.62210338, -0.15754589, -0.30275514, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)

min_vals = np.array(
    [-0.170207382202148,	0.245196914672852,	-0.0967208938598633,	-1.38428982933447,	-1.10439313008477,	-1.28774131822944,	-1.49457599639893,	-0.819060058593750,	-1.15087868690491,	-8.71387673543985,	-4.31210097148843,	-8.71643611647120])
max_vals = np.array(
    [0.232169052124023,	0.393803253173828,	0.235029663085938,	1.14079474718160,	0.874471847112893,	1.87626156961726,	1.53390144348145,	0.720140380859375,	1.07798648834229,	7.05122764975299,	6.13566299399358,	9.37093483564043])

# save data
data = {
    'delta_u': np.zeros((sim_step, n_u)),
    'x_lift': np.zeros((sim_step, space_dim)),
    'e_lift': np.zeros((sim_step, space_dim)),
    'e': np.zeros((sim_step, n_x)),  # 时间
    'state': np.zeros((sim_step, n_x)),  # 状态 (位置和速度)
    'action': np.zeros((sim_step, n_u)),  # 控制输入
    'K': K,
    'G': static_koopman_G
}

x_d_normalization = normalization_state(x_d, min_vals, max_vals)
x_d_normalization = torch.tensor(x_d_normalization, dtype=torch.float32)
x_d_lift_temp = net.encode_only(x_d_normalization.T)
x_d_lift = torch.hstack((x_d_normalization.T, x_d_lift_temp)).detach().numpy().T

x_pos_list = np.zeros((3, 1))
x_ang_radian_list = np.zeros((3, 1))
sensor_data = []

#
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
x_ang_radian_list = x_ang_radian
x_pos_vel = sensor_data[3:6].reshape(3, 1)
x_ang_radian_vel = np.array([0, 0, 0]).reshape(3, 1)

# mat = scipy.io.loadmat("control_data/result.mat")
# K = mat['K']

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
    x = np.concatenate(
        (x_pos.reshape(1, 3), x_ang_radian.reshape(1, 3), x_pos_vel.reshape(1, 3), x_ang_radian_vel.reshape(1, 3)),
        axis=1).T

    x_normalization = normalization_state(x, min_vals, max_vals)
    x_normalization = torch.tensor(x_normalization, dtype=torch.float32)
    x_lift_temp = net.encode_only(x_normalization.T)
    x_lift = torch.hstack((x_normalization.T, x_lift_temp)).detach().numpy().T

    e_lift = x_lift - x_d_lift

    delta_u = -np.dot(K, e_lift)
    static_u = np.dot(static_koopman_G, x_d_lift)

    u = 1*delta_u.flatten() + 1*static_u.flatten() + 0*u_ref.flatten()
    u = np.clip(u, 0, 1).flatten()

    u_input = np.clip(3.00 * u, 0, 3)
    buffer = struct.pack("dddddddddddd", u_input[0], u_input[1], u_input[2], u_input[3],
                         u_input[4], u_input[5], u_input[6], u_input[7],
                         u_input[8], u_input[9], u_input[10], u_input[11])  # 设置控制器输入
    write_len = ser.write(buffer)

    e = x - x_d

    data['delta_u'][iter] = delta_u.flatten()
    data['x_lift'][iter] = x_lift.flatten()
    data['e_lift'][iter] = e_lift.flatten()
    data['e'][iter] = e.flatten()
    data['state'][iter] = x.T.flatten()
    data['action'][iter] = u.flatten()

    toc = time.time()
    time.sleep(np.clip(t_s - (toc - tic), 0, t_s))

save_path = 'control_data/result.mat'
scipy.io.savemat(save_path, data)
# load data and plot
result = scio.loadmat('control_data/result.mat')
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

print(x[-1],u[-1])

axs[0].plot(e[:, :3])
axs[0].set_title("ex-ey-ez")
axs[0].set_xlabel("step")
axs[0].set_ylabel("e")
eytick1 = axs[0].get_ylim()
axs[0].text(sim_step/2,0.6*eytick1[0],'e_x:%f'%(e[-1,0]),fontdict={'color':'b','size':'12'})
axs[0].text(sim_step/2,0.8*eytick1[0],'e_y:%f'%(e[-1,1]),fontdict={'color':'b','size':'12'})
axs[0].text(sim_step/2,1*eytick1[0],'e_z:%f'%(e[-1,2]),fontdict={'color':'b','size':'12'})
print('ex', e[-1, 0])
print('ey', e[-1, 1])
print('ez', e[-1, 2])

axs[1].plot(e[:, 3:6])
axs[1].set_title("e-angle")
axs[1].set_xlabel("step")
axs[1].set_ylabel("e")
eytick2 = axs[1].get_ylim()
axs[1].text(sim_step/2,0.6*eytick2[0],'e_anx:%f'%(e[-1,3]),fontdict={'color':'b','size':'12'})
axs[1].text(sim_step/2,0.8*eytick2[0],'e_any:%f'%(e[-1,4]),fontdict={'color':'b','size':'12'})
axs[1].text(sim_step/2,1*eytick2[0],'e_anz:%f'%(e[-1,5]),fontdict={'color':'b','size':'12'})
print('etheta1', e[-1, 3])
print('etheta2', e[-1, 4])
print('etheta3', e[-1, 5])

axs[2].plot(u[:, :])
axs[2].set_title("u1-u8")
axs[2].set_xlabel("step")
axs[2].set_ylabel("u")
eytick3 = axs[2].get_ylim()
axs[2].text(sim_step/2,0.95*eytick3[1],'u1:%f'%(u[-1,0]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.85*eytick3[1],'u2:%f'%(u[-1,1]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.75*eytick3[1],'u3:%f'%(u[-1,2]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.65*eytick3[1],'u4:%f'%(u[-1,3]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.55*eytick3[1],'u1:%f'%(u[-1,4]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.45*eytick3[1],'u2:%f'%(u[-1,5]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.35*eytick3[1],'u3:%f'%(u[-1,6]),fontdict={'color':'b','size':'12'})
axs[2].text(sim_step/2,0.25*eytick3[1],'u4:%f'%(u[-1,7]),fontdict={'color':'b','size':'12'})

plt.tight_layout()
plt.show()
