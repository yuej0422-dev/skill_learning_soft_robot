"""
The learned A is contructed from eigenvalues
"""
import argparse
from pprint import pprint
import numpy as np
# from scipy.linalg import matrix_rank
import scipy.io as scio
import control
import torch
from DDK_softR302_one_seg1 import DeepEDMD
from jy_learning_static_u_seg_HPN_1 import Network as U_net
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


def value_iteration(Z, n, m, N, L, H_vec, Q, R, Kn, K):  # in lifted space
    # Z data matrix
    # n number of states
    # m number of inputs
    # N number of measurements
    # L size of z-vector
    # K policy at current update step
    # Q state cost matrix
    # R input cost matrix
    phi = np.zeros((L, N - 1))
    upsilon = np.zeros((N - 1, 1))
    for k in range(N - 1):
        z_k = vec_z(Z[:, k])
        phi[:, k] = z_k
        e_k_1 = Z[0:n, k + 1]
        delta_u_k_1 = -np.dot(K, e_k_1)
        z_k_1 = vec_z(np.concatenate((e_k_1, delta_u_k_1)))
        e_k = Z[0:n, k]
        delta_u_k = Z[n + 1:n + m, k]
        upsilon[k] = np.dot(np.dot(e_k.T, Q), e_k) + np.dot(np.dot(delta_u_k.T, R), delta_u_k) + np.dot(H_vec.T, z_k_1)
    return phi, upsilon


def estimate_policy(phi, upsilon, nr, mr, L, Pk, H_vec):
    # for j in range(1):  # L-1
    Y_vec = phi[:, -1].reshape(-1, 1)
    d = upsilon[-1]
    denomi = 1 + np.dot(Y_vec.T, np.dot(Pk, Y_vec))
    Hnume = np.dot(Pk, np.dot(Y_vec, (d - np.dot(Y_vec.T, H_vec))))
    Pnume = np.dot(Pk, np.dot(Y_vec, np.dot(Y_vec.T, Pk)))
    H_vec = H_vec + Hnume / float(denomi)
    Pk = Pk - Pnume / float(denomi)
    H = vec_H_inv(H_vec)
    return H, H_vec, Pk


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


# def predict_u(state_des, net_control, u_dim):
#     state_des = state_des[:6, :]
#
#     min_vals = np.array([-1., -1., -1., -1., -0.01180667, 0.18211005, 0.0177896, -0.67733394, -0.26274892, -1.04744356])
#     max_vals = np.array([1., 1., 1., 1., 0.12378424, 0.23994904, 0.1399321, 0.99112143, 0.2101465, 0.79459266])
#     min_vals = np.array([-1., -1., -1., -1., -0.01783669, 0.18211005, 0.0177896, -0.67733394, -0.26274892, -1.04744356])
#     max_vals = np.array([1., 1., 1., 1., 0.12378424, 0.24028485, 0.1399321, 0.99112143, 0.2101465, 0.86723318])
#     state_des_normalized = normalization_state(state_des, min_vals[u_dim:], max_vals[u_dim:]).T
#     control0_hat = net_control.encode_net(torch.from_numpy(state_des_normalized).float()).detach().numpy()
#     return control0_hat
#
# # load control_network
# model_path = "HPN_one_v2/HPN_one_v2_with_Regularization_layer4.pth"
# dicts_control = torch.load(model_path, map_location=torch.device('cpu'))
# state_dict_control = dicts_control["model"]
# Elayer_control = dicts_control["layer"]
# net_control = U_net(Elayer_control)
# net_control.load_state_dict(state_dict_control)
# # print(net_control)
# static_koopman_G = scipy.io.loadmat('G_matrix_2111.mat')
# static_koopman_G = static_koopman_G['G']

# 读取模型参数
with open('SoftRobot_2025_02_16_15_04/args.pkl', 'rb') as f:
    args_dict = pickle.load(f)
args = argparse.Namespace(**args_dict)
args = vars(args)

dck = DeepEDMD.load_from_checkpoint(checkpoint_path='SoftRobot_2025_02_16_15_04/logdir/epoch=249-train_loss=0.03288.ckpt', save_params=False, args=args)

koopman_net = dck.koopman_net
koopman_net.eval()
encoder = dck.encoder_net
encoder.eval()

A = koopman_net.A.detach().numpy()
B = koopman_net.B.detach().numpy()
T = koopman_net.T.detach().numpy().T  # 12*12

# 状态空间维度
n_x = 12
n_u = 12
n_lift = 12
space_dim = n_x + n_lift
m = n_u
ny = 0
nr = space_dim + ny
mr = m
t_s = 0.02  # 0.01 0.03
sim_step = 500  # 3000
n_learning_step = 0
n_explore_step = 0  # 1000
learning_rate = 0.0001  # 0.0005
z_size = nr + m  # Number of states in z-vector
L = int(z_size * (z_size + 1) / 2)  # /Size of vectorized z
N = L + 5  # 1* L  Number of measurements before evaluation % 10* L;
# K =  zeros(m, n) # the starting point of the learning process
beta = 0.1  # parameter for RLS
Pk = beta * np.eye(L)  # parameter for RLS
gamma = 0.98  # 0.95

# 定义性能指标权重矩阵
Q_temp_1 = 2 * np.array([[2, 2, 2]])  # 2.5
Q_temp_2 = 1 * np.ones((1, 3))  # 2.5
Q_temp_3 = 0.5 * np.ones((1, 6)) # 2.5
Q_temp_4 = 1 * np.ones((1, 12)) # 2.5
Q_temp_5 = 0 * np.ones((1, ny)) # 0.2
# Q_temp = np.concatenate((Q_temp_1, Q_temp_2, Q_temp_3, Q_temp_4), axis=1)
Q_temp = np.concatenate((Q_temp_1, Q_temp_2, Q_temp_3, Q_temp_4, Q_temp_5), axis=1)
# print('Q_temp',Q_temp)
Q = np.diag(Q_temp.flatten())
R = (10 * np.eye(4, 4))  # 20

num_real = int(np.mod(space_dim, 2))
num_complex_pair = int(space_dim / 2)
temp_A = form_A_from_eigenvalues(space_dim, num_real, num_complex_pair, A)
C_f = np.concatenate((np.linalg.inv(T), np.zeros((12, 12))), axis=1)
C = C_f[0:ny, :]

At = np.block([
    [temp_A.T, np.zeros((temp_A.shape[0], ny))],
    [t_s * C, np.eye(ny)]
])
tb = B.T
Bt = np.concatenate((B.T, np.zeros((ny, 4))))
controller = LQR(At, Bt, Q, R, space_dim=space_dim)
q = np.zeros((ny, 1))
Gm = np.block([
    [Q, np.zeros((Q.shape[0], R.shape[0]))],
    [np.zeros((R.shape[0], Q.shape[0])), R]
])  # 29*29
H1 = 10 * np.eye(nr + m)  # 29*29
H1yy = H1[nr:nr + mr, nr:nr + mr]  # 2*2
H1yx = H1[nr:nr + mr, 0:nr]  # 2*27
L1 = -np.dot(np.linalg.inv(H1yy), H1yx)  # 2*27
for i in range(8000):
    T12 = np.concatenate((At, Bt), axis=1)  # 27*29
    T34 = np.concatenate((np.dot(L1, At), np.dot(L1, Bt)), axis=1)  # 2*29
    Trans = np.concatenate((T12, T34))  # 29*29
    H1 = Gm + gamma * np.dot(np.dot(Trans.T, H1), Trans)
    H1yy = H1[nr:nr + mr, nr:nr + mr]
    H1yx = H1[nr:nr + mr, 0:nr]
    L1 = -np.dot(np.linalg.inv(H1yy), H1yx)  # 2*27
Kn = -L1  # offline solution
H = H1
K = Kn
H_vec = vec_H(H)  # Vectorized value function matrix


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

u_ref = np.array([[0.1, 0.1, 0.45, 0.45, 0.08, 0.08, 0.35, 0.35,0.08, 0.08, 0.45, 0.45]])
x_d = np.array([[1.59225388e-01,  3.14065582e-01,  8.27972794e-02, -2.14799468e-03, 5.26162110e-02, -9.62383327e-01, 0, 0, 0, 0, 0, 0]]).reshape(-1,1)

min_vals = np.array(
    [-0.170207382202148,	0.245196914672852,	-0.0967208938598633,	-1.38428982933447,	-1.10439313008477,	-1.28774131822944,	-1.49457599639893,	-0.819060058593750,	-1.15087868690491,	-8.71387673543985,	-4.31210097148843,	-8.71643611647120])
max_vals = np.array(
    [0.232169052124023,	0.393803253173828,	0.235029663085938,	1.14079474718160,	0.874471847112893,	1.87626156961726,	1.53390144348145,	0.720140380859375,	1.07798648834229,	7.05122764975299,	6.13566299399358,	9.37093483564043])

# save data
data = {
    'delta_u': np.zeros((sim_step, n_u)),
    'x_lift': np.zeros((sim_step, space_dim)),
    'e_extend': np.zeros((sim_step, nr)),
    'e': np.zeros((sim_step, n_x)),  # 时间
    'state': np.zeros((sim_step, n_x)),  # 状态 (位置和速度)
    'action': np.zeros((sim_step, n_u)),  # 控制输入
    'K': np.zeros((K.shape[0], K.shape[1])),
    'H': np.zeros((H.shape[0], H.shape[1]))
}

x_d_normlization = normalization_state(x_d, min_vals, max_vals)
x_d_normlization = torch.tensor(x_d_normlization.T, dtype=torch.float32)
T = dck.koopman_net.T
x_d_temp = torch.mm(x_d_normlization, T)
x_d_lift_temp = dck.encoder_net(torch.mm(x_d_normlization, T))
x_d_lift = torch.hstack((x_d_temp, x_d_lift_temp)).T
# print('x_d_lift',x_d_lift.shape)
x_d_lift = x_d_lift.detach().numpy()
# print(x_d_lift.shape)


x_pos_list = np.zeros((3, 1))  # x_0[:3,]
x_ang_radian_list = np.zeros((3, 1))  # quaternion_to_euler(x_0[4:7,], 'sxyz').reshape(3,1)
phi = np.zeros((L, N - 1))
upsilon = np.zeros(N - 1)
Z = np.zeros((nr + m, sim_step))
update_j = 25  # 5
alpha = 50  # 50
K_error = np.zeros(int((sim_step - alpha) / update_j))
Kerroriter = 0
batch_size = 50  # 5
learning_rate = 1 * learning_rate
epsilon = 1e-8
beta1 = 0.9  # 0.9
beta2 = 0.999  # 0.999
m_hat = np.zeros(H_vec.shape[0]).reshape(-1, 1)
v_hat = np.zeros(H_vec.shape[0]).reshape(-1, 1)
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
# x_ang_radian_vel = (x_ang_radian_list[:, -1] - x_ang_radian_list[:, -2]) / 0.02
x0 = np.concatenate(
    (x_pos.reshape(1, 3), x_ang_radian.reshape(1, 3), x_pos_vel.reshape(1, 3), x_ang_radian_vel.reshape(1, 3)),
    axis=1).T
e_extend = np.zeros((nr,1))
delta_u = np.zeros((m,1))
e_temp = np.zeros((24,1))

x_normlization = normalization_state(x0, min_vals, max_vals)
x_normlization = torch.tensor(x_normlization.T, dtype=torch.float32)
x_temp = torch.mm(x_normlization, T)
x_lift_temp = dck.encoder_net(torch.mm(x_normlization, T))
# print('x_d_lift_temp.shape',x_d_lift_temp.shape) # torch.Size([1, 12])
x_lift = torch.hstack((x_temp, x_lift_temp)).T
x_lift = x_lift.detach().numpy()

e_lift = x_lift - x_d_lift
e_temp = np.array(e_lift).reshape(-1, 1)
e_extend = np.concatenate((e_temp, q))

e_extend_former = e_extend
u_former = delta_u
# q = q + t_s * np.dot(C, e_temp)

# # mat = scipy.io.loadmat("control_data/randomH_LQR_result.mat")
# # mat = scipy.io.loadmat("control_data/LB_diagH_1-2.mat")
# mat = scipy.io.loadmat("control_data/LB_LQRint_mbH_1-1.mat")
# mat = scipy.io.loadmat("control_data/result.mat")
# K = mat['K']
# H = mat['H']
# H_vec = vec_H(H)

K0 = K
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

    x_normlization = normalization_state(x, min_vals, max_vals)
    x_normlization = torch.tensor(x_normlization.T, dtype=torch.float32)
    x_temp = torch.mm(x_normlization, T)
    x_lift_temp = dck.encoder_net(torch.mm(x_normlization, T))
    # print('x_d_lift_temp.shape',x_d_lift_temp.shape) # torch.Size([1, 12])
    x_lift = torch.hstack((x_temp, x_lift_temp)).T
    x_lift = x_lift.detach().numpy()

    e_lift = x_lift - x_d_lift
    e_temp = np.array(e_lift).reshape(-1, 1)
    e_extend = np.concatenate((e_temp, q))

    # enforce control limit
    a1 = 0.001 * np.ones((m, 1))
    a2 = 0.001 * np.ones((m, 1))
    sim_noise = 0.0001 * np.random.normal(loc=0, scale=1000, size=(m, 1))
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
                1.2 * iter) ** 2 + 0.4 * np.sin(1.12 * iter) ** 3 + 0.5 * np.cos(2.4 * iter) * np.sin(8 * iter) ** 2 +
                                    0.3 * np.sin(1 * iter) * np.cos(0.8 * iter) ** 2 + 0.3 * np.sin(
                4 * iter) ** 3 + 0.4 * np.cos(2 * iter) * np.sin(5 * iter) ** 4 + 0.3 * np.sin(5 * iter) ** 5)

    # if iter % 2:
    if iter < n_explore_step:
        delta_u = -np.dot(K, e_extend) + 0.5 * n_explor
    else:
        delta_u = -np.dot(K, e_extend)

    # u_pred = predict_u(x_d, net_control, n_u)
    # u_pred = np.dot(static_koopman_G, x_d_lift)
    u = 1 * delta_u.flatten() + 0 * u_ref.flatten()
    u = np.clip(u, 0, 1).flatten()

    u_input = np.clip(3.00 * u, 0, 3)
    buffer = struct.pack("dddddddddddd", u_input[0], u_input[1], u_input[2], u_input[3],
                         u_input[4], u_input[5], u_input[6], u_input[7],
                         u_input[8], u_input[9], u_input[10], u_input[11])  # 设置控制器输入
    write_len = ser.write(buffer)

    q = q + t_s * np.dot(C, e_temp)

    Z[0:nr, iter] = e_extend_former.flatten()
    Z[nr:nr + m, iter] = u_former.flatten()
    e_extend_next = e_extend

    for iteration in range(N - 2):
        upsilon[iteration] = upsilon[iteration + 1]
        phi[:, iteration] = phi[:, iteration + 1]
    z_j = vec_z(Z[:, iter])
    phi[:, -1] = z_j[:, 0]
    xa_k_1 = e_extend_next
    delta_u_k_1 = -np.dot(K, xa_k_1)  # 2*1
    z_k_1 = vec_z(np.concatenate((xa_k_1, delta_u_k_1)))  # xxx*1

    reward = (np.dot(np.dot(Z[0:nr, iter].T, Q), Z[0:nr, iter]) +
              np.dot(np.dot(Z[nr:nr + m, iter].T, R), Z[nr:nr + m, iter]))
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
        m_hat_bias_corrected = m_hat / (1 - beta1 ** (iter - alpha))
        v_hat_bias_corrected = v_hat / (1 - beta2 ** (iter - alpha))
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
    time.sleep(np.clip(t_s - (toc - tic), 0, t_s))

print('K0',K0)
print('K',K)
data['K'] = K
data['H'] = H
data['Z'] = Z
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
