# 去动捕 RGB-D 视觉本体状态恢复

## 1. 研究定位

该阶段解决的问题是：部署阶段不再访问动捕状态，仅使用主视角 RGB-D 多帧观测和历史控制，恢复软体机械臂当前本体状态，使后续状态技能或 VLA 仍能获得类似 proprioception 的输入。

第一阶段已经有成熟的状态空间控制和状态空间 Koopman encoder：

\[
s_t=
\begin{bmatrix}
p_t&r_t&v_t&\omega_t
\end{bmatrix}^{\top}
\in\mathbb R^{12}
\]

其中 \(p_t,r_t,v_t,\omega_t\) 分别为 TCP 位置、姿态、线速度和角速度。

本阶段的目标不是普通“图像到状态回归”，而是：

```text
RGB-D 三维几何重构
        ↓
多帧时序状态估计
        ↓
训练阶段动捕特权监督
        ↓
部署阶段无动捕本体状态恢复
```

## 2. 输入输出定义

部署输入为长度 \(K\) 的 RGB-D 历史和历史控制：

\[
\mathcal O_t=
\{
(I_{t-K+1:t},D_{t-K+1:t}),
u_{t-K+1:t-1}
\}
\]

其中 \(u\) 可以是历史 delta TCP、历史期望 TCP 命令或历史气压，具体取决于系统日志中最稳定、最同步的控制量。

模型输出：

\[
\hat s_t=
\begin{bmatrix}
\hat p_t&\hat r_t&\hat v_t&\hat\omega_t
\end{bmatrix}^{\top}
\]

可选输出视觉状态 latent：

\[
z_t^V
\]

若需要物体操作，也可同时估计物体状态：

\[
\hat o_t
\]

## 3. 数据采集与同步

训练阶段同步记录：

\[
\mathcal D_t=
\{
I_t,
D_t,
s_t^{mocap},
u_t,
pressure_t,
g_t,
\tau_t
\}
\]

如设备允许，还建议保存：

- 控制命令发送时间；
- 气压执行反馈；
- 动捕原始位姿；
- RGB-D 硬件时间戳；
- 物体动捕位姿；
- episode id 和 reset 边界。

时间同步非常关键。建议以 RGB-D 帧时间为基准，对动捕位姿插值，并明确 transition：

\[
(s_t,u_t,s_{t+1})
\]

避免误配为：

\[
(s_t,u_{t+1},s_{t+1})
\]

如果存在固定控制延迟 \(\delta\)，需要离线估计并对齐：

\[
s_{t+1}\approx f(s_t,u_{t-\delta})
\]

## 4. 相机标定与坐标统一

RGB-D 相机需要内参：

\[
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
\]

还需要相机坐标系到动捕世界坐标系或机器人基坐标系的外参：

\[
{}^WT_C=
\begin{bmatrix}
{}^WR_C&{}^Wt_C\\
0&1
\end{bmatrix}
\]

深度像素 \((u,v,d)\) 反投影为相机坐标：

\[
x_C=\frac{(u-c_x)d}{f_x},\quad
y_C=\frac{(v-c_y)d}{f_y},\quad
z_C=d
\]

再转换到世界坐标：

\[
\begin{bmatrix}
x_W\\1
\end{bmatrix}
=
{}^WT_C
\begin{bmatrix}
x_C\\1
\end{bmatrix}
\]

训练、验证和部署必须使用同一套坐标归一化参数。

## 5. RGB-D 预处理与点云构造

每帧先获得机器人区域 mask：

\[
M_t\in\{0,1\}^{H\times W}
\]

可用方法包括：

- 固定背景差分；
- 深度范围过滤；
- 颜色阈值；
- 轻量分割网络；
- 工作空间裁剪。

深度预处理包括：

- 无效深度过滤；
- 深度范围裁剪；
- 中值或双边滤波；
- 孤立点剔除；
- 统计离群点过滤；
- RGB 与深度对齐。

反投影得到彩色点云：

\[
P_t=
\{(x_i^W,y_i^W,z_i^W,r_i,g_i,b_i)\}_{i=1}^{N_t}
\]

再通过体素下采样、最远点采样或随机采样固定点数：

\[
P_t\in\mathbb R^{N\times 6}
\]

首版建议从 \(N=1024\) 或 \(2048\) 开始。

## 6. 模型结构

推荐首版结构：

```text
连续 K 帧 RGB-D
        ↓
机器人分割与深度预处理
        ↓
反投影彩色点云
        ↓
PointNet++ / Point Transformer 逐帧编码
        ↓
拼接历史控制
        ↓
GRU / Temporal Encoder
        ↓
12 维状态预测头
        ↓
可选视觉 latent 头
```

形式化表示：

\[
h_t^{geo}=E_{3D}(P_t)
\]

\[
h_t^{temp}=E_{temp}(h_{t-K+1:t}^{geo},u_{t-K+1:t-1})
\]

\[
\hat s_t=D_s(h_t^{temp})
\]

\[
z_t^V=P_z(h_t^{temp})
\]

多帧是必要的，因为单帧 RGB-D 可以估计几何位姿，但无法稳定判断速度、回弹趋势和迟滞分支。

## 7. 损失函数

状态监督：

\[
\mathcal L_{state}
=
\lambda_p\|\hat p_t-p_t\|^2
+\lambda_r\mathcal L_r
+\lambda_v\|\hat v_t-v_t\|^2
+\lambda_\omega\|\hat\omega_t-\omega_t\|^2
\]

姿态建议使用旋转测地距离，或在工程上对姿态维度做标准化后使用 Smooth L1。

可选几何重构约束：

\[
\mathcal L_{geo}=\mathcal L_{CD}
\]

其中 \(\mathcal L_{CD}\) 是预测点云与输入点云之间的 Chamfer Distance。该项用于避免几何编码器只学习状态回归捷径。

如果使用第一阶段已经训练好的状态空间 Koopman encoder 作为 teacher，可加入局部对齐：

\[
\mathcal L_{align}
=
\left\|
z_t^V-\operatorname{sg}(E_{koop}^{state}(s_t^{mocap}))
\right\|^2
\]

若希望视觉 latent 具有受控演化结构，可加入视觉状态恢复环节自己的 Koopman 约束：

\[
\mathcal L_{koop}^{visual}
=
\left\|
z_{t+1}^V-(A^V z_t^V+B^V u_t)
\right\|^2
\]

这里 \(z_t^V,A^V,B^V\) 只属于视觉状态恢复环节，不与 RLT 或视觉语言空间 Koopman 强行共享。

总损失可写为：

\[
\mathcal L
=
\lambda_s\mathcal L_{state}
+\lambda_g\mathcal L_{geo}
+\lambda_a\mathcal L_{align}
+\lambda_k\mathcal L_{koop}^{visual}
\]

首版可以先只使用 \(\mathcal L_{state}\)，再逐步加入其他项。

## 8. 训练阶段

推荐分阶段训练：

1. **几何预训练**：训练点云编码器和可选几何解码器，使其稳定表达软臂三维构型。
2. **多帧状态估计**：加入 GRU 和状态头，监督预测 12 维 TCP 状态。
3. **视觉 latent 对齐**：可选加入 \(L_{align}\)，让视觉 latent 接近状态空间 teacher latent。
4. **受控演化约束**：可选加入 \(L_{koop}^{visual}\)，提升闭环稳定性。
5. **闭环微调**：在视觉状态估计驱动下进行短时安全闭环测试，收集分布偏移数据再训练。

## 9. 部署流程

部署阶段不使用动捕：

```text
RGB-D 多帧 + 历史控制
        ↓
点云构造与时序编码
        ↓
视觉估计本体状态
        ↓
VLA 或状态技能策略
        ↓
delta TCP / 期望 TCP
        ↓
底层气压控制器
```

建议逐步比较：

- 动捕状态闭环；
- 动捕 + 视觉融合闭环；
- 纯视觉估计状态闭环。

## 10. 评估指标

状态估计指标：

- TCP 位置 RMSE，建议单位 mm；
- 姿态角误差，建议单位 degree；
- 线速度 RMSE；
- 角速度 RMSE；
- 视觉 latent 与 teacher latent 的 MSE；
- 1/3/5 步 latent rollout 误差。

闭环控制指标：

- pick 成功率；
- place 成功率；
- 最终目标误差；
- 达到阈值所需时间；
- 动作平滑度；
- 视觉闭环相对动捕闭环的性能差距。

## 11. 风险与处理

- 深度对软体材料失效：调整曝光、相机角度、表面纹理，必要时增加第二视角。
- TCP 遮挡：末端增加轻量视觉标志或局部点云分支。
- 速度标签噪声：使用滤波后的动捕速度，避免直接对噪声位姿差分。
- 时间同步偏差：记录硬件时间戳并离线估计延迟。
- 视觉闭环分布漂移：先做离线验证，再小范围安全闭环，最后逐步扩大任务分布。
