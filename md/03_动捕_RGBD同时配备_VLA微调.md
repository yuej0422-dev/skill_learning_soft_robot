# 动捕与RGB同时配备下的 VLA 微调

## 1. 阶段定位

该阶段按照刚体机器人 VLA 的部署流程来做。刚体机器人通常可以直接读取末端状态或关节状态；软体机械臂自身难以稳定获得完整本体状态，因此在该阶段由动捕系统提供等价的末端状态信息。

因此，该阶段不需要改变成熟 VLA 的基本范式：

```text
RGB-D / RGB 图像 + 语言指令 + 末端状态
        ↓
VLA
        ↓
delta TCP
        ↓
动作适配层
        ↓
底层气压控制器
        ↓
气压
```

动捕在这里的作用是让软体机械臂具备类似刚体机器人 proprioception 的输入条件，而不是让 VLA 学习气压控制。

## 2. 输入输出接口

VLA 输入：

\[
o_t=
\{I_t,D_t,l_t,s_t^{mocap}\}
\]

其中：

- \(I_t,D_t\)：RGB-D 观测，若 VLA 流程只使用 RGB，则深度可仅用于记录或辅助标定；
- \(l_t\)：语言指令；
- \(s_t^{mocap}\)：动捕提供的末端状态，等价于刚体 VLA 中的末端 proprioception。

VLA 输出：

\[
a_t=\Delta x_t^{TCP}
\]

即当前控制周期内的 delta TCP 动作。它可以包含位置增量和姿态增量：

\[
\Delta x_t^{TCP}=[\Delta p_t,\Delta r_t]
\]

不要求 VLA 输出期望气压，也不要求 VLA 直接输出底层控制器内部变量。

## 3. 动作适配层

VLA 输出 delta TCP 后，经过很短的动作适配层转换为底层控制器可执行的期望 TCP 位姿与速度。

位置部分：

\[
x_{tcp,t}^{des}=x_{tcp,t}^{mocap}+\Delta x_t^{TCP}
\]

速度部分可由动作周期近似得到：

\[
\dot{x}_{tcp,t}^{des}=\frac{\Delta x_t^{TCP}}{\Delta t}
\]

工程实现时需要加入：

- delta TCP 幅值裁剪；
- 姿态增量合法化；
- 速度上限；
- 动作低通或 rate limit；
- 工作空间边界检查。

随后调用已有底层控制器：

\[
(x_{tcp,t}^{des},\dot{x}_{tcp,t}^{des})
\rightarrow pressure_t
\]

## 4. 数据记录格式

该阶段按刚体 VLA 数据格式记录，不必每个时间步额外保存底层控制器期望输入作为核心监督标签。

推荐记录：

\[
\mathcal D_t=
\{
I_t,
D_t,
l_t,
s_t^{mocap},
a_t^{demo},
pressure_t,
\tau_t
\}
\]

其中：

- \(s_t^{mocap}\)：动捕提供的 TCP 位姿和速度；
- \(a_t^{demo}\)：示教或专家策略对应的 delta TCP；
- \(pressure_t\)：实际发送气压，可作为系统日志和安全分析，不作为 VLA 首要输出标签；
- \(\tau_t\)：时间戳。

如果任务涉及物体，额外记录：

\[
o_t^{mocap},g_t,phase_t,success
\]

其中 \(o_t^{mocap}\) 是物体位姿，\(g_t\) 是目标区域或目标 pose，\(phase_t\) 是可选阶段标签。

## 5. 训练目标

基础 VLA 监督微调直接学习 delta TCP：

\[
\mathcal L_{SFT}
=
\left\|
\hat{\Delta x}_{t:t+H}^{TCP}
-
\Delta x_{t:t+H}^{TCP,*}
\right\|^2
\]

如果使用 action chunk，则 VLA 输出一段未来 delta TCP：

\[
\hat a_{t:t+H}
=
\{\Delta x_{t}^{TCP},\ldots,\Delta x_{t+H-1}^{TCP}\}
\]

执行时可以采用 receding horizon，只执行第一个动作或前几个动作。

## 6. 状态空间 Koopman 的使用边界

该阶段可以选择把提前训练好的状态空间 Koopman latent 作为额外 proprioception：

\[
z_t^S=E_{koop}^{state}(s_t^{mocap})
\]

则 VLA 输入可以扩展为：

\[
\{I_t,D_t,l_t,s_t^{mocap},z_t^S\}
\]

但这不是刚体 VLA 部署流程的必要条件。若加入，也仅表示动捕状态空间环节的辅助状态，不与后续 RLT 或视觉语言空间 Koopman 共享。

## 7. 部署流程

在线执行：

```text
读取 RGB-D / RGB
读取语言指令
读取动捕 TCP 状态
        ↓
VLA 输出 delta TCP
        ↓
clip / filter / workspace check
        ↓
delta TCP 转换为期望 TCP 位姿和速度
        ↓
底层控制器输出气压
        ↓
软体机械臂执行
```

该流程与刚体 VLA 的主要差别只有一点：刚体机器人可直接获得末端状态，软体机械臂在此阶段由动捕提供该状态。

## 8. 验证指标

以 pick and place 为例，评估：

- pick 成功率；
- place 成功率；
- 目标位置误差；
- delta TCP 动作平滑度；
- 气压变化平滑度；
- 不同初始物体位置的泛化；
- 语言指令变化下的执行稳定性。
