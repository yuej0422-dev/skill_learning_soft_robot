# 基于 D4RL antmaze-medium-diverse-v2 的线性二次型 Koopman RL 数据构造指南

本文说明如何从 D4RL 离线数据集 `antmaze-medium-diverse-v2` 构造适配线性二次型 Koopman-based RL 的训练数据。核心思想是：

1. 从 D4RL 原始 transition 中抽取 Ant 的二维位置、机器人本体状态和绝对动作。
2. 将绝对动作 \(u_t\) 转换为控制增量 \(\Delta u_t=u_t-u_{t-1}\)。
3. 将状态增广为 \([e_t,o_t,u_{t-1}]\)，保证增量控制下的 Markov 性。
4. 用 lifting 函数 \(\phi(\cdot)\) 得到 Koopman latent state。
5. 在线性系统
   \[
   \xi_{t+1}=A_{\rm aug}\xi_t+B_{\rm aug}\Delta u_t+c_{\rm aug}
   \]
   上构造二次型代价或奖励。

---

## 1. 环境与数据假设

### 1.1 D4RL 环境

目标环境：

```python
env_name = "antmaze-medium-diverse-v2"
```

典型 D4RL 用法：

```python
import gym
import d4rl

env = gym.make("antmaze-medium-diverse-v2")
dataset = env.get_dataset()
```

常见字段包括：

```text
observations        # shape: [N, obs_dim]
actions             # shape: [N, action_dim]
rewards             # shape: [N]
terminals           # shape: [N]
timeouts            # shape: [N], 某些版本可能存在
next_observations   # 某些封装或 qlearning_dataset 中可能存在
```

也可以使用：

```python
dataset = d4rl.qlearning_dataset(env)
```

但如果你需要严格恢复 episode 边界，优先检查 `env.get_dataset()` 是否包含 `timeouts`。AntMaze 数据中 `terminals` 不一定足以表达所有轨迹切分点。

### 1.2 AntMaze observation 拆分

在 D4RL AntMaze 常见状态表示中：

```text
obs_t[0:2]  = p_xy,t        # Ant torso 的全局 x, y 位置
obs_t[2:]   = o_t           # Ant 本体观测，例如姿态、关节、速度等
```

因此：

\[
p_{xy,t} = \mathrm{obs}_t[0:2]
\]

\[
o_t = \mathrm{obs}_t[2:]
\]

AntMaze 的 action 通常是 8 维关节力矩：

\[
u_t \in [-1,1]^8
\]

即：

```python
u_t = action_t
```

---

## 2. 目标误差与状态定义

### 2.1 目标位置

设当前任务目标为：

\[
g_{xy}\in\mathbb{R}^2
\]

对 `antmaze-medium-diverse-v2`，需要先明确你要构造的数据服务于哪种目标设置：

| 设置 | 说明 | 推荐用途 |
|---|---|---|
| 固定评估目标 | 使用环境默认 evaluation goal | 复现标准 AntMaze 任务 |
| 数据内重标注目标 | 从未来状态或成功片段采样目标 | goal-conditioned / hindsight 数据增强 |
| 多目标训练 | 为每条 transition 附加采样目标 \(g_{xy}\) | 更适合 diverse 数据 |

如果先做最小可运行版本，建议使用固定目标：

```python
goal_xy = env.target_goal.copy()
```

如果环境对象没有暴露 `target_goal`，可以通过 reset 后的 env 属性或 wrapper 检查：

```python
print(env.unwrapped.__dict__.keys())
```

### 2.2 位置误差

定义目标误差：

\[
e_t = g_{xy} - p_{xy,t}
\]

其中：

\[
p_{xy,t}=\mathrm{obs}_t[0:2]
\]

### 2.3 原始物理状态

定义：

\[
x_t = [e_t,o_t]
\]

其中：

```python
p_xy = obs[:2]
proprio = obs[2:]
e = goal_xy - p_xy
x = concat(e, proprio)
```

---

## 3. 为什么要使用控制增量

原始 D4RL action 是实际发送给 MuJoCo Ant 的绝对控制量：

\[
u_t
\]

如果希望策略输出控制增量：

\[
a_t=\Delta u_t
\]

则实际执行动作是：

\[
u_t=\operatorname{clip}(u_{t-1}+\Delta u_t,u_{\min},u_{\max})
\]

这样做的好处是：在稳定运动或接近目标时，策略可以学习

\[
\Delta u_t\rightarrow 0
\]

但允许维持一个非零绝对力矩：

\[
u_t\rightarrow u^\ast\neq 0
\]

这比强行让策略直接输出趋近 0 的绝对动作更适合持续支撑、站立和平衡类机器人控制。

---

## 4. 增广 MDP 构造

由于动作由增量积分得到：

\[
u_t=u_{t-1}+\Delta u_t
\]

下一个物理状态不仅依赖 \([e_t,o_t]\) 和 \(\Delta u_t\)，也依赖上一时刻的绝对动作 \(u_{t-1}\)。因此需要增广状态：

\[
\boxed{
s_t=[e_t,o_t,u_{t-1}]
}
\]

动作定义为：

\[
\boxed{
a_t=\Delta u_t
}
\]

下一状态为：

\[
\boxed{
s_{t+1}=[e_{t+1},o_{t+1},u_t]
}
\]

一条离线 transition 变为：

\[
\boxed{
(s_t,\Delta u_t,r_t,s_{t+1},d_t)
}
\]

也就是：

\[
\boxed{
([e_t,o_t,u_{t-1}],u_t-u_{t-1},r_t,[e_{t+1},o_{t+1},u_t],d_t)
}
\]

---

## 5. 从 D4RL 数据转为增量控制数据

### 5.1 Episode 边界

动作差分不能跨 episode 计算。对每个 transition，需要判断它是否是一条轨迹的起点。

推荐边界来源优先级：

1. `timeouts[t-1] == True`
2. `terminals[t-1] == True`
3. `next_observations[t-1]` 与 `observations[t]` 不连续
4. Ant 位置跳变过大，例如：
   \[
   \|p_{xy,t}-p_{xy,t-1}\|_2 > \tau_{\rm jump}
   \]

其中 \(\tau_{\rm jump}\) 可以先取 `1.0` 到 `2.0`，再结合数据统计调整。

### 5.2 初始上一动作

每个 episode 起点没有真实的 \(u_{t-1}\)。常见处理：

```python
prev_u = zeros(action_dim)
```

即：

\[
u_{-1}=0
\]

如果你的仿真 reset 后存在默认控制量，则应使用真实 reset 控制量。

### 5.3 动作差分

对非 episode 起点：

\[
\Delta u_t=u_t-u_{t-1}
\]

对 episode 起点：

\[
\Delta u_0=u_0-u_{-1}
\]

若 \(u_{-1}=0\)，则：

\[
\Delta u_0=u_0
\]

注意：如果你在后续执行中会进行 action clipping，则构造数据时的 \(u_t\) 应理解为已经实际发送给环境的 clipped action。D4RL 的 `actions` 通常已经是实际动作。

---

## 6. Koopman lifting 与线性动力学

### 6.1 Lifting 输入

先对物理状态做 lifting：

\[
z_t=\phi(x_t)=\phi(e_t,o_t)
\]

其中：

\[
z_t\in\mathbb{R}^{d_z}
\]

然后构造增广 Koopman 状态：

\[
\boxed{
\xi_t=
\begin{bmatrix}
z_t\\
u_{t-1}
\end{bmatrix}
}
\]

下一状态：

\[
\boxed{
\xi_{t+1}=
\begin{bmatrix}
z_{t+1}\\
u_t
\end{bmatrix}
}
\]

### 6.2 绝对动作形式

假设 Koopman latent dynamics 近似满足：

\[
z_{t+1}=Az_t+Bu_t+c
\]

代入：

\[
u_t=u_{t-1}+\Delta u_t
\]

得到：

\[
z_{t+1}=Az_t+Bu_{t-1}+B\Delta u_t+c
\]

### 6.3 增广线性系统

定义：

\[
\xi_t=
\begin{bmatrix}
z_t\\
u_{t-1}
\end{bmatrix}
\]

则：

\[
\boxed{
\xi_{t+1}
=
A_{\rm aug}\xi_t
+
B_{\rm aug}\Delta u_t
+
c_{\rm aug}
}
\]

其中：

\[
A_{\rm aug}
=
\begin{bmatrix}
A&B\\
0&I
\end{bmatrix}
\]

\[
B_{\rm aug}
=
\begin{bmatrix}
B\\
I
\end{bmatrix}
\]

\[
c_{\rm aug}
=
\begin{bmatrix}
c\\
0
\end{bmatrix}
\]

这个结构显式保留了动作积分关系，不需要神经网络重新学习

\[
u_t=u_{t-1}+\Delta u_t
\]

---

## 7. Koopman 模型训练目标

这里直接采用绝对动作形式训练 Koopman 模型：

\[
\mathcal{L}_{\rm dyn}
=
\left\|
z_{t+1}-(Az_t+Bu_t+c)
\right\|_2^2
\]

具体训练细节参考原训练代码 `Learning_Koopman_with_Reg_HPN.py`，这里不展开。采用绝对动作 \(u_t\) 训练更方便，同时由

\[
u_t=u_{t-1}+\Delta u_t
\]

诱导出的增广形式也符合近似线性演化：

\[
\xi_{t+1}
=
A_{\rm aug}\xi_t+B_{\rm aug}\Delta u_t+c_{\rm aug}
\]

---

## 8. 基于 Koopman Encoder 重构增广 MDP

训练好 Koopman encoder 后，不再把物理状态 \([e_t,o_t]\) 直接作为 RL 状态，而是先编码为：

\[
z_t=\phi(e_t,o_t)
\]

然后将第 4 节中的增广 MDP 重构到 Koopman latent 空间：

\[
\boxed{
s_t=
\xi_t=
\begin{bmatrix}
\phi(e_t,o_t)\\
u_{t-1}
\end{bmatrix}
}
\]

\[
\boxed{
a_t=\Delta u_t=u_t-u_{t-1}
}
\]

\[
\boxed{
s_{t+1}
=
\xi_{t+1}
=
\begin{bmatrix}
\phi(e_{t+1},o_{t+1})\\
u_t
\end{bmatrix}
}
\]

因此，D4RL 中每条离线 transition 会被转成：

\[
\boxed{
(
\xi_t,\Delta u_t,r_t,\xi_{t+1},d_t
)
}
\]

即：

\[
\boxed{
\left(
\begin{bmatrix}
\phi(e_t,o_t)\\
u_{t-1}
\end{bmatrix},
\Delta u_t,
r_t,
\begin{bmatrix}
\phi(e_{t+1},o_{t+1})\\
u_t
\end{bmatrix}
\right)
}
\]

这样构造出的所有离线 transition 都隐含了 Koopman latent linear 约束：

\[
\xi_{t+1}
=
A_{\rm aug}\xi_t+B_{\rm aug}\Delta u_t+c_{\rm aug}
\]

后续 offline RL 可以直接在 \(\xi_t\) 上训练，actor 输出控制增量 \(\Delta u_t\)。
参考文章中的Q函数参数化设置，可直接获取actor的解析解

---

## 9. 线性二次型代价设计

在 Koopman latent 增广 MDP 上，阶段代价设计为：

\[
\boxed{
\ell_t
=
\xi_t^\top Q_\xi \xi_t
+
\Delta u_t^\top R_\Delta \Delta u_t
}
\]

奖励为：

\[
\boxed{
r_t=-\ell_t
}
\]

其中：

1. \(Q_\xi\succeq0\)：latent state 及上一时刻绝对动作的权重。
2. \(R_\Delta\succ0\)：控制增量平滑项。
3. \(\Delta u_t\)：策略真正输出的 action。

保留物理可解释性，让 Koopman decoder 为 \( [x_t;\phi(x_t)] \)，这样前n维度就是原状态

对 AntMaze，初始版本建议优先约束：

1. 位置误差 \(e_t\)。
2. 躯干速度、角速度等稳定性相关变量。
3. 控制增量 \(\Delta u_t\)。

可选加入很弱的绝对动作正则(\(\lambda_u\)可以先为0)：

\[
\ell_t
=
\xi_t^\top Q_\xi \xi_t
+
\Delta u_t^\top R_\Delta \Delta u_t
+
\lambda_u u_t^\top R_u u_t
\]

其中：

\[
0<\lambda_u\ll1
\]

该项只用于抑制动作漂移，不应主导优化，否则可能让策略错误地追求：

\[
u_t\rightarrow0
\]

---

## 10. 数据转换输出格式

建议将转换后的数据保存为 `.npz`：

```text
koopman_antmaze_medium_diverse_v2.npz
```

字段建议：

```text
x                 # [M, x_dim], 原始物理状态 [e, proprio]
next_x            # [M, x_dim]
z                 # [M, z_dim], phi(e, proprio)
next_z            # [M, z_dim]
u_prev            # [M, action_dim]
u                 # [M, action_dim]
delta_u           # [M, action_dim]
xi                # [M, z_dim + action_dim], [z, u_prev]
next_xi           # [M, z_dim + action_dim], [next_z, u]
reward_d4rl       # [M], 原始 D4RL sparse reward
reward_lqr        # [M], 自定义二次型 reward
done              # [M]
episode_start     # [M]
goal_xy           # [M, 2] 或 [2]
```

---

## 11. 推荐整体流程

### Step 1: 读取 D4RL 数据

```python
env = gym.make("antmaze-medium-diverse-v2")
dataset = env.get_dataset()
```

检查字段：

```python
print(dataset.keys())
print(dataset["observations"].shape)
print(dataset["actions"].shape)
```

### Step 2: 确定目标 \(g_{xy}\)

最小版本：

```python
goal_xy = env.target_goal.copy()
```

如果做 hindsight relabeling：

```python
goal_xy = observations[future_index, :2]
```

### Step 3: 拆分 observation

\[
obs_t\rightarrow(p_{xy,t},o_t)
\]

\[
e_t=g_{xy}-p_{xy,t}
\]

\[
x_t=[e_t,o_t]
\]

### Step 4: 用绝对动作训练 Koopman encoder

用 D4RL 中的绝对动作 \(u_t\) 训练：

\[
z_t=\phi(x_t)
\]

\[
z_{t+1}\approx Az_t+Bu_t+c
\]

训练代码参考 `Learning_Koopman_with_Reg_HPN.py`。

### Step 5: 用训练好的 encoder 编码离线数据

\[
z_t=\phi(e_t,o_t)
\]

\[
z_{t+1}=\phi(e_{t+1},o_{t+1})
\]

### Step 6: 按 episode 构造动作增量

\[
\Delta u_t=u_t-u_{t-1}
\]

禁止跨 episode 做差分。

### Step 7: 构造 Koopman latent 增广状态

\[
\xi_t=[z_t,u_{t-1}]
\]

\[
\xi_{t+1}=[z_{t+1},u_t]
\]

此时 transition 具有近似线性约束：

\[
\xi_{t+1} = A_{\rm aug}\xi_t+B_{\rm aug}\Delta u_t+c_{\rm aug}
\]

### Step 8: 构造二次型 reward

\[
r_t^{\rm lqr}
=
-
(
\xi_t^\top Q_\xi\xi_t
+
\Delta u_t^\top R_\Delta\Delta u_t
)
\]

也可以先用物理空间的 \(e_t\) 版本做调试：

\[
r_t^{\rm lqr}
=
-
(
e_t^\top Q_ee_t
+
\Delta u_t^\top R_\Delta\Delta u_t
)
\]

### Step 9: 用转换后的 latent 数据训练 RL

参考文章内容针对(\( \xi_t, \Delta u_t, r_t, \xi_{t+1} \))做离线RL，Critic做参数化处理为h向量

策略输入：

\[
\xi_t
\]

策略输出：

\[
\Delta u_t
\]

执行时恢复绝对动作：

\[
u_t=\operatorname{clip}(u_{t-1}+\Delta u_t,-1,1)
\]

---

<!-- ## 12. 实验检查清单

转换后建议检查：

1. `delta_u` 不应在 episode 边界出现异常大跳变。
2. `u` 与 D4RL 原始 `actions` 基本一致。
3. `next_xi[..., -action_dim:]` 应等于当前 `u`。
4. `xi[..., -action_dim:]` 应等于上一时刻 `u_prev`。
5. `z` 与 `next_z` 应由同一个训练好的 Koopman encoder 生成。
6. `reward_lqr` 的量级不要过大，必要时标准化 \(e_t\)、proprioception、latent state 和 action。
7. 如果策略执行时频繁触碰 action 边界，适当增大 \(R_\Delta\) 或加入弱 \(u_t\) 正则。

---

## 13. 注意事项

1. `antmaze-medium-diverse-v2` 是稀疏奖励、长时程、可拼接性很强的数据集，原始 D4RL reward 不一定适合直接训练线性二次型控制器。
2. diverse 数据包含随机起点和随机目标生成逻辑，若只使用固定 \(g_{xy}\)，部分 transition 可能与目标方向不完全一致。
3. 如果做 goal relabeling，需要把 \(g_{xy}\) 作为数据字段保存，并在训练时输入 encoder 或 cost function。
4. 不建议直接对全部 Koopman latent 维度设置单位 \(Q\)，应通过 \(C_yz_t\) 选出有物理意义的变量。
5. 增量动作会改变 action distribution，训练 offline RL 时应注意 \(\Delta u_t\) 的分布范围和行为策略支持集。
6. 如果使用 CQL、IQL、TD3+BC 等离线算法，需要确认 actor 输出的是 \(\Delta u_t\)，而不是原始 \(u_t\)。
7. 评估时必须维护 `prev_u`，每次环境 reset 后将 `prev_u` 重置为零或真实 reset 控制量。

---

## 14. 最小端到端结构

最终你希望得到下面的数据与模型关系：

```text
D4RL obs_t, action_t, reward_t, next_obs_t
        |
        v
split obs:
    p_xy,t = obs_t[:2]
    o_t    = obs_t[2:]
        |
        v
goal error:
    e_t = goal_xy - p_xy,t
    x_t = [e_t, o_t]
        |
        v
Koopman training:
    train phi with absolute action u_t
    z_{t+1} ~= A z_t + B u_t + c
        |
        v
Koopman encoding:
    z_t = phi(x_t)
    z_{t+1} = phi(x_{t+1})
        |
        v
delta action:
    delta_u_t = u_t - u_{t-1}
        |
        v
latent augmented transition:
    xi_t = [z_t, u_{t-1}]
    xi_{t+1} = [z_{t+1}, u_t]
        |
        v
linear dynamics:
    xi_{t+1} = A_aug xi_t + B_aug delta_u_t + c_aug
        |
        v
quadratic reward:
    r_t = -(xi_t^T Q_xi xi_t + delta_u_t^T R_delta delta_u_t)
```

这就是从 D4RL AntMaze 离线数据到线性二次型 Koopman-based RL 数据的基本转换路径。

---

## 参考资料

1. D4RL GitHub: https://github.com/Farama-Foundation/D4RL
2. D4RL Tasks Wiki, AntMaze: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
3. Gymnasium-Robotics AntMaze documentation: https://robotics.farama.org/envs/maze/ant_maze/ -->
