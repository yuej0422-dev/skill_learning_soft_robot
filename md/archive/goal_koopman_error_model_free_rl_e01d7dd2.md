# 基于目标预测与 Koopman 误差空间的 Model-Free RL 流程

## 1. 问题设定

考虑视觉语言条件下的机器人 manipulation 任务。

机械臂使用 **delta TCP** 作为控制输入：

\[
u_t=\Delta x_t^{\mathrm{TCP}}
\]

其中 \(u_t\) 表示当前控制周期内的末端位姿增量，而不是相邻两次动作之间的差分。

当：

\[
u_t=0
\]

低层控制器会维持当前 TCP 位姿。因此，当机械臂已经到达任务目标并且末端速度为零时：

\[
e_t=0,\qquad u_t=0
\]

可以构成一个自然的平衡状态。

---

## 2. 原始观测

每一步观测由视觉语言 embedding 与机器人本体状态组成。

视觉语言模型输出：

\[
h_t=F_{\mathrm{VLM}}(I_t,l)
\]

其中：

- \(I_t\)：当前 RGB 或 RGB-D 图像；
- \(l\)：任务语言指令；
- \(h_t\)：视觉语言 embedding。

机器人本体状态记为：

\[
p_t
\]

其中可以包括：

- TCP 位姿；
- TCP 线速度与角速度；
- 关节位置与关节速度；
- 夹爪状态；
- 力与力矩；
- 其他任务相关状态。

拼接后得到：

\[
x_t=[h_t,p_t]
\]

由于 \(p_t\) 中包含末端速度项，即使当前动作 \(u_t=0\)，系统仍可区分“静止在当前位置”和“仍带有运动趋势”的状态，从而使观测更接近 Markov 状态。

---

## 3. Koopman Encoder

将视觉语言 embedding 与机器人状态输入 Koopman encoder：

\[
\boxed{
z_t=\phi_\theta(h_t,p_t)
}
\]

其中：

\[
z_t\in\mathbb{R}^{d_z}
\]

为任务相关的 Koopman latent state。

该表示需要同时满足：

1. 在 delta TCP 动作作用下具有近似线性动力学；
2. 与目标 Koopman 向量之间的距离能够反映任务完成进度。

---

## 4. 专家轨迹与真实目标

假设拥有 \(N\) 条成功专家轨迹：

\[
\tau_i=
\left\{
(I_{i,t},p_{i,t},u_{i,t})
\right\}_{t=0}^{T_i}
\]

对于第 \(i\) 条轨迹，真实目标图像与目标状态取自成功终态：

\[
I_{g,i}=I_{i,T_i}
\]

\[
p_{g,i}=p_{i,T_i}
\]

目标 Koopman 向量定义为：

\[
\boxed{
z_{g,i}
=
\operatorname{sg}
\left[
\phi_\theta
\left(
F_{\mathrm{VLM}}(I_{g,i},l_i),
p_{g,i}
\right)
\right]
}
\]

其中 \(\operatorname{sg}\) 表示 stop-gradient。

也可以使用最后 \(K\) 个成功状态的平均值降低单帧噪声：

\[
z_{g,i}
=
\operatorname{sg}
\left[
\frac{1}{K}
\sum_{j=0}^{K-1}
z_{i,T_i-j}
\right]
\]

同一个 episode 内的所有 transition 均使用同一个真实目标 \(z_{g,i}\)。

---

## 5. Koopman 误差

对于第 \(i\) 条轨迹中的第 \(t\) 个状态，定义：

\[
\boxed{
e_{i,t}=z_{i,t}-z_{g,i}
}
\]

对应二次型距离：

\[
\boxed{
d_{i,t}
=
e_{i,t}^{\top}Qe_{i,t}
}
\]

其中：

\[
Q\succeq0
\]

由于终态自身被定义为目标：

\[
e_{i,T_i}
=
z_{i,T_i}-z_{g,i}
\approx0
\]

因此可以通过轨迹排序约束，使 \(d_{i,t}\) 与任务剩余进度建立相关性。

---

## 6. Koopman Encoder 训练

### 6.1 名义线性动力学

真实 Koopman latent dynamics 可以表示为：

\[
\boxed{
z_{t+1}=Az_t+Bu_t+c_t
}
\]

其中 \(c_t\) 是未建模残差，可能来自：

- Koopman 线性化误差；
- 接触模式切换；
- 摩擦与形变；
- 视觉表征误差；
- 控制延迟；
- 未观测状态；
- 外界扰动。

这里包含 \(c_t\) 时使用等号，因为 \(c_t\) 表示所有未被 \(Az_t+Bu_t\) 描述的剩余项。

训练 Koopman encoder 时不显式建模 \(c_t\)，只拟合名义线性部分：

\[
\boxed{
z_{t+1}\approx Az_t+Bu_t
}
\]

对应损失：

\[
\boxed{
\mathcal L_{\mathrm{dyn}}
=
\left\|
z_{t+1}-Az_t-Bu_t
\right\|_2^2
}
\]

因此，训练残差：

\[
z_{t+1}-Az_t-Bu_t
\]

隐式对应 \(c_t\)，但 \(c_t\) 不作为 Koopman 模型中的显式参数参与学习。

Koopman 模型的目标不是完全复现真实系统，而是提取主要的线性可控结构。

---

### 6.2 误差线性动力学

后续强化学习希望直接以 \(e_t\) 作为状态，因此进一步约束误差系统：

真实误差动力学写为：

\[
\boxed{
e_{t+1}=A_e e_t+B_eu_t+c_t^e
}
\]

其中 \(c_t^e\) 是误差空间中的未建模残差。

训练时同样不显式建模 \(c_t^e\)，只拟合：

\[
\boxed{
e_{t+1}\approx A_e e_t+B_eu_t
}
\]

对应损失：

\[
\boxed{
\mathcal L_{\mathrm{err}}
=
\left\|
e_{t+1}
-
A_e e_t
-
B_eu_t
\right\|_2^2
}
\]

这样可以直接学习：

\[
\text{当前任务误差}
+
\text{delta TCP 动作}
\longrightarrow
\text{下一时刻任务误差}
\]

相比只学习绝对 Koopman dynamics，该损失更直接服务于后续基于误差状态的强化学习。

---

### 6.3 任务进度排序损失

普通 VLM embedding 的数值距离不一定与任务完成程度正相关。

仅依靠 Koopman 线性预测损失，也不能自动保证：

\[
\|z_t-z_g\|
\]

能够表示任务进度。

因此使用同一条成功专家轨迹中的时间顺序构造排序监督。

定义：

\[
d_{i,t}
=
(z_{i,t}-z_{g,i})^\top
Q
(z_{i,t}-z_{g,i})
\]

对于：

\[
t<t+k
\]

希望后面的状态整体上更接近成功终态：

\[
d_{i,t}>d_{i,t+k}
\]

使用排序损失：

\[
\boxed{
\mathcal L_{\mathrm{rank}}
=
\max
\left(
0,
m+d_{i,t+k}-d_{i,t}
\right)
}
\]

其中 \(m>0\) 为 margin。

这里的 \(z_{i,t}\)、\(z_{i,t+k}\) 和 \(z_{g,i}\) 都来自真实观测经过 encoder 得到的 Koopman 向量，而不是通过 Koopman dynamics rollout 预测得到的向量。

因此，排序关系不会受到多步动力学预测误差的污染。

不建议要求每个相邻 transition 都严格满足距离递减，因为 manipulation 轨迹中可能存在：

- 局部微调；
- 暂时远离；
- 先抬高再下降；
- 接触后重新对准。

更适合在同一 episode 内采样间隔为 \(k\) 的状态对。

---

### 6.4 表征正则

为避免 Koopman encoder 发生表示坍塌，可以加入 latent variance 约束：

\[
\mathcal L_{\mathrm{var}}
=
\sum_j
\max
\left(
0,
\sigma_{\min}
-
\operatorname{Std}(z^{(j)})
\right)
\]

也可以加入状态重构损失：

\[
\mathcal L_{\mathrm{rec}}
=
\left\|
D_\omega(z_t)-x_t
\right\|_2^2
\]

其中：

\[
x_t=[h_t,p_t]
\]

重构项不是必要条件，但可以帮助 latent 保留视觉语义、本体状态和速度信息。

---

### 6.5 总损失

Koopman encoder 的完整训练目标可以写为：

\[
\boxed{
\mathcal L_{\mathrm{Koopman}}
=
\lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}
+
\lambda_{\mathrm{err}}\mathcal L_{\mathrm{err}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+
\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}
+
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
}
\]

最小实现可以只保留：

\[
\boxed{
\mathcal L_{\mathrm{Koopman}}
=
\lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}
+
\lambda_{\mathrm{err}}\mathcal L_{\mathrm{err}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
}
\]

三项分别用于：

1. 学习绝对 Koopman 状态的名义线性动力学；
2. 学习任务误差的名义线性动力学；
3. 让误差二次距离与任务完成度相关。

---

## 7. Goal Predictor

完成 Koopman encoder 训练后，将 encoder 冻结。

构造目标预测器：

\[
\boxed{
\hat z_{g,i,t}
=
G_\eta(z_{i,t},l_i)
}
\]

对于同一个 episode 中的每一个时刻 \(t\)，监督目标均为同一个真实目标 Koopman 向量：

\[
z_{g,i}
\]

goal predictor 只使用目标预测损失：

\[
\boxed{
\mathcal L_{\mathrm{goal}}
=
\left\|
\hat z_{g,i,t}
-
\operatorname{sg}(z_{g,i})
\right\|_2^2
}
\]

goal predictor 不再承担：

- Koopman 线性约束；
- 排序约束；
- 奖励预测；
- 成功分类。

这些性质已经由 Koopman encoder 的训练完成。

goal predictor 只负责：

> 根据当前任务状态和语言指令，预测该 episode 最终对应的目标 Koopman 向量。

在线执行时，可以在 episode 开始时预测一次：

\[
\hat z_g=G_\eta(z_0,l)
\]

并在整个 episode 中固定该目标，避免目标向量随当前观测不断漂移。

---

## 8. 在线误差构造

在线执行时：

\[
h_t=F_{\mathrm{VLM}}(I_t,l)
\]

\[
z_t=\phi_\theta(h_t,p_t)
\]

通过 goal predictor 得到固定目标：

\[
\hat z_g
\]

定义在线 Koopman 误差：

\[
\boxed{
e_t=z_t-\hat z_g
}
\]

该误差同时具有两层意义：

1. 在 delta TCP 动作作用下具有近似线性演化；
2. 其二次距离被训练为与任务完成进度相关。

---

## 9. 基于误差系统的 Model-Free RL

### 9.1 状态

强化学习状态定义为：

\[
\boxed{
s_t=e_t
}
\]

即：

\[
s_t=z_t-\hat z_g
\]

对于单一任务、相似目标分布和局部 manipulation，可以先采用该最小状态。

如果后续发现相同误差在不同目标下对应不同动力学，可以扩展为：

\[
s_t=[e_t,\hat z_g]
\]

---

### 9.2 动作

机械臂接口本身为 delta TCP，因此强化学习动作直接定义为：

\[
\boxed{
a_t=u_t=\Delta x_t^{\mathrm{TCP}}
}
\]

不需要再次定义：

\[
\Delta u_t=u_t-u_{t-1}
\]

否则相当于对 delta TCP 再积分一次。

---

### 9.3 奖励

定义二次型阶段代价：

\[
\boxed{
\ell_t
=
e_t^\top Qe_t
+
u_t^\top Ru_t
}
\]

其中：

- \(Q\succeq0\)：任务误差权重；
- \(R\succ0\)：delta TCP 动作幅值权重。

强化学习奖励为：

\[
\boxed{
r_t
=
-
\left(
e_t^\top Qe_t
+
u_t^\top Ru_t
\right)
}
\]

如果需要动作平滑，可以附加：

\[
(u_t-u_{t-1})^\top
R_s
(u_t-u_{t-1})
\]

此时 \(u_{t-1}\) 只用于计算平滑代价，而不是为了恢复绝对控制量。

---

### 9.4 Transition

最终 model-free RL 使用：

\[
\boxed{
(e_t,u_t,r_t,e_{t+1},d_t)
}
\]

其中：

- \(e_t\)：当前 Koopman 任务误差；
- \(u_t\)：delta TCP 动作；
- \(r_t\)：二次型奖励；
- \(e_{t+1}\)：真实下一观测编码得到的误差；
- \(d_t\)：终止标记。

RL 不使用：

\[
A_e e_t+B_eu_t
\]

生成虚拟下一状态，而是直接使用真实环境或真机产生的 \(e_{t+1}\)。

因此该方法属于：

\[
\boxed{
\text{Koopman-structured representation}
+
\text{model-free reinforcement learning}
}
\]

而不是使用 Koopman rollout 的 model-based RL。

---

## 10. Model-Free RL 对残差的补偿

真实误差动力学为：

\[
e_{t+1}
=
A_e e_t+B_eu_t+c_t^e
\]

Koopman encoder 训练时只拟合：

\[
e_{t+1}\approx A_e e_t+B_eu_t
\]

而 model-free RL 的 Bellman target 使用真实下一状态：

\[
\boxed{
y_t
=
r_t+\gamma V(e_{t+1})
}
\]

其中 \(e_{t+1}\) 已经包含 \(c_t^e\) 的真实影响。

因此，Critic 实际学习的是：

\[
Q^\pi(e_t,u_t)
=
\mathbb E
\left[
r_t+\gamma V^\pi(e_{t+1})
\mid e_t,u_t
\right]
\]

如果残差在训练数据和执行分布中具有稳定规律，价值函数和策略可以从真实 transition 中学习对残差影响的补偿。

Koopman 线性结构在这里主要作为归纳偏置，用于：

1. 构造低维可控状态；
2. 形成与任务进度相关的误差坐标；
3. 提供可直接构造二次型奖励的状态空间；
4. 降低后续 model-free RL 的学习难度。

它不需要成为完全精确的环境模型。

---

## 11. 完整训练流程

### 阶段一：采集专家数据

采集成功专家轨迹：

\[
\mathcal D_{\mathrm{demo}}
=
\{\tau_i\}_{i=1}^{N}
\]

每条轨迹包括：

- RGB 或 RGB-D 图像；
- 语言指令；
- 机器人本体状态；
- delta TCP 动作；
- 成功终态图像和终态状态。

### 阶段二：提取视觉语言 embedding

\[
h_{i,t}
=
F_{\mathrm{VLM}}(I_{i,t},l_i)
\]

VLM 可以保持冻结。

### 阶段三：训练 Koopman encoder

\[
z_{i,t}
=
\phi_\theta(h_{i,t},p_{i,t})
\]

利用真实终态构造：

\[
z_{g,i}
=
\operatorname{sg}(z_{i,T_i})
\]

计算：

\[
e_{i,t}=z_{i,t}-z_{g,i}
\]

联合优化：

\[
\mathcal L_{\mathrm{dyn}}
\]

\[
\mathcal L_{\mathrm{err}}
\]

\[
\mathcal L_{\mathrm{rank}}
\]

以及可选的：

\[
\mathcal L_{\mathrm{var}},
\qquad
\mathcal L_{\mathrm{rec}}
\]

### 阶段四：训练 goal predictor

冻结 Koopman encoder。

对同一个 episode 中的每个 \(z_{i,t}\)，预测相同的：

\[
z_{g,i}
\]

只优化：

\[
\mathcal L_{\mathrm{goal}}
=
\|
\hat z_{g,i,t}-z_{g,i}
\|^2
\]

### 阶段五：构造在线误差

episode 开始时：

\[
\hat z_g=G_\eta(z_0,l)
\]

每一步：

\[
z_t=\phi_\theta(h_t,p_t)
\]

\[
e_t=z_t-\hat z_g
\]

### 阶段六：Model-Free RL

状态：

\[
s_t=e_t
\]

动作：

\[
a_t=u_t=\Delta x_t^{\mathrm{TCP}}
\]

奖励：

\[
r_t=
-
\left(
e_t^\top Qe_t
+
u_t^\top Ru_t
\right)
\]

使用真实 transition：

\[
(e_t,u_t,r_t,e_{t+1},d_t)
\]

进行 model-free RL 更新。

---

## 12. 流程图

```text
RGB / RGB-D image I_t + language l
                    |
                    v
             Frozen VLM
                    |
                    v
       visual-language embedding h_t

h_t + robot state p_t
(TCP pose, TCP velocity, joints, gripper, force...)
                    |
                    v
            Koopman encoder
                    |
                    v
                   z_t

successful terminal image/state
                    |
                    v
                  z_g

             e_t = z_t - z_g
                    |
       +------------+-------------+
       |                          |
       v                          v
nominal dynamics loss       progress ranking loss
||z_{t+1}-Az_t-Bu_t||²      d_t > d_{t+k}
||e_{t+1}-A_e e_t-B_eu_t||²
       |                          |
       +------------+-------------+
                    |
                    v
      task-aligned Koopman error space

freeze Koopman encoder
                    |
                    v
train goal predictor:
every z_t in one episode -> same z_g
                    |
                    v
          predicted goal z_hat_g
                    |
                    v
             e_t = z_t-z_hat_g
                    |
                    v
              model-free RL

state:      s_t = e_t
action:     a_t = delta TCP
reward:     -(e_t^T Q e_t + u_t^T R u_t)
transition: (e_t,u_t,r_t,e_{t+1},done)
```

---

## 13. 核心 Insight

未来图像或未来 latent 预测的目的，本质上是为动作提供一个可优化的目标差异。

该方法不直接在高维图像空间中预测并比较未来结果，而是学习一个同时具有：

- 名义线性动力学；
- 任务进度排序；
- 目标条件结构；

的 Koopman 误差空间。

最终将视觉语言任务转化为：

\[
\boxed{
\text{寻找能够直接缩小 Koopman 任务误差的 delta TCP 动作}
}
\]

整体结构为：

\[
\boxed{
\text{Visual-language observation}
\rightarrow
\text{task-aligned Koopman error}
\rightarrow
\text{quadratic dense reward}
\rightarrow
\text{model-free RL}
}
\]

Koopman 模型只提取主要线性结构，不显式建模残差 \(c_t\)；model-free RL 通过真实 transition 学习包含残差影响后的价值函数与策略。
