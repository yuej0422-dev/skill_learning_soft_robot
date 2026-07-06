# 轻量软体 VLA：基于 Goal / Koopman Error 的路线

## 1. 阶段定位

该文档收束 `goal_koopman_error_model_free_rl_e01d7dd2.md` 的核心思想，用于构建面向软体机械臂操作任务的轻量 VLA。

目标不是训练一个完全通用的大型 VLA，而是在 pick and place 等一类软体操作任务上，通过任务相关 latent、目标预测、误差空间和偏差控制获得更好的泛化性与数据效率。

这里的 Koopman encoder 作用于视觉语言或任务表征空间，不等同于状态空间 Koopman encoder。

## 2. 问题设定

VLA 或轻量策略最终输出 delta TCP：

\[
u_t=\Delta x_t^{TCP}
\]

底层气压控制器负责将当前 TCP 状态和 delta TCP 转换为期望 TCP 位姿/速度，再输出气压。

当：

\[
u_t=0
\]

底层控制器维持当前 TCP 位姿。因此在任务目标处，如果任务误差为零且速度为零，可以形成自然平衡状态：

\[
e_t=0,\qquad u_t=0
\]

这使得基于误差收敛的控制和 RL 奖励设计比较自然。

## 3. 原始观测与视觉语言表征

输入包括图像、语言和可选本体状态：

\[
o_t=(I_t,l_t,p_t)
\]

其中：

- \(I_t\)：RGB 或 RGB-D 图像；
- \(l_t\)：语言指令；
- \(p_t\)：可选状态，例如动捕 TCP 状态或视觉估计状态。

VLM/VLA 输出视觉语言 hidden states 或 embedding：

\[
H_t^{VLM/VLA}
\]

视觉语言空间 Koopman encoder 或任务 encoder 将其压缩为任务 latent：

\[
z_t=E_{koop}^{VL}(H_t^{VLM/VLA},p_t)
\]

若不使用显式 Koopman，也可以记为普通任务 encoder：

\[
z_t=E_{task}(H_t^{VLM/VLA},p_t)
\]

后续所有目标预测和误差构造都在该任务 latent 空间中进行。

## 4. 专家轨迹与目标 latent

假设有成功专家轨迹：

\[
\tau_i=
\{(I_{i,t},l_i,p_{i,t},u_{i,t})\}_{t=0}^{T_i}
\]

对每条成功轨迹，用终态构造目标 latent：

\[
z_{g,i}
=
\operatorname{sg}(z_{i,T_i})
\]

为了降低单帧噪声，也可使用最后 \(K\) 个成功状态平均：

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

同一条 episode 内所有 transition 使用同一个真实目标 latent \(z_{g,i}\)。

## 5. Koopman Error

定义任务误差：

\[
e_{i,t}=z_{i,t}-z_{g,i}
\]

二次距离：

\[
d_{i,t}=e_{i,t}^{\top}Qe_{i,t}
\]

希望该距离不仅是数值距离，还能与任务剩余进度相关。因此需要结合动力学约束和进度排序约束训练 latent。

## 6. 视觉语言空间 Koopman 训练

如果使用视觉语言空间 Koopman encoder，目标不是完全复现真实系统，而是提取对控制有用的低维动态结构。

名义 latent dynamics：

\[
z_{t+1}\approx Az_t+Bu_t
\]

损失：

\[
\mathcal L_{dyn}
=
\left\|
z_{t+1}-Az_t-Bu_t
\right\|^2
\]

误差空间 dynamics：

\[
e_{t+1}\approx A_e e_t+B_eu_t
\]

损失：

\[
\mathcal L_{err}
=
\left\|
e_{t+1}-A_e e_t-B_eu_t
\right\|^2
\]

这里的 \(A,B,A_e,B_e\) 属于视觉语言/任务 latent 空间，不与状态空间 Koopman 共享。

## 7. 任务进度排序损失

普通 VLM embedding 的距离不一定表示任务完成进度。因此对同一条成功轨迹采样 \(t<t+k\)，希望后面的状态更接近成功终态：

\[
d_{i,t}>d_{i,t+k}
\]

使用 margin ranking loss：

\[
\mathcal L_{rank}
=
\max(0,m+d_{i,t+k}-d_{i,t})
\]

不要求每个相邻 transition 都严格距离递减，因为实际操作中可能存在局部调整、绕行、接触后重新对准等阶段。

## 8. 表征正则

为避免 latent 坍塌，可加入 variance 正则：

\[
\mathcal L_{var}
=
\sum_j
\max(0,\sigma_{min}-Std(z^{(j)}))
\]

也可以加入重构损失，让 latent 保留视觉语言和状态信息：

\[
\mathcal L_{rec}
=
\left\|
D(z_t)-[H_t^{VLM/VLA},p_t]
\right\|^2
\]

最小实现可以先使用：

\[
\mathcal L_{Koopman}
=
\lambda_{dyn}\mathcal L_{dyn}
+\lambda_{err}\mathcal L_{err}
+\lambda_{rank}\mathcal L_{rank}
\]

再根据训练稳定性加入 \(\mathcal L_{var}\) 和 \(\mathcal L_{rec}\)。

## 9. Goal Predictor

训练完任务 latent 或 Koopman encoder 后，冻结该 encoder，训练 goal predictor：

\[
\hat z_g=G_\eta(z_t,l_t)
\]

监督目标为同一 episode 的真实终态 latent：

\[
\mathcal L_{goal}
=
\left\|
\hat z_g-\operatorname{sg}(z_g)
\right\|^2
\]

goal predictor 不再承担 Koopman 线性约束或排序约束，只负责根据当前任务状态和语言指令预测该 episode 应到达的目标 latent。

在线执行时建议在 episode 或任务阶段开始时预测一次：

\[
\hat z_g=G_\eta(z_0,l)
\]

并在短时间窗口内固定，避免目标 latent 随观测漂移。

## 10. 在线误差构造

在线执行每一步：

\[
z_t=E_{koop}^{VL}(H_t^{VLM/VLA},p_t)
\]

\[
e_t=z_t-\hat z_g
\]

轻量策略、critic 或 residual actor 使用：

\[
s_t^{RL}=e_t
\]

如果相同误差在不同目标下对应不同动作，可扩展为：

\[
s_t^{RL}=[e_t,\hat z_g]
\]

也可以与 RLT latent 或视觉估计状态拼接。

## 11. 偏差控制

推荐不从零生成完整动作，而是在参考动作或参考目标附近做修正。

动作偏差：

\[
u_t=u_t^{ref}+\Delta u_t
\]

目标偏差：

\[
g_t=g_t^{ref}+\Delta g_t
\]

其中：

- \(u_t^{ref}\)：由 SFT 后 VLA、状态技能策略或规则策略给出；
- \(\Delta u_t\)：由轻量 residual policy 根据 \(e_t\) 输出；
- 最终 \(u_t\) 仍是 delta TCP。

这样可以保留 VLA 的视觉语言先验和示教行为，同时让在线 RL 只学习局部修正。

## 12. Model-Free RL 接口

RL 状态：

\[
s_t^{RL}=e_t
\]

或：

\[
s_t^{RL}=[e_t,\hat z_g,z_t^{RLT}]
\]

动作：

\[
a_t=\Delta u_t
\]

最终执行：

\[
u_t=u_t^{ref}+a_t
\]

奖励可写为：

\[
r_t
=
-
\left(
e_t^\top Qe_t
+a_t^\top Ra_t
\right)
\]

如果需要动作平滑，可加入：

\[
(u_t-u_{t-1})^\top R_s(u_t-u_{t-1})
\]

transition 使用真实环境或真机下一状态：

\[
(e_t,a_t,r_t,e_{t+1},done)
\]

RL 不使用 Koopman rollout 生成虚拟下一状态，因此该方法是：

```text
Koopman-structured representation
        +
model-free residual RL
```

而不是 model-based RL。

## 13. 与 RLT 的结合

RLT 从 VLA hidden states 中提取紧凑状态：

\[
z_t^{RLT}=E_{RLT}(H_t^{VLM/VLA})
\]

视觉语言空间 Koopman latent 可作为 RLT 训练过程中的动态结构约束或额外 RL state：

\[
x_t^{RL}=[z_t^{RLT},e_t]
\]

或：

\[
x_t^{RL}=[z_t^{RLT},z_t,\hat z_g,e_t]
\]

具体选择应从简单开始，避免把 RLT、Koopman、goal predictor 和 RL 一次性全部耦合。

## 14. 完整流程

```text
RGB-D / RGB + language
        ↓
VLM / VLA hidden states
        ↓
视觉语言空间 Koopman encoder 或 task encoder
        ↓
任务 latent z_t
        ↓
goal predictor 得到 z_g
        ↓
e_t = z_t - z_g
        ↓
residual policy / model-free RL
        ↓
修正 VLA 参考 delta TCP
        ↓
动作适配层
        ↓
底层气压控制器
```

## 15. 推荐实验

以 pick and place 为主任务，逐步比较：

1. VLA SFT 直接输出 delta TCP；
2. VLA SFT + RLT；
3. VLA SFT + goal predictor；
4. VLA SFT + Koopman error；
5. VLA SFT + Koopman error + residual RL。

评估指标：

- 成功率；
- 数据量需求；
- 目标位置误差；
- 不同物体初始位置泛化；
- 不同语言指令泛化；
- 在线微调样本效率；
- delta TCP 平滑度；
- 气压控制平滑度和安全性。

## 16. 最小可行版本

首个版本可以只实现：

```text
VLA hidden state
        ↓
task latent encoder
        ↓
goal predictor
        ↓
error state
        ↓
residual policy
        ↓
修正参考 delta TCP
```

稳定后再加入显式 Koopman dynamics、RLT actor-critic 和在线 residual RL。
