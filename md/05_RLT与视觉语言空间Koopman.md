# RLT 与视觉语言空间 Koopman

## 1. 阶段定位

该阶段面向 VLA 后训练和在线 RL。其核心思想是：不直接在高维 VLA hidden states 上做强化学习，而是借鉴 RLT 思想提取紧凑任务状态，并在训练 RLT 的过程中引入视觉语言空间 Koopman encoder。

这里的 Koopman encoder 作用于 VLM/VLA hidden states 或 token embedding，不等同于状态空间 Koopman encoder。

## 2. 输入对象

VLA 或 VLM 的中间表征记为：

\[
H_t^{VLM/VLA}
=
[h_t^1,h_t^2,\ldots,h_t^L]
\]

其中 \(L\) 是 token 数量。

视觉语言空间 Koopman encoder 定义为：

\[
z_t^{VLK}
=
E_{koop}^{VL}(H_t^{VLM/VLA})
\]

该 latent 用于表达任务阶段、视觉语言语义和动作后果相关的紧凑动态信息。

## 3. 与状态空间 Koopman 的区别

状态空间 Koopman：

\[
z_t^S=E_{koop}^{state}(s_t)
\]

视觉语言空间 Koopman：

\[
z_t^{VLK}=E_{koop}^{VL}(H_t^{VLM/VLA})
\]

二者输入对象不同、训练目标不同、使用阶段不同。

不要求：

\[
z_t^S=z_t^{VLK}
\]

也不要求二者共享 encoder、矩阵或 latent 维度。

## 4. RLT 紧凑状态

RLT encoder 从 VLA hidden states 中提取紧凑 RL state：

\[
z_t^{RLT}=E_{RLT}(H_t^{VLM/VLA})
\]

可选地，将视觉语言空间 Koopman latent 纳入 RL 状态：

\[
x_t^{RL}=[z_t^{RLT},z_t^{VLK}]
\]

如果已有视觉本体状态或动捕状态，也可以扩展为：

\[
x_t^{RL}=[z_t^{RLT},z_t^{VLK},\hat s_t]
\]

具体使用哪种组合应由实验复杂度和数据规模决定。

## 5. Koopman 训练目标

视觉语言空间 Koopman 可以使用局部演化约束：

\[
z_{t+1}^{VLK}
\approx
A^{VL}z_t^{VLK}+B^{VL}a_t
\]

对应损失：

\[
\mathcal L_{koop}^{VL}
=
\left\|
z_{t+1}^{VLK}
-
(A^{VL}z_t^{VLK}+B^{VL}a_t)
\right\|^2
\]

也可以加入任务进度或目标误差约束，让该 latent 更适合后续 RL。

## 6. 偏差控制

在线 RL 不建议从零生成动作，而是围绕 VLA 参考动作或参考目标做偏差修正。

动作偏差：

\[
a_t=a_t^{ref}+\Delta a_t
\]

目标偏差：

\[
x_{tcp}^{des}=x_{ref}^{des}+\Delta x
\]

其中 \(\Delta a_t\) 或 \(\Delta x\) 由轻量 actor 输出。

这样可以保留 VLA 的先验能力，同时把在线 RL 限制在较小修正范围内，提高安全性和样本效率。

## 7. 推荐训练流程

1. 先完成 VLA SFT，使其能输出可执行参考动作或目标。
2. 冻结或半冻结 VLA 主干，训练 RLT encoder 提取紧凑状态。
3. 在 RLT 训练过程中加入视觉语言空间 Koopman encoder。
4. 训练轻量 actor-critic，对参考动作或目标做偏差修正。
5. 只在稳定后逐步扩大 RL 可调范围。
