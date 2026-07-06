# Pick and Place 任务示例与 Mark 点布置

## 1. 任务定位

后续技术文档统一使用 pick and place 作为示例任务。该任务足够典型，包含接近、对准、抓取、移动、放置和撤离等阶段，同时比复杂装配任务更适合作为第一类技能学习和 VLA 微调对象。

典型阶段为：

```text
approach object
        ↓
align TCP
        ↓
grasp / attach
        ↓
lift
        ↓
move to target region
        ↓
descend
        ↓
place / release
        ↓
retreat
```

## 2. TCP 末端 Mark 点

TCP 末端需要布置刚体 marker cluster，用于恢复末端 6D pose。

建议：

- 使用 3-4 个非共线 marker；
- marker 相对 TCP 工具坐标系固定；
- 布置尽量非对称，避免姿态解算歧义；
- 不遮挡抓取接触面；
- 不改变末端执行器的主要接触特性。

动捕系统输出 TCP 位姿后，应统一转换到机器人控制使用的世界坐标系或基坐标系。

## 3. 物体 Mark 点

pick and place 中的被操作物体需要独立刚体 marker。

建议：

- 每个物体使用独立 marker cluster；
- 使用 3-4 个非对称 marker；
- 避开抓取面、放置接触面和长期遮挡面；
- marker 固定牢靠，避免抓取过程中相对物体滑动；
- 尽量降低 marker 对物体质量分布和接触摩擦的影响。

如果物体较小，可优先保证位置跟踪稳定；姿态估计可根据任务需要决定是否使用。

## 4. 桌面与目标区域

桌面或实验平台需要定义稳定世界坐标。

建议：

- 设置固定 marker cluster 或标定板；
- 定义桌面平面；
- 定义 pick 区域和 place 区域；
- 记录目标区域中心、尺寸和允许误差；
- 保证 reset 后坐标定义不变。

对于放置任务，目标可以表示为：

\[
g=(p_{target}, R_{target})
\]

也可以简化为目标区域中心：

\[
g=p_{target}
\]

## 5. 软臂本体 Mark 点

如果首版只做 TCP 状态技能学习，软臂本体 marker 不是必须项。

如果需要建模软体形变、迟滞或中间构型，建议沿软臂布置 2-4 组截面 marker：

- 每组 3 个小 marker；
- 尽量不影响软臂弯曲和气动变形；
- 避免 marker 在常见动作中互相遮挡；
- 可用于估计软臂中间形状或构型特征。

这些本体 marker 可作为后续视觉状态恢复和动力学分析的辅助监督。

## 6. RGB-D 与动捕外参标定

如果同时采集 RGB-D，需要将相机坐标系与动捕世界坐标系对齐。

建议使用：

- AprilTag；
- ArUco；
- 棋盘格；
- 或已知几何尺寸的标定物。

目标是获得：

\[
{}^WT_C
\]

即从相机坐标系到动捕世界坐标系的变换。

所有 RGB-D 反投影点云、动捕 TCP 状态和物体位姿最终应统一到同一世界坐标系中。

## 7. 数据记录建议

每个时间步建议记录：

\[
\mathcal D_t=
\{s_t^{mocap}, o_t^{mocap}, g_t, x_{tcp}^{des}, \dot{x}_{tcp}^{des}, pressure_t, I_t, D_t, \tau_t\}
\]

其中：

- \(s_t^{mocap}\)：TCP 状态；
- \(o_t^{mocap}\)：物体状态；
- \(g_t\)：放置目标或任务目标；
- \(pressure_t\)：气压控制量；
- \(I_t,D_t\)：RGB-D 观测；
- \(\tau_t\)：统一时间戳。

数据记录要明确 episode 边界、reset 状态和任务成功标记。
