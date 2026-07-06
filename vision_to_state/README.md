# LeRobot RGB-D 到 12 维 TCP State 训练流程

本文档是 `vision_to_state/` 下的训练流程入口。ZIP 转 LeRobot 的流程已经移到项目根目录：

```text
/home/yuej/skill_learning_soft_robot/lerobot_conversion/README.md
```

## 1. 训练目标

从主视角相机 `camera2` 的 RGB-D 数据监督学习 12 维 `observation.state`：

```text
[x, y, z, rx, ry, rz, vx, vy, vz, wx, wy, wz]
```

当前训练只重点优化前 6 维 TCP pose：

- target 使用 `meta/stats.json` 中 `observation.state` 的 `q01/q99` 做 min-max 归一化。
- 归一化 target 会 clip 到 `[-1, 1]`。
- loss 权重为前 6 维 pose = 1，后 6 维 velocity = 0。
- 模型仍然输出 12 维，后 6 维只是保留接口，不作为优化目标。

## 2. 数据结构

当前使用的数据集：

```text
vision_to_state/lerobot_v3_cam2_file002_24m18s
```

关键文件：

```text
data/chunk-000/file-000.parquet
videos/observation.images.cam_2/chunk-000/file-000.mp4
depth_raw/cam_2/episode_*.npz
meta/stats.json
meta/extra/depth_index.parquet
```

RGB 图像从 LeRobot v3 的 MP4 中按帧解码。Depth 不在 LeRobot 原生 video 字段里，而是以无损 `uint16` NPZ sidecar 保存。训练脚本用 `meta/extra/depth_index.parquet` 根据下面三个键对齐 depth：

```text
episode_index + frame_index + camera_name
```

读取到的 RGB-D 会做同样裁剪：

- 左侧裁掉 1/4。
- 顶部裁掉 1/10。
- resize 到训练参数指定的 `image_size`。
- depth clip 到 `depth_clip_mm`，再缩放到 `[0, 1]`。

## 3. 环境

```bash
cd /home/yuej/skill_learning_soft_robot
conda activate soft_robot_state
```

该环境需要：

```text
torch
av
numpy
pyarrow
Pillow
tqdm
```

如果缺 PyAV：

```bash
/home/yuej/miniconda3/envs/soft_robot_state/bin/python -m pip install av
```

## 4. 运行训练

```bash
PYTHONPATH=vision_to_state/src \
/home/yuej/miniconda3/envs/soft_robot_state/bin/python \
  -m vision_to_state.train_lerobot_rgbd_state \
  --root /home/yuej/skill_learning_soft_robot/vision_to_state/lerobot_v3_cam2_file002_24m18s \
  --camera cam_2 \
  --run-dir /home/yuej/skill_learning_soft_robot/vision_to_state/runs/lerobot_cam2_file002_q01q99_pm1_vel0 \
  --epochs 15 \
  --batch-size 64 \
  --image-size 160 \
  --seq-len 1 \
  --pose-weight 1 \
  --velocity-weight 0 \
  --device auto \
  --amp
```

脚本会先构建内存缓存：

1. 顺序解码 `cam_2` MP4 得到 RGB。
2. 按 episode 批量读取 `depth_raw/cam_2/*.npz`。
3. 使用 `depth_index.parquet` 写入对应帧的 depth 通道。
4. 按 episode 切分 train/val/test。

## 5. 输出

默认输出目录：

```text
vision_to_state/runs/lerobot_cam2_file002_q01q99_pm1_vel0
```

主要产物：

```text
best.pt
last.pt
history.json
normalizer_q01_q99.json
config_used.json
test_metrics.json
run_summary.md
```

当前一次训练结果：

```text
best val RMSE epoch: 3
best val pose RMSE: 0.038635
best val MAE epoch: 10
best val pose MAE: 0.018650
test pose RMSE: 0.028174
test pose MAE: 0.022410
```

velocity loss 权重为 0，所以 velocity 指标只作为记录，不代表训练优化目标。

## 6. 复跑建议

如果只想评估前 6 维 TCP pose，看：

```text
pose_rmse_mean
pose_mae_mean
rmse[0:6]
mae[0:6]
```

如果后续要让速度也准确，把命令里的：

```bash
--velocity-weight 0
```

改成较小值，例如：

```bash
--velocity-weight 0.05
```

如果要使用时序信息，把：

```bash
--seq-len 1
```

改成 `4` 或更大。训练脚本会自动保证窗口不跨 episode。
