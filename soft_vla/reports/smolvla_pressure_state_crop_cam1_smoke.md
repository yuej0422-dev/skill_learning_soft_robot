# SmolVLA Pressure State + Cam1 Crop Smoke

## 目标

- 数据集：`/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp_pressure_state`
- 新增配置：`configs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke.yaml`
- 不覆盖旧训练配置；该新配置设置 `reports.update_shared: false`，训练产物写入独立输出目录。

## 图像处理

- `observation.images.cam_1` 在进入 SmolVLA preprocessor 前裁掉右侧 `1/5`。
- 实测形状：`[1, 3, 720, 1280] -> [1, 3, 720, 1024]`。
- `cam_2`、`cam_3` 不裁剪，保持 `[1, 3, 480, 640]`。

## 归一化策略

- `observation.state[12]`：夹爪状态，强制 identity，processor mean=0/std=1。
- `action[6]`：夹爪目标动作，强制 identity，processor mean=0/std=1。
- 其余 `observation.state` 维度：使用数据集默认 mean/std。
- 其余 `action` 维度：使用数据集默认 mean/std。

说明：LeRobot/SmolVLA 当前 normalizer 通常按整段 feature 应用同一种规范化方式，因此这里通过 patch processor stats 的方式实现“同一向量内仅夹爪 identity，其余维度默认 mean/std”的混合归一化。

## Smoke 训练结果

- 命令：`python scripts/train.py --config configs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke.yaml --overwrite`
- 训练模式：全参微调
- 步数：`2`
- loss：step1 `0.4903168`，step2 `1.3689610`
- 可训练参数：`402,737,376 / 450,046,176`
- 可训练比例：`0.89488`
- 峰值 GPU 显存：约 `4.99 GB`
- 权重更新：已确认，`max_abs_difference=0.00018310546875`

## 产物

- 输出目录：`outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke`
- checkpoint：`outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke/checkpoints/000002/pretrained_model`
- first batch 检查：`outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke/first_batch_report.json`
- mixed normalization 检查：`reports/mixed_normalization_report.md`

## 离线推理测试

- 配置：`configs/inference_real_7_03_1_pressure_state_crop_cam1_smoke.yaml`
- 命令：`python scripts/offline_inference.py --config configs/inference_real_7_03_1_pressure_state_crop_cam1_smoke.yaml`
- 帧数：`3`
- action chunk：`[1, 50, 19]`
- 单步 action：`[19]`
- first sample raw cam_1：`[3, 720, 1024]`
- first sample processed cam_1：`[1, 3, 720, 1024]`
- mean latency：`227.266 ms`
- p95 latency：`341.238 ms`
- peak GPU memory：`0.927 GB`
- 离线 MAE：`0.053788`
- 夹爪预测值：`[0.0]`
