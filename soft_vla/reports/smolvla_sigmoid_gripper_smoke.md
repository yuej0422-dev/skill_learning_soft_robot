# SmolVLA Sigmoid Gripper Smoke

## 改动

- 训练保持全参微调。
- 夹爪动作维度为 `action[6]`。
- 夹爪动作误差权重为 `1.0`，与其它 action 维度一致。
- 夹爪动作不使用 clip 约束。
- 训练时将最终预测的 `action[6]` 当作 logit，单独经过 `sigmoid` 后与 0/1 目标计算 MSE。
- 推理时同样对最终输出的 `action[6]` 做 `sigmoid(action[6])`，dry-run safety filter 只限制 TCP 前 6 维，不再阈值化夹爪。

## 配置

- 训练配置：`configs/smolvla_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke.yaml`
- 推理配置：`configs/inference_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke.yaml`
- 训练输出：`outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke`
- 推理输出：`outputs/offline_inference_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke`

## 训练 Smoke

- 命令：`python scripts/train.py --config configs/smolvla_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke.yaml --overwrite`
- 步数：`2`
- 模式：全参微调
- 可训练参数：`402,737,376 / 450,046,176`
- step1 loss：`3.1074529`
- step2 loss：`2.6699212`
- step1 gripper sigmoid range：`[0.4950389, 0.5258659]`
- step2 gripper sigmoid range：`[0.4047245, 0.5826761]`
- gripper weight：`1.0`
- peak GPU memory：`4.991 GB`
- 权重更新：已确认，`max_abs_difference=0.00018310546875`

## 推理 Smoke

- 命令：`python scripts/offline_inference.py --config configs/inference_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke.yaml`
- 帧数：`3`
- action chunk：`[1, 50, 19]`
- 单步 action：`[19]`
- cam_1 raw：`[3, 720, 1024]`
- cam_1 processed：`[1, 3, 720, 1024]`
- gripper output values：`[0.500116765499115, 0.5102718472480774, 0.5293283462524414]`
- mean latency：`235.217 ms`
- p95 latency：`353.281 ms`
- peak GPU memory：`0.927 GB`

结论：sigmoid bounded gripper 路径已跑通；夹爪输出为连续 0-1 范围内的小数，不是 clip 或 threshold 后的二值。
