# 软体机械臂实物调试流程

本文档从“非实物验证已通过”开始，目标是在接入真实硬件时按低风险顺序调试。除非命令显式带 `--hardware-enabled`，脚本都不会连接真实压力驱动或发送压力。

## 0. 前置状态

先确认离线链路仍然通过：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/run_nonhardware_control_validation.py \
  --episode-index 0 \
  --max-frames 3 \
  --device cpu
```

期望输出：

```text
ok: true
failed: []
```

临时 smoke 产物只应放在 `/tmp`，通过后可以删除；不要删除仓库中的正式测试脚本和配置。

当前实物部署默认 checkpoint / run：

- Koopman: `motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt`
- KORL AWAC 前馈: `motion_control_training/KORL/runs/feedforward/awac_quadq_2k_eval_2x256/best.pt`
- pressure MLP 前馈: `motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt`
- SmolVLA: `soft_vla/outputs/full_runs/smolvla_full_full20000_bs8_20260704_180614/checkpoints/020000/pretrained_model`
- LeRobot 数据集: `lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp`

上位机部署时建议整体复制仓库根目录 `skill_learning_soft_robot`，至少包含：

- `soft_vla/`
- `motion_control_training/`
- `lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/`

脚本默认 checkpoint 使用仓库相对路径；`soft_vla/configs/motion_control_eval.yaml` 中记录的是当前开发机绝对路径，搬到上位机后如果使用该 yaml，需要按上位机路径更新。

推荐使用打包脚本生成精简部署包。实物完整调试包包含 SmolVLA 正式模型和 LeRobot RGB 视频，但排除 archive、training state、历史 checkpoint、depth videos、AutoDL 产物和原始采集目录：

```bash
cd /home/cao/skill_learning_soft_robot

soft_vla/scripts/real_robot/components/package_real_robot_deploy.sh \
  --with-smolvla \
  --with-smolvla-extra \
  --with-lerobot-videos \
  --output /home/cao/skill_learning_soft_robot_real_robot_deploy.tar.gz
```

当前估算：

- motion-control 最小包：几十 MB 级。
- 加 `--with-smolvla`：额外约 `865M`。
- 加 `--with-smolvla-extra`：额外带 `015000` 和 `010000` 两个 SmolVLA checkpoint，约 `1.7G`。
- 加 `--with-lerobot-videos`：额外约 `699M`。
- 当前三 checkpoint 实物完整调试包未压缩约 `3.3G`，压缩后约 `2.8G`，因为 `.safetensors` 和 `.mp4` 已经不太可压缩。

## 1. 硬件信息确认

确认串口、LuMo IP、刚体 ID、压力 packet 通道数：

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
ls -l /dev/serial/by-id/* 2>/dev/null
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python -m serial.tools.list_ports -v
```

当前控制串口为 CH340：

```text
/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 -> ../../ttyUSB0
```

优先使用 `/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0`，比 `/dev/ttyUSB0`
更稳定。若 CH340 只在日志里短暂出现后消失，通常是 Ubuntu 的 `brltty` 抢占了串口：

```bash
sudo systemctl stop brltty-udev.service brltty.service
sudo systemctl mask brltty-udev.service brltty.service
```

若打开串口报 `Permission denied`，确认当前用户在 `dialout` 组：

```bash
groups
sudo usermod -aG dialout $USER
```

加组后需要重新登录，或打开新 shell 执行 `newgrp dialout` 后再运行串口测试。

当前默认配置：

- LuMo IP: `192.168.140.1`
- rigid body id: `1`
- serial: `/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0`
- baudrate: `115200`
- pressure packet channels: 默认 `16`

注意：压力控制接口固定按 16 个 double 解包，前 12 路为本体压力，后 4 路为夹爪压力。
不要向真实控制器发送 12 路 packet，否则下位机会按 16 路协议错位解析后续帧。

## 2. 只读动捕状态

只连接 LuMo，不打开串口，不发送压力：

Ubuntu 直连 Windows/FZMotion 上位机时，`--ip` 是发布动捕数据的 Windows/FZMotion
主机 IP。若 Windows 端配置为 `192.168.140.1`，Ubuntu 有线口需要在同一网段，
例如：

```bash
nmcli connection modify "有线连接 1" \
  ipv4.addresses 192.168.140.2/24 \
  ipv4.method manual \
  ipv6.method ignore \
  connection.autoconnect yes
nmcli connection up "有线连接 1"
ip -br addr show enp12s0
```

若 `ping 192.168.140.1` 无响应，再看二层是否可达：

```bash
ip neigh show dev enp12s0
```

`FAILED` 通常表示 Windows 有线网卡不是 `192.168.140.1`、网线/网口不通，或 Windows
防火墙/网络配置阻止访问。此时先在 Windows 上确认该网卡 IPv4，再把下面命令中的
`--ip` 改成 Windows 端实际地址。

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/hil_read_lumo_state.py \
  --hardware-enabled \
  --ip 192.168.140.1 \
  --rigid-body-id 1 \
  --receive-timeout-ms 1000 \
  --samples 100 \
  --frequency 50
```

验收：

- 能稳定读到 12 维 state。
- 位置单位为米，角度单位为弧度。
- 没有 NaN/Inf。
- 50 Hz 读取 p95 小于 20 ms。

## 3. 压力驱动 mock 与真实零压

先 mock：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/test_pressure_driver.py \
  --packet-channels 16 \
  --pressure 0.0
```

真实串口只发零压：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/test_pressure_driver.py \
  --real \
  --port /dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 \
  --baudrate 115200 \
  --packet-channels 16 \
  --pressure 0.0
```

验收：

- 串口能打开。
- 发送字节数为 `packet_channels * 8`。
- 关闭时会再次发送 zero packet。

## 4. 50 Hz HIL Idle

目标：只读状态，并持续发送安全初始压力。建议第一次使用 zero pressure。

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/hil_idle_50hz.py \
  --hardware-enabled \
  --ip 192.168.140.1 \
  --rigid-body-id 1 \
  --port /dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 \
  --packet-channels 16 \
  --duration-s 5 \
  --initial-pressure-norm 0.0 \
  --log-jsonl /tmp/soft_vla_hil_idle.jsonl
```

验收：

- 50 Hz loop 无异常退出。
- p95 小于 20 ms。
- watchdog/timeout 未触发。
- 停止时发送 zero pressure。

## 5. 单点目标 mock

在不碰硬件的情况下先跑单点目标控制：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/debug_single_point_target_real.py \
  --mock \
  --target-delta 0.001,0,0,0,0,0 \
  --duration-s 2 \
  --feedforward pressure_model \
  --feedback integral_lqr \
  --device cuda \
  --feedback-gain-scale 0.05 \
  --pressure-scale 0.2 \
  --log-jsonl /tmp/soft_vla_single_point_mock.jsonl
```

## 6. 单点目标实物低风险测试

第一次实物单点目标建议：

- 首次实物建议从 `target-delta` 1 mm 或 0.01 rad 量级开始；脚本不再硬限制目标幅度。
- `pressure-scale` 范围为 `[0, 1]`；首次实物仍建议从 `0.2` 开始。
- `feedback-gain-scale <= 0.05`
- `packet-channels=16`
- 现场有人看急停。

命令：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/debug_single_point_target_real.py \
  --hardware-enabled \
  --ip 192.168.140.1 \
  --rigid-body-id 1 \
  --port /dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 \
  --packet-channels 16 \
  --target-delta 0.001,0,0,0,0,0 \
  --duration-s 2 \
  --feedforward pressure_model \
  --feedback none \
  --pressure-scale 0.2 \
  --delta-tcp-scale 1.0 \
  --log-jsonl /tmp/soft_vla_single_point.jsonl \
  --plot-path /tmp/soft_vla_single_point.png
```

验收：

- 压力无突变。
- state 无 timeout。
- 机器人未超工作空间。
- 结束后 zero pressure。

## 7. Episode 小幅回放

第 7 点的控制语义固定为：

```text
LeRobot observation.state[:12] + LeRobot action[:6]
    -> 构造 10 Hz target/reference
    -> 展开为 5 个 50 Hz reference
LuMo measured state[:12]
    -> 与 reference 做状态层级闭环
    -> feedforward + Koopman integral feedback
    -> pressure driver
```

也就是说，数据集里的 `state` 只用于给每个数据 `action` 提供记录时的目标起点；真正闭环时，`current_state12` 和 `tracking_error` 必须来自 LuMo 实测状态。

实物回放前必须先跑离线 dry-run。这个 dry-run 不接 LuMo，只验证数学链路：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/dry_run_episode_motion_control.py \
  --episode-index 0 \
  --max-frames 20 \
  --feedforward pressure_model \
  --feedback integral_lqr \
  --delta-tcp-scale 0.1 \
  --pressure-scale 0.2 \
  --feedback-gain-scale 0.05 \
  --q-tcp6-weight 1.0 \
  --q-state-tail-weight 0.1 \
  --q-latent-weight 0.1 \
  --q-integral-weight 0.5 \
  --r-weight 10.0
```

然后跑 replay wiring mock。这里仍然不碰硬件，但会走 `replay_episode_real.py`
的同一套 target 构建和闭环接口，mock state 使用 perfect-tracking 方式更新：

`--max-frames` 指定回放多少个 10 Hz episode frame；`0` 或 `-1` 表示回放完整 episode。

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/replay/replay_episode_real.py \
  --mock \
  --episode-index 0 \
  --max-frames 5 \
  --feedforward pressure_model \
  --feedback integral_lqr \
  --delta-tcp-scale 0.1 \
  --pressure-scale 0.2 \
  --feedback-gain-scale 0.05 \
  --q-tcp6-weight 1.0 \
  --q-state-tail-weight 0.1 \
  --q-latent-weight 0.1 \
  --q-integral-weight 0.5 \
  --r-weight 10.0 \
  --log-jsonl /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_replay_mock.jsonl \
  --plot-path /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_replay_mock.png
```

真实小幅 episode replay 只有完成 HIL idle 与单点目标后再打开：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/replay/replay_episode_real.py \
  --hardware-enabled \
  --ip 192.168.140.1 \
  --rigid-body-id 1 \
  --port /dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 \
  --packet-channels 16 \
  --episode-index 0 \
  --max-frames 5 \
  --feedforward pressure_model \
  --feedback integral_lqr \
  --delta-tcp-scale 0.1 \
  --pressure-scale 0.2 \
  --feedback-gain-scale 0.05 \
  --q-tcp6-weight 1.0 \
  --q-state-tail-weight 0.1 \
  --q-latent-weight 0.1 \
  --q-integral-weight 0.5 \
  --r-weight 10.0 \
  --log-jsonl /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_replay_real_ep0_small.jsonl \
  --plot-path /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_replay_real_ep0_small.png
```

验收：

- 日志中 `target_source=lerobot_observation_state_plus_lerobot_action_delta`。
- 日志中 `closed_loop_state_source=lumo_measured_state`。
- 每个 50 Hz 周期都有 `measured_state`、`reference_state`、`tracking_error_tcp6`。
- 结束后发送 zero pressure。

闭环增益参数说明：

- `feedback-gain-scale`: 对最终反馈压力 `-K[e;q]` 做整体缩放，范围为 `[0, 1]`；实物初调优先从 `0.02~0.05` 开始。
- `q-tcp6-weight`: Q 中 TCP pose 六维权重，默认 `1.0`。
- `q-state-tail-weight`: Q 中 state 后六维速度/角速度权重，默认 `0.1`。
- `q-latent-weight`: Q 中 Koopman lifted latent 剩余维度权重，默认 `0.1`。
- `q-integral-weight`: Q 中积分误差 ny 维权重，默认 `0.5`。
- `r-weight`: R 中压力控制代价，默认 `10.0`；调大则反馈压力更保守。

如果使用 `fixed_k_integral`，先用同一组 Q/R 生成固定 K：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/components/build_fixed_k_integral.py \
  --output /tmp/soft_vla_fixed_k_integral.npz \
  --q-tcp6-weight 1.0 \
  --q-state-tail-weight 0.1 \
  --q-latent-weight 0.1 \
  --q-integral-weight 0.5 \
  --r-weight 10.0
```

## 8. SmolVLA 四进程部署框架

SmolVLA 实物部署框架按 4 个异步进程拆分：

1. `control_50hz`: 50 Hz 底层控制进程，只读取最新 50 Hz reference 并输出压力。
2. `upper_10hz`: 10 Hz action dispatch 进程，消费 action chunk，生成 5 个 50 Hz reference。
3. `smolvla_inference`: SmolVLA 推理进程，异步生成 action chunk，不能阻塞 50 Hz。
4. `async_logger`: 异步日志进程，消费各进程日志。

先跑 mock 四进程框架，不加载真实模型、不连接硬件。当前这条路径已经会初始化
底层 motion policy，默认 `pressure_model + fixed_k_integral`，但 state/pressure
IO 仍是 mock：

```bash
bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
```

验收：

- `ok=true`
- 4 个进程 exitcode 都是 0。
- `control_50hz` 步数约为 `duration_s * 50`。
- `upper_10hz` 步数约为 `duration_s * 10`。
- `smolvla_inference` 至少产生 1 个 chunk。
- logger 有记录。

然后跑真实 SmolVLA checkpoint 的非实物四进程 smoke。这里会真正加载
`pretrained_model`，默认从 LeRobot 数据集回放观测，异步生成 action chunk；底层
50 Hz 使用同一套 motion policy，但 IO 仍是 mock，不会打开串口：

```bash
VLA_BACKEND=smolvla MAX_INFERENCE_CHUNKS=1 DURATION_S=12 \
  bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
```

再跑实时三路相机 smoke。采集代码中 `cam_1` 是 ZED left，`cam_2/cam_3`
是两台 RealSense；训练 checkpoint 的输入键为
`observation.images.cam_1/cam_2/cam_3`，尺寸分别为 `720x1280`、
`480x640`、`480x640`，RGB float `[0,1]`。

注意：

- `/dev/video8` 是电脑内置摄像头，不是 ZED。
- Ubuntu 下真正的 ZED 节点当前是 `/dev/video0`/`/dev/video1`，设备名为 `ZED: ZED`。
- 默认 smoke 只允许选择设备名含 ZED 的节点；如果读到纯绿色/黑图会失败，不能自动降级到内置摄像头。
- 训练数据中的 `observation.images.cam_1` 是 ZED left 图，视频分辨率为 `1280x720`，不是 `2560x720`。
- 采集脚本在 Windows 上请求 ZED USB side-by-side 流 `2560x720`，随后一分为二保存 left half，因此进入 LeRobot/SmolVLA 的 `cam_1` 是 `720x1280x3`。
- 当前 Ubuntu 上按采集脚本请求 `2560x720` 时，ZED 前几帧可能是纯绿色；相机源会等待连续可用帧后才通过 smoke，避免早期坏帧进入推理。
- 低分辨率诊断 `672x376` 会明显比训练数据模糊，只能用于排查是否读到真正 ZED left，不能作为 VLA 部署输入。
- cam2/cam3 不依赖 `pyrealsense2` 枚举顺序，当前显式绑定为 `cam2=401522072797`、`cam3=408322072769`。

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/smoke_live_cameras.py \
  --zed-eye left \
  --output-dir soft_vla/artifacts/real_robot/camera_smoke_live
```

验收：

- `ok=true`。
- `observation.images.cam_1` shape 为 `[720, 1280, 3]`。
- `observation.images.cam_2/cam_3` shape 为 `[480, 640, 3]`。
- 每路 `usable=true`，不能是黑图或纯色图。

然后跑真实 SmolVLA + 实时三路相机 + 底层 motion policy 的非实物 smoke：

```bash
VLA_BACKEND=smolvla LIVE_OBSERVATION=1 MAX_INFERENCE_CHUNKS=1 DURATION_S=20 \
  bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
```

如果要同时连接真实 LuMo 动捕、但仍不打开串口压力输出，使用 `STATE_HARDWARE=1`。
这条路径会读真实 `192.168.140.1` 动捕和三路相机，加载真实 SmolVLA checkpoint，
底层 motion policy 仍会计算压力，但 pressure driver 是 mock，不会向实物下发：

```bash
STATE_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 \
  CAMERA_PREVIEW=1 MODE=receding_horizon \
  MAX_INFERENCE_CHUNKS=1 DURATION_S=20 \
  RUN_LABEL=smolvla_live_mocap_receding_horizon_mock_pressure \
  bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
```

预览说明：

- `CAMERA_PREVIEW=1` 会额外启动 `camera_preview` 进程。
- 默认预览尺寸由 `CAMERA_PREVIEW_SCALE=0.5` 控制，频率由 `CAMERA_PREVIEW_FPS=10` 控制。
- 当前环境已安装带 QT5 highgui 的 `opencv-python`，可以直接 `imshow` 开窗；若后续环境没有显示能力，脚本仍会把最新 mosaic 写到本次输出目录的 `camera_preview_latest.jpg`。

最后才允许打开实物完整链路：

```bash
RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 \
  bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
```

可一次性尝试几种部署模式：

```bash
/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/benchmark_smolvla_deploy_modes.py \
  --real-policy \
  --modes receding_horizon temporal_ensemble chunk single_step \
  --duration-s 60 \
  --max-inference-chunks 1 \
  --device cuda
```

模式含义：

- `receding_horizon`: 默认推荐。推理进程持续产出 chunk，10 Hz 上层每隔 `replan_interval` 步切换到新 chunk，旧队列没用完时保留可执行动作。
- `temporal_ensemble`: TE。按 absolute upper step 对齐历史 chunk，对 TCP 六维做加权融合；夹爪先阈值化/投票式处理，避免连续平均直接下发。
- `chunk`: 固定 chunk 顺序消费，适合最小化策略逻辑变量。
- `single_step`: 每次只消费一个动作。当前 SmolVLA 接口产出的是 50 步 chunk，single_step 只取第 0 个动作并要求每个 10 Hz tick 都有 fresh chunk，因此只作为压力测试/对照，不建议作为当前实物默认模式。

当前真实 checkpoint 非实物 smoke 结果：

- `receding_horizon`: 通过，真实 LuMo + 实时三相机 + 真实 SmolVLA，pressure mock，首个 chunk 推理约 368 ms。
- `temporal_ensemble`: 通过，真实 LuMo + 实时三相机 + 真实 SmolVLA，pressure mock，首个 chunk 推理约 373 ms。
- `chunk`: 通过，真实 LuMo + 实时三相机 + 真实 SmolVLA，pressure mock，首个 chunk 推理约 374 ms。
- `single_step`: 进程通过，但 10 Hz 上层全部 fallback，原因如上，不进入实物验收默认路径。

真实 SmolVLA 接真实压力硬件只有在以下全部完成后再打开：

- HIL idle 通过。
- 单点目标实物通过。
- episode 小幅回放通过。
- 确认压力 packet 通道数。
- 确认 SmolVLA action 仍为 10 Hz delta TCP，不按 50 Hz 消费 chunk。
- 实时三路相机 smoke 通过，且 ZED 使用 left 图像。
- 先用 `LIVE_OBSERVATION=1` 的 Mock IO 跑通真实 SmolVLA 推理。

当前完整链路为：

`live cameras + latest 13D state -> SmolVLA inference -> 10 Hz chunk dispatch -> ReferenceGenerator -> 50 Hz MotionControlRuntime -> SafetyManager -> 16D pressure packet`

`STATE_HARDWARE=1` 时，50 Hz 进程只打开 LuMo state source；`RUN_HARDWARE=1`
时，50 Hz 进程同时打开 LuMo state source 和 serial pressure driver。未设置
`RUN_HARDWARE=1` 时，压力输出始终是 MockPressureDriver。

正式 SmolVLA 推理开始前，`control_50hz` 会先构建底层 motion policy。终端会打印：

```text
[soft_vla] requested motion policy: ...
[soft_vla] motion policy initialized: feedforward=..., feedback=..., fixed_k_source=...
```

`smolvla_inference` 会等待该初始化完成后再加载/推理 SmolVLA，默认等待上限由
`MOTION_POLICY_READY_TIMEOUT_S=120` 控制。

默认 `WAIT_FOR_START_KEY=1`。motion policy 初始化完成、SmolVLA 权重加载完成后，
主进程会提示按任意键开始；按键前不会进入 50 Hz/10 Hz 执行循环，也不会打开串口
进入持续控制。`DURATION_S` 从按键后开始计时。

默认 `WAIT_FOR_FIRST_ACTION_CHUNK=1`。按键后 inference 进程会先打开相机并推理第一段
action chunk；`upper_10hz` 和 `control_50hz` 会等第一段 chunk 到达后再进入执行循环。
这样正式运行不会在前几秒因为 action queue 为空而持续下发 `queue_underrun_fallback`。
等待上限由 `FIRST_ACTION_TIMEOUT_S=120` 控制。

运行中默认每 10 个 10 Hz step 打印一次：

```text
[soft_vla] upper_step=... elapsed_ms=... source=... action=[dx=..., dy=..., dz=..., droll=..., dpitch=..., dyaw=..., gripper=...]
```

频率由 `ACTION_PRINT_INTERVAL_STEPS=10` 控制，设为 `0` 或负数可关闭。

SmolVLA 的 `observation.state` 是 13 维：

```text
[LuMo state12, gripper_open]
```

其中 `state12` 来自动捕。LuMo 不提供夹爪传感器，因此 `gripper_open` 是估计值：
初始为 `INITIAL_GRIPPER_OPEN=1`，之后跟随上层 action/reference 中的夹爪命令更新。

SmolVLA action 的最后一维先经过阈值后处理：

- `action_gripper > GRIPPER_OPEN_THRESHOLD`：置为 `1`，表示 open。
- `action_gripper < GRIPPER_CLOSE_THRESHOLD`：置为 `0`，表示 close。
- 中间区间：保持上一条 gripper action 不变。

默认：

```bash
GRIPPER_CLOSE_THRESHOLD=0.2
GRIPPER_OPEN_THRESHOLD=0.8
```

后四路压力映射为：open -> `[3,0,0,0]`，close -> `[0,3,0,0]`。

中途 `Ctrl+C` 或收到 `SIGTERM` 时，主进程会通知所有子进程退出；`control_50hz`
会连续发送 3 次 zero pressure，再关闭串口。日志 summary 中应能看到
`interrupted_by=SIGINT` 和 `safe_exit_zero_packets=3`。

## 9. 记录与停机

每次实物测试必须保存：

- JSONL 日志。
- 控制周期 p50/p95/p99。
- 最大压力。
- 最大 pressure slew。
- state timeout 次数。
- command timeout 次数。
- safety flags。

任何异常立即：

1. 急停。
2. 发送 zero pressure。
3. 关闭串口。
4. 清空积分项。
5. 保存日志，不继续跑下一阶段。
