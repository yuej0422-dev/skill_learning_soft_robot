# 实物机器人关键检查

这里仅记录正式部署前仍需使用的关键硬件检查。除非命令另有说明，均从仓库根目录
`/home/cao/skill_learning_soft_robot` 执行。

## 1. LuMo 状态读取

只连接 LuMo，不发送气压：

```bash
python soft_vla/scripts/real_robot/diagnostics/hil_read_lumo_state.py --hardware-enabled
```

先确认位姿、速度、时间戳和刚体 ID 正常，再继续其他硬件检查。

## 2. 气压串口

先用低压力检查串口和 16 通道数据包：

```bash
python soft_vla/scripts/real_robot/diagnostics/test_pressure_driver.py \
  --hardware-enabled \
  --packet-channels 16 \
  --pressure 0.1
```

串口设备可通过 `--port` 指定。

## 3. 50 Hz 安全空载检查

同时读取 LuMo 状态并发送受限的恒定低压力：

```bash
python soft_vla/scripts/real_robot/diagnostics/hil_idle_50hz.py \
  --hardware-enabled \
  --packet-channels 16 \
  --initial-pressure-norm 0.0 \
  --duration-s 5
```

脚本限制 `initial-pressure-norm` 不得超过 `0.2`，结束时发送零压力。

## 4. 三路相机

```bash
python soft_vla/scripts/real_robot/diagnostics/smoke_live_cameras.py
```

确认三路画面来源、尺寸和颜色顺序正常。

## 5. Xbox 手柄

```bash
bash soft_vla/scripts/real_robot/diagnostics/smoke_xbox_controller.sh
```

如需指定设备：

```bash
GAMEPAD_DEVICE_PATH=/dev/input/eventX \
  bash soft_vla/scripts/real_robot/diagnostics/smoke_xbox_controller.sh
```

## 6. 人工介入 live dry-run

该入口启用相机、LuMo、手柄、SmolVLA 推理和 episode 记录，但强制关闭真实气压输出：

```bash
bash soft_vla/scripts/real_robot/diagnostics/smolvla_deploy_human_intervention_live_dryrun.sh
```

确认上述检查全部正常后，再使用 `scripts/real_robot/deploy/` 下的正式部署入口。
