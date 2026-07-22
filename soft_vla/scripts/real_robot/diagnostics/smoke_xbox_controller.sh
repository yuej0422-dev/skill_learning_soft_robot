#!/usr/bin/env bash
set -euo pipefail

# 第 1 层：只测试 Xbox 手柄，不启动视觉、不启动 SmolVLA、不接机械臂。
# 当前人工控制映射：
#   左摇杆左右 -> x，左摇杆前后 -> z，RT/LT -> 竖直 y。
#   右摇杆 -> pitch/yaw；同一时刻只按主导方向输出一个欧拉角方向。
#   A -> 夹爪关闭，Y -> 夹爪打开，X -> 成功结束，B -> 失败结束。
# 用法：
#   bash soft_vla/scripts/real_robot/diagnostics/smoke_xbox_controller.sh
#   GAMEPAD_DEVICE_PATH=/dev/input/eventX bash soft_vla/scripts/real_robot/diagnostics/smoke_xbox_controller.sh

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
cd "$ROOT"

GAMEPAD_BACKEND=${GAMEPAD_BACKEND:-evdev}
GAMEPAD_DEVICE_PATH=${GAMEPAD_DEVICE_PATH:-}
DURATION_S=${DURATION_S:-10}

echo "[soft_vla] listing input devices"
"$PY" -m soft_vla.human_intervention.xbox_controller --backend "$GAMEPAD_BACKEND" --list-devices

args=(--backend "$GAMEPAD_BACKEND" --duration-s "$DURATION_S" --print-events)
[[ -n "$GAMEPAD_DEVICE_PATH" ]] && args+=(--device-path "$GAMEPAD_DEVICE_PATH")

echo "[soft_vla] printing Xbox events for ${DURATION_S}s; move sticks and press A/Y/X/B"
"$PY" -m soft_vla.human_intervention.xbox_controller "${args[@]}"
