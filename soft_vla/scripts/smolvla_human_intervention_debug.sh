#!/usr/bin/env bash
set -euo pipefail

# Xbox 人工介入链路隔离调试入口。
# 不加载 SmolVLA、不读取相机、不打开串口、不下发机械臂压力；
# 只检查：evdev 原始事件 -> 手柄解码 -> 进程队列 -> 10Hz 映射/仲裁。
#
# 推荐运行：
#   GAMEPAD_DEBUG_PRINT_ALL=1 bash soft_vla/scripts/smolvla_human_intervention_debug.sh
# 然后依次按 A/Y/X/B、移动摇杆和 RT/LT，观察 debug_listener/debug_upper 输出。

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== 调试运行 =====
DURATION_S=${DURATION_S:-30}                 # 调试持续时间，单位秒
UPPER_HZ=${UPPER_HZ:-10}                     # 模拟上层人工介入循环频率
GAMEPAD_DEBUG_PRINT_ALL=${GAMEPAD_DEBUG_PRINT_ALL:-0}  # 1: 每个 raw/upper tick 都打印；0: 只打印变化/按键

# ===== 手柄设备 =====
GAMEPAD_BACKEND=${GAMEPAD_BACKEND:-evdev}    # 当前只支持 evdev
GAMEPAD_DEVICE_PATH=${GAMEPAD_DEVICE_PATH:-} # 空表示自动找；也可指定 /dev/input/event27
GAMEPAD_POLL_HZ=${GAMEPAD_POLL_HZ:-50}       # 手柄监听进程轮询频率

# ===== 防误触与释放 =====
GAMEPAD_DEADZONE=${GAMEPAD_DEADZONE:-0.15}              # 摇杆/扳机超过该值才触发人工接管
INTERVENTION_RELEASE_DEADZONE=${INTERVENTION_RELEASE_DEADZONE:-0.10}  # 接管后低于该值才释放
INTERVENTION_RELEASE_TICKS=${INTERVENTION_RELEASE_TICKS:-1}           # 连续多少个 10Hz tick 低输入后释放

# ===== 人工 action 映射 =====
# 坐标映射：左摇杆左/右 -> z-/z+，左摇杆后/前 -> x+/x-，RT/LT -> y+/y-。
# xz 平面固定单轴锁：同一时刻只输出 +x/-x/+z/-z 中一个方向。
HUMAN_MAX_DELTA_POS=${HUMAN_MAX_DELTA_POS:-0.005}  # 单个 10Hz tick 最大平移 delta，单位 m
HUMAN_MAX_DELTA_ROT=${HUMAN_MAX_DELTA_ROT:-0.025}  # 单个 10Hz tick 最大旋转 delta，单位 rad
ROTATION_ENABLED=${ROTATION_ENABLED:-1}            # 1: 启用右摇杆旋转
ROTATION_AXIS=${ROTATION_AXIS:-pitch_yaw}          # pitch_yaw: 右摇杆同一时刻只输出 pitch 或 yaw

# ===== 人工目标积分 =====
HUMAN_TARGET_INTEGRATION=${HUMAN_TARGET_INTEGRATION:-1}  # 1: 同方向连续输入累计 target delta
HUMAN_TARGET_MAX_POS_OFFSET=${HUMAN_TARGET_MAX_POS_OFFSET:-0.01}  # 积分平移上限，单位 m
HUMAN_TARGET_MAX_ROT_OFFSET=${HUMAN_TARGET_MAX_ROT_OFFSET:-0.05}  # 积分旋转上限，单位 rad
HANDOVER_BLEND_STEPS=${HANDOVER_BLEND_STEPS:-2}     # 人工/VLA 切换平滑步数；本调试仅用于观察仲裁
BLEND_TCP_ONLY=${BLEND_TCP_ONLY:-1}                 # 1: 只平滑 TCP，不平滑夹爪
BLEND_GRIPPER=${BLEND_GRIPPER:-0}                   # 1: 也平滑夹爪

args=(
  soft_vla/scripts/deploy_smolvla_human_intervention_debug.py
  --duration-s "$DURATION_S"
  --upper-hz "$UPPER_HZ"
  --gamepad-backend "$GAMEPAD_BACKEND"
  --gamepad-poll-hz "$GAMEPAD_POLL_HZ"
  --gamepad-deadzone "$GAMEPAD_DEADZONE"
  --intervention-release-deadzone "$INTERVENTION_RELEASE_DEADZONE"
  --intervention-release-ticks "$INTERVENTION_RELEASE_TICKS"
  --human-max-delta-pos "$HUMAN_MAX_DELTA_POS"
  --human-max-delta-rot "$HUMAN_MAX_DELTA_ROT"
  --rotation-axis "$ROTATION_AXIS"
  --human-target-max-pos-offset "$HUMAN_TARGET_MAX_POS_OFFSET"
  --human-target-max-rot-offset "$HUMAN_TARGET_MAX_ROT_OFFSET"
  --handover-blend-steps "$HANDOVER_BLEND_STEPS"
)

[[ -n "$GAMEPAD_DEVICE_PATH" ]] && args+=(--gamepad-device-path "$GAMEPAD_DEVICE_PATH")
[[ "$GAMEPAD_DEBUG_PRINT_ALL" == "1" ]] && args+=(--print-all)
[[ "$ROTATION_ENABLED" == "1" ]] && args+=(--rotation-enabled)
[[ "$HUMAN_TARGET_INTEGRATION" != "1" ]] && args+=(--no-human-target-integration)
[[ "$BLEND_TCP_ONLY" == "1" ]] && args+=(--blend-tcp-only)
[[ "$BLEND_GRIPPER" == "1" ]] && args+=(--blend-gripper)

echo "[soft_vla] human intervention debug only: no SmolVLA, no camera, no serial pressure output"
echo "[soft_vla] GAMEPAD_BACKEND=$GAMEPAD_BACKEND GAMEPAD_DEVICE_PATH=${GAMEPAD_DEVICE_PATH:-auto} DURATION_S=$DURATION_S"
"$PY" "${args[@]}"
