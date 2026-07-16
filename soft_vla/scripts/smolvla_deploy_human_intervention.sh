#!/usr/bin/env bash
set -euo pipefail

# 第 3 层：SmolVLA + Xbox 人工介入完整部署入口。
# 原始 soft_vla/scripts/smolvla_deploy.sh 不受影响；本脚本走新增 runtime。
#
# 推荐测试顺序：
#   1) 只测手柄：
#      bash soft_vla/scripts/smoke_xbox_controller.sh
#   2) 不接机械臂压力输出，但开启视觉/状态/手柄/SmolVLA：
#      bash soft_vla/scripts/smolvla_deploy_human_intervention_live_dryrun.sh
#   3) 完整实物控制：
#      REMOTE_CONTROL_DEBUG=1 SAVE_HUMAN_EPISODES=1 RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1   bash soft_vla/scripts/smolvla_deploy_human_intervention.sh
#      REMOTE_CONTROL_DEBUG=0 SAVE_HUMAN_EPISODES=1 RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1   bash soft_vla/scripts/smolvla_deploy_human_intervention.sh

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== 运行模式 =====
RUN_HARDWARE=${RUN_HARDWARE:-0}          # 1: 打开串口并真实下发压力；0: pressure mock
STATE_HARDWARE=${STATE_HARDWARE:-0}      # 1: 读取真实 LuMo 动捕；RUN_HARDWARE=1 时通常也应开启
VLA_BACKEND=${VLA_BACKEND:-mock}         # mock 或 smolvla
LIVE_OBSERVATION=${LIVE_OBSERVATION:-0}  # 1: 用实时 ZED left + 两路 RealSense；0: 用 replay 观测
CAMERA_PREVIEW=${CAMERA_PREVIEW:-0}      # 1: 另开进程显示三相机 resize mosaic
HUMAN_INTERVENTION=${HUMAN_INTERVENTION:-1}  # 本脚本固定走人工介入 runtime，保留该变量用于日志可读性
WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}  # 1: motion policy 和 SmolVLA 权重就绪后，按任意键再开始执行
WAIT_FOR_FIRST_ACTION_CHUNK=${WAIT_FOR_FIRST_ACTION_CHUNK:-1}  # 1: 第一段 action chunk 就绪后再启动 10Hz/50Hz 控制

# ===== 任务与 SmolVLA 执行器 =====
TASK=${TASK:-"pick up the apple and place it on the electronic scale"}  # SmolVLA language task
MODE=${MODE:-receding_horizon}           # receding_horizon / chunk / temporal_ensemble / single_step
DURATION_S=${DURATION_S:-0}              # 运行时长，单位秒；0 表示无上限，只由 ESC/Ctrl+C 终止
CHUNK_SIZE=${CHUNK_SIZE:-50}             # SmolVLA 每次输出 action chunk 长度
EXECUTION_HORIZON=${EXECUTION_HORIZON:-10}  # chunk 模式一次消费多少个 10Hz action
REPLAN_INTERVAL=${REPLAN_INTERVAL:-5}    # receding_horizon 每多少个 10Hz step 切换新 chunk
CHUNK_TRIGGER_MARGIN=${CHUNK_TRIGGER_MARGIN:-1}  # 提前请求下一段 chunk 的余量
CHUNK_EXPECTED_STALE_STEPS=${CHUNK_EXPECTED_STALE_STEPS:-2}  # 预期推理延迟对应的 stale step 数
CHUNK_WORST_STALE_STEPS=${CHUNK_WORST_STALE_STEPS:-5}  # 最坏推理延迟容忍的 stale step 数
MAX_INFERENCE_CHUNKS=${MAX_INFERENCE_CHUNKS:-}  # 空表示不限制；smoke/debug 可设 1
ACTION_PRINT_INTERVAL_STEPS=${ACTION_PRINT_INTERVAL_STEPS:-10}  # 每 N 个 10Hz step 打印 action；<=0 关闭
FIRST_ACTION_TIMEOUT_S=${FIRST_ACTION_TIMEOUT_S:-120}  # 等第一段 action chunk 的最长时间

# ===== 底层 motion policy =====
FEEDFORWARD=${FEEDFORWARD:-pressure_model}  # pressure_model / awac
FEEDBACK=${FEEDBACK:-fixed_k_integral}      # none / integral_lqr / fixed_k_integral
DEVICE=${DEVICE:-cuda}                     # SmolVLA 推理设备
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}       # SmolVLA delta TCP 缩放
PRESSURE_SCALE=${PRESSURE_SCALE:-1}         # 最终压力缩放，1 表示完整 0-3 bar 范围
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.1}  # feedback 增益缩放
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-0.5}    # integral feedback 的误差积分上限
MOTION_POLICY_READY_TIMEOUT_S=${MOTION_POLICY_READY_TIMEOUT_S:-120}  # 等 motion policy 初始化的最长时间
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}         # fixed_k_integral 离线 K 的 TCP6 tracking 权重
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}  # fixed_k_integral 状态尾部权重
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}     # fixed_k_integral Koopman latent 权重
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5} # fixed_k_integral 积分误差权重
R_WEIGHT=${R_WEIGHT:-50.0}                  # fixed_k_integral 控制代价权重

# ===== 动捕与串口 =====
LUMO_IP=${LUMO_IP:-192.168.140.1}       # Windows 动捕发送 IP
RIGID_BODY_ID=${RIGID_BODY_ID:-1}        # LuMo rigid body id
RECEIVE_TIMEOUT_MS=${RECEIVE_TIMEOUT_MS:-1000}  # 动捕接收超时
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}  # 气压控制串口

# ===== Xbox 手柄人工介入 =====
GAMEPAD_BACKEND=${GAMEPAD_BACKEND:-evdev}   # Linux 下默认 evdev
GAMEPAD_DEVICE_PATH=${GAMEPAD_DEVICE_PATH:-}  # 空表示自动找手柄；也可指定 /dev/input/eventX
GAMEPAD_DEBUG=${GAMEPAD_DEBUG:-0}  # 已不在正式部署中打印逐帧手柄日志；调试请用 smolvla_human_intervention_debug.sh
# 防误触：摇杆/扳机超过 GAMEPAD_DEADZONE 才进入人工接管；
# 已经接管后，输入低于 INTERVENTION_RELEASE_DEADZONE 才释放，形成滞回，避免轻微晃动反复触发。
GAMEPAD_DEADZONE=${GAMEPAD_DEADZONE:-0.15}
INTERVENTION_RELEASE_DEADZONE=${INTERVENTION_RELEASE_DEADZONE:-0.10}
INTERVENTION_RELEASE_TICKS=${INTERVENTION_RELEASE_TICKS:-1}
# 坐标映射固定为：左摇杆左/右 -> z-/z+，左摇杆后/前 -> x+/x-，RT/LT 控制竖直 y。
# 左摇杆 x/z 平面固定单轴锁：同一时刻只输出 +x/-x/+z/-z 中一个方向。
# 按键：A 关闭夹爪 gripper=0，Y 打开夹爪 gripper=1，X 记录成功并复位进入下一条，B 记录失败并复位进入下一条。
HUMAN_MAX_DELTA_POS=${HUMAN_MAX_DELTA_POS:-0.005}
HUMAN_MAX_DELTA_ROT=${HUMAN_MAX_DELTA_ROT:-0.025}
# 右摇杆旋转：默认开启 pitch_yaw；同一时刻只按主导方向输出 pitch 或 yaw，不输出 roll。
ROTATION_ENABLED=${ROTATION_ENABLED:-1}
ROTATION_AXIS=${ROTATION_AXIS:-pitch_yaw}
# 人工目标积分：同一方向连续输入会累计目标偏移；换方向会重置该方向累计，避免反向后继承旧偏移。
HUMAN_TARGET_INTEGRATION=${HUMAN_TARGET_INTEGRATION:-1}
HUMAN_TARGET_MAX_POS_OFFSET=${HUMAN_TARGET_MAX_POS_OFFSET:-0.02}  # 人工积分平移累计上限，单位 m
HUMAN_TARGET_MAX_ROT_OFFSET=${HUMAN_TARGET_MAX_ROT_OFFSET:-0.05}  # 人工积分旋转累计上限，单位 rad
HANDOVER_BLEND_STEPS=${HANDOVER_BLEND_STEPS:-2}
BLEND_TCP_ONLY=${BLEND_TCP_ONLY:-1}
BLEND_GRIPPER=${BLEND_GRIPPER:-0}
# 遥控调试：置 1 后，上层把 SmolVLA 输出强制为 [0,0,0,0,0,0,0.5]，
# 这样真实执行只会来自手柄接管动作，方便单独调遥控方向/速度/夹爪。
REMOTE_CONTROL_DEBUG=${REMOTE_CONTROL_DEBUG:-0}
SAVE_HUMAN_EPISODES=${SAVE_HUMAN_EPISODES:-1}  # 1: 保存人工介入 episode；0: 只运行不保存
# X/B 结束 episode 后的安全复位：先向底层直接发送 16 路 0 压，再等待机械臂泄压/回落。
EPISODE_END_RESET_SLEEP_S=${EPISODE_END_RESET_SLEEP_S:-7}
EPISODE_END_RESET_ZERO_PACKETS=${EPISODE_END_RESET_ZERO_PACKETS:-3}

# ===== 相机与夹爪状态 =====
ZED_INDEX=${ZED_INDEX:-}                 # 空表示自动找设备名含 ZED 的 video 节点
ZED_EYE=${ZED_EYE:-left}                 # cam_1 使用 ZED left
ZED_WIDTH=${ZED_WIDTH:-2560}             # ZED side-by-side 输入宽度；left half 进入 cam_1
ZED_HEIGHT=${ZED_HEIGHT:-720}
ZED_FPS=${ZED_FPS:-30}
REALSENSE_SERIAL_CAM2=${REALSENSE_SERIAL_CAM2:-401522072797}  # cam_2 RealSense 序列号
REALSENSE_SERIAL_CAM3=${REALSENSE_SERIAL_CAM3:-408322072769}  # cam_3 RealSense 序列号
ZED_WARMUP_USABLE_FRAMES=${ZED_WARMUP_USABLE_FRAMES:-10}      # ZED 启动后丢弃/等待的可用帧数
REALSENSE_WARMUP_USABLE_FRAMES=${REALSENSE_WARMUP_USABLE_FRAMES:-10}  # RealSense warmup 帧数
MIN_REALSENSE_MEAN=${MIN_REALSENSE_MEAN:-40}  # RealSense 图像亮度低于该均值认为不可用
INITIAL_GRIPPER_OPEN=${INITIAL_GRIPPER_OPEN:-1}  # LuMo 无夹爪传感器；state13 最后一维初始用该估计值
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0.01}  # action gripper < 该值 -> close/0
GRIPPER_OPEN_THRESHOLD=${GRIPPER_OPEN_THRESHOLD:-0.99999}  # action gripper > 该值 -> open/1；中间区间保持上次状态
CAMERA_PREVIEW_SCALE=${CAMERA_PREVIEW_SCALE:-0.5}
CAMERA_PREVIEW_FPS=${CAMERA_PREVIEW_FPS:-10}
CAMERA_PREVIEW_WINDOW=${CAMERA_PREVIEW_WINDOW:-soft_vla_live_cameras}

EPISODE_INDEX=${EPISODE_INDEX:-0}        # replay observation 时使用的数据 episode index
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}     # replay video 后端
REPO_ID=${REPO_ID:-local/soft_robot_7_03_1_delta_tcp}  # replay dataset repo id

IO_LABEL=mock_pressure
[[ "$RUN_HARDWARE" == "1" ]] && IO_LABEL=hardware
OBS_LABEL=replay_obs
[[ "$LIVE_OBSERVATION" == "1" ]] && OBS_LABEL=live_obs
# ===== 数据与日志 =====
RUN_LABEL=${RUN_LABEL:-"human_${VLA_BACKEND}_${OBS_LABEL}_${MODE}_${FEEDFORWARD}_${FEEDBACK}_${IO_LABEL}"}  # 本次运行名
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}  # 本次运行所有输出根目录
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/smolvla_human_intervention.jsonl"}  # 全进程调试日志
EPISODE_SAVE_ROOT=${EPISODE_SAVE_ROOT:-"$OUTPUT_DIR/episodes"}  # episode 保存根目录；每段为 episode_0000/

args=(
  soft_vla/scripts/deploy_smolvla_human_intervention.py
  --mode "$MODE"
  --duration-s "$DURATION_S"
  --chunk-size "$CHUNK_SIZE"
  --execution-horizon "$EXECUTION_HORIZON"
  --replan-interval "$REPLAN_INTERVAL"
  --chunk-trigger-margin "$CHUNK_TRIGGER_MARGIN"
  --chunk-expected-stale-steps "$CHUNK_EXPECTED_STALE_STEPS"
  --chunk-worst-stale-steps "$CHUNK_WORST_STALE_STEPS"
  --delta-tcp-scale "$DELTA_TCP_SCALE"
  --pressure-scale "$PRESSURE_SCALE"
  --feedforward "$FEEDFORWARD"
  --feedback "$FEEDBACK"
  --feedback-gain-scale "$FEEDBACK_GAIN_SCALE"
  --max-integral-error "$MAX_INTEGRAL_ERROR"
  --q-tcp6-weight "$Q_TCP6_WEIGHT"
  --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT"
  --q-latent-weight "$Q_LATENT_WEIGHT"
  --q-integral-weight "$Q_INTEGRAL_WEIGHT"
  --r-weight "$R_WEIGHT"
  --motion-policy-ready-timeout-s "$MOTION_POLICY_READY_TIMEOUT_S"
  --device "$DEVICE"
  --ip "$LUMO_IP"
  --rigid-body-id "$RIGID_BODY_ID"
  --receive-timeout-ms "$RECEIVE_TIMEOUT_MS"
  --port "$PORT"
  --packet-channels 16
  --episode-index "$EPISODE_INDEX"
  --video-backend "$VIDEO_BACKEND"
  --repo-id "$REPO_ID"
  --log-jsonl "$LOG_JSONL"
  --task "$TASK"
  --action-print-interval-steps "$ACTION_PRINT_INTERVAL_STEPS"
  --initial-gripper-open "$INITIAL_GRIPPER_OPEN"
  --gripper-close-threshold "$GRIPPER_CLOSE_THRESHOLD"
  --gripper-open-threshold "$GRIPPER_OPEN_THRESHOLD"
  --episode-end-reset-sleep-s "$EPISODE_END_RESET_SLEEP_S"
  --episode-end-reset-zero-packets "$EPISODE_END_RESET_ZERO_PACKETS"
  --first-action-timeout-s "$FIRST_ACTION_TIMEOUT_S"
  --gamepad-backend "$GAMEPAD_BACKEND"
  --gamepad-deadzone "$GAMEPAD_DEADZONE"
  --intervention-release-deadzone "$INTERVENTION_RELEASE_DEADZONE"
  --intervention-release-ticks "$INTERVENTION_RELEASE_TICKS"
  --human-max-delta-pos "$HUMAN_MAX_DELTA_POS"
  --human-max-delta-rot "$HUMAN_MAX_DELTA_ROT"
  --rotation-axis "$ROTATION_AXIS"
  --human-target-max-pos-offset "$HUMAN_TARGET_MAX_POS_OFFSET"
  --human-target-max-rot-offset "$HUMAN_TARGET_MAX_ROT_OFFSET"
  --handover-blend-steps "$HANDOVER_BLEND_STEPS"
  --episode-save-root "$EPISODE_SAVE_ROOT"
  --zed-eye "$ZED_EYE"
  --zed-width "$ZED_WIDTH"
  --zed-height "$ZED_HEIGHT"
  --zed-fps "$ZED_FPS"
  --zed-warmup-usable-frames "$ZED_WARMUP_USABLE_FRAMES"
  --realsense-warmup-usable-frames "$REALSENSE_WARMUP_USABLE_FRAMES"
  --min-realsense-mean "$MIN_REALSENSE_MEAN"
  --camera-preview-scale "$CAMERA_PREVIEW_SCALE"
  --camera-preview-fps "$CAMERA_PREVIEW_FPS"
  --camera-preview-window "$CAMERA_PREVIEW_WINDOW"
)

[[ -n "$GAMEPAD_DEVICE_PATH" ]] && args+=(--gamepad-device-path "$GAMEPAD_DEVICE_PATH")
[[ -n "$ZED_INDEX" ]] && args+=(--zed-index "$ZED_INDEX")
[[ -n "$MAX_INFERENCE_CHUNKS" ]] && args+=(--max-inference-chunks "$MAX_INFERENCE_CHUNKS")
[[ "$WAIT_FOR_START_KEY" == "1" ]] && args+=(--wait-for-start-key)
[[ "$WAIT_FOR_FIRST_ACTION_CHUNK" != "1" ]] && args+=(--no-wait-for-first-action-chunk)
[[ "$LIVE_OBSERVATION" == "1" ]] && args+=(--live-observation)
[[ "$CAMERA_PREVIEW" == "1" ]] && args+=(--camera-preview)
[[ "$VLA_BACKEND" == "smolvla" ]] && args+=(--real-policy)
[[ "$VLA_BACKEND" != "smolvla" && "$RUN_HARDWARE" == "0" ]] && args+=(--mock)
[[ "$RUN_HARDWARE" == "1" ]] && args+=(--hardware-enabled)
[[ "$STATE_HARDWARE" == "1" ]] && args+=(--state-hardware-enabled)
[[ "$ROTATION_ENABLED" == "1" ]] && args+=(--rotation-enabled)
[[ "$HUMAN_TARGET_INTEGRATION" != "1" ]] && args+=(--no-human-target-integration)
[[ "$BLEND_TCP_ONLY" == "1" ]] && args+=(--blend-tcp-only)
[[ "$BLEND_GRIPPER" == "1" ]] && args+=(--blend-gripper)
[[ "$REMOTE_CONTROL_DEBUG" == "1" ]] && args+=(--remote-control-debug)
[[ "$SAVE_HUMAN_EPISODES" != "1" ]] && args+=(--no-save-human-episodes)
[[ -n "$REALSENSE_SERIAL_CAM2" ]] && args+=(--realsense-serial-cam2 "$REALSENSE_SERIAL_CAM2")
[[ -n "$REALSENSE_SERIAL_CAM3" ]] && args+=(--realsense-serial-cam3 "$REALSENSE_SERIAL_CAM3")

echo "[soft_vla] human intervention runtime: HUMAN_INTERVENTION=$HUMAN_INTERVENTION MODE=$MODE GAMEPAD_BACKEND=$GAMEPAD_BACKEND REMOTE_CONTROL_DEBUG=$REMOTE_CONTROL_DEBUG"
echo "[soft_vla] output dir: $OUTPUT_DIR"
echo "[soft_vla] episode root: $EPISODE_SAVE_ROOT"
"$PY" "${args[@]}"
