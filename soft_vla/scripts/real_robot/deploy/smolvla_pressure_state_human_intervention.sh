#!/usr/bin/env bash
set -euo pipefail

# Pressure-state SmolVLA + Xbox 人工介入独立入口。
#
# 控制语义：
#   state25  = [12D 运动状态, 1D 夹爪状态, 当前12D归一化压力指令]
#   action19 = [6D delta TCP, 1D夹爪动作, 12D压力指令偏差]
#   正常执行：VLA delta TCP -> target state；VLA pressure delta -> 前馈压力。
#   人工介入：手柄替换 VLA delta TCP/夹爪来定义 target state；实际执行 pressure delta=0，
#             即保持当前压力作为前馈；VLA pressure delta 仅作 shadow 记录；
#             Koopman 50Hz 闭环继续追踪人工 target。
#   底层使用标准50Hz数据训练的 Full-A history-v2 Koopman（history=30）和 fixed_k_integral；
#   VLA/人工 target 默认10Hz零阶保持；RESET_INTEGRAL_ON_TARGET=0时积分q跨target保持连续，
#   设为1时每收到一个新target便清零q。
#
# 保存格式：
#   EPISODE_SAVE_ROOT/episode_NNNN/data.csv 与原 human-intervention 格式完全一致；
#   三相机图片目录名称和编号完全一致；
#   data.csv 同步保存实际状态/气压、实际执行19D action及阈值前原始 VLA 19D action；
#   LOG_JSONL 保留每个50Hz周期的 VLA前馈、闭环修正、最终压力和跟踪误差。
#
# 推荐测试顺序：
#   1) mock：DEVICE=cpu DURATION_S=2 WAIT_FOR_START_KEY=0 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_human_intervention.sh
#   2) 实时观测但不下发压力：STATE_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_human_intervention.sh
#   3) 实物：RESET_INTEGRAL_ON_TARGET=1 RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_human_intervention.sh

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== 运行模式 =====
RUN_HARDWARE=${RUN_HARDWARE:-0}
STATE_HARDWARE=${STATE_HARDWARE:-0}
VLA_BACKEND=${VLA_BACKEND:-mock}
LIVE_OBSERVATION=${LIVE_OBSERVATION:-0}
CAMERA_PREVIEW=${CAMERA_PREVIEW:-0}
WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}
WAIT_FOR_FIRST_ACTION_CHUNK=${WAIT_FOR_FIRST_ACTION_CHUNK:-1}

# ===== 上层 VLA 与底层控制 =====
CONTROL_FREQUENCY=${CONTROL_FREQUENCY:-50}
UPPER_FREQUENCY=${UPPER_FREQUENCY:-10}
if [[ "$CONTROL_FREQUENCY" != "50" && "$CONTROL_FREQUENCY" != "50.0" ]]; then
  echo "[soft_vla] pressure-state intervention requires CONTROL_FREQUENCY=50, got $CONTROL_FREQUENCY" >&2
  exit 2
fi
TASK=${TASK:-"pick up the apple and place it on the electronic scale"}
MODE=${MODE:-receding_horizon}
DURATION_S=${DURATION_S:-0}
CHUNK_SIZE=${CHUNK_SIZE:-50}
EXECUTION_HORIZON=${EXECUTION_HORIZON:-10}
REPLAN_INTERVAL=${REPLAN_INTERVAL:-5}
CHUNK_TRIGGER_MARGIN=${CHUNK_TRIGGER_MARGIN:-1}
CHUNK_EXPECTED_STALE_STEPS=${CHUNK_EXPECTED_STALE_STEPS:-2}
CHUNK_WORST_STALE_STEPS=${CHUNK_WORST_STALE_STEPS:-5}
MAX_INFERENCE_CHUNKS=${MAX_INFERENCE_CHUNKS:-}
ACTION_PRINT_INTERVAL_STEPS=${ACTION_PRINT_INTERVAL_STEPS:-10}
FIRST_ACTION_TIMEOUT_S=${FIRST_ACTION_TIMEOUT_S:-120}

# ===== 模型与运动控制 =====
CHECKPOINT=${CHECKPOINT:-"$ROOT/soft_vla/outputs/full_runs/smolvla_pressure_state_bs8_20k_pressure_state_bs8_20k_20260715_161801/checkpoints/020000/pretrained_model"}
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"}
REPO_ID=${REPO_ID:-local/soft_robot_7_03_1_delta_tcp}
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-"$ROOT/motion_control_training/koopman/experiments/fullA_history_v2/runs/koopman_pressure16_fullA_history_v2_smoke_model_hparams_fullwindows_epoch1000_wandb_online_20260716/best.pt"}
KOOPMAN_ARCHITECTURE=${KOOPMAN_ARCHITECTURE:-fullA_history_v2}
FEEDBACK=${FEEDBACK:-fixed_k_integral}
FIXED_K_PATH=${FIXED_K_PATH:-}
RESET_INTEGRAL_ON_TARGET=${RESET_INTEGRAL_ON_TARGET:-0}  # 1: 每个新target清零q；0: 保留q
DEVICE=${DEVICE:-cuda}
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}
PRESSURE_DELTA_SCALE=${PRESSURE_DELTA_SCALE:-1}
PRESSURE_SCALE=${PRESSURE_SCALE:-1}
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.25}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-5}
MOTION_POLICY_READY_TIMEOUT_S=${MOTION_POLICY_READY_TIMEOUT_S:-120}
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50.0}

if [[ "$RESET_INTEGRAL_ON_TARGET" != "0" && "$RESET_INTEGRAL_ON_TARGET" != "1" ]]; then
  echo "[soft_vla] RESET_INTEGRAL_ON_TARGET must be 0 or 1, got $RESET_INTEGRAL_ON_TARGET" >&2
  exit 2
fi

# ===== Xbox 人工介入（映射与原 human-intervention 脚本一致） =====
GAMEPAD_BACKEND=${GAMEPAD_BACKEND:-evdev}
GAMEPAD_DEVICE_PATH=${GAMEPAD_DEVICE_PATH:-}
GAMEPAD_DEBUG=${GAMEPAD_DEBUG:-0}
GAMEPAD_DEADZONE=${GAMEPAD_DEADZONE:-0.15}
INTERVENTION_RELEASE_DEADZONE=${INTERVENTION_RELEASE_DEADZONE:-0.10}
INTERVENTION_RELEASE_TICKS=${INTERVENTION_RELEASE_TICKS:-1}
HUMAN_MAX_DELTA_POS=${HUMAN_MAX_DELTA_POS:-0.005}
HUMAN_MAX_DELTA_ROT=${HUMAN_MAX_DELTA_ROT:-0.025}
ROTATION_ENABLED=${ROTATION_ENABLED:-1}
ROTATION_AXIS=${ROTATION_AXIS:-pitch_yaw}
HUMAN_TARGET_INTEGRATION=${HUMAN_TARGET_INTEGRATION:-1}
HUMAN_TARGET_MAX_POS_OFFSET=${HUMAN_TARGET_MAX_POS_OFFSET:-0.01}
HUMAN_TARGET_MAX_ROT_OFFSET=${HUMAN_TARGET_MAX_ROT_OFFSET:-0.05}
HANDOVER_BLEND_STEPS=${HANDOVER_BLEND_STEPS:-2}
BLEND_TCP_ONLY=${BLEND_TCP_ONLY:-1}
BLEND_GRIPPER=${BLEND_GRIPPER:-0}
REMOTE_CONTROL_DEBUG=${REMOTE_CONTROL_DEBUG:-0}
SAVE_HUMAN_EPISODES=${SAVE_HUMAN_EPISODES:-1}
EPISODE_END_RESET_SLEEP_S=${EPISODE_END_RESET_SLEEP_S:-7}
EPISODE_END_RESET_ZERO_PACKETS=${EPISODE_END_RESET_ZERO_PACKETS:-3}

# ===== 动捕、串口与相机 =====
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}
RECEIVE_TIMEOUT_MS=${RECEIVE_TIMEOUT_MS:-1000}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
ZED_INDEX=${ZED_INDEX:-}
ZED_EYE=${ZED_EYE:-left}
ZED_WIDTH=${ZED_WIDTH:-2560}
ZED_HEIGHT=${ZED_HEIGHT:-720}
ZED_FPS=${ZED_FPS:-30}
CAM1_CROP_RIGHT_FRACTION=${CAM1_CROP_RIGHT_FRACTION:-0.2}
REALSENSE_SERIAL_CAM2=${REALSENSE_SERIAL_CAM2:-401522072797}
REALSENSE_SERIAL_CAM3=${REALSENSE_SERIAL_CAM3:-408322072769}
ZED_WARMUP_USABLE_FRAMES=${ZED_WARMUP_USABLE_FRAMES:-10}
REALSENSE_WARMUP_USABLE_FRAMES=${REALSENSE_WARMUP_USABLE_FRAMES:-10}
MIN_REALSENSE_MEAN=${MIN_REALSENSE_MEAN:-40}
INITIAL_GRIPPER_OPEN=${INITIAL_GRIPPER_OPEN:-1}
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0}
GRIPPER_OPEN_THRESHOLD=${GRIPPER_OPEN_THRESHOLD:-1}
CAMERA_PREVIEW_SCALE=${CAMERA_PREVIEW_SCALE:-0.5}
CAMERA_PREVIEW_FPS=${CAMERA_PREVIEW_FPS:-10}
CAMERA_PREVIEW_WINDOW=${CAMERA_PREVIEW_WINDOW:-soft_vla_pressure_state_human_live_cameras}

# ===== 数据与日志 =====
EPISODE_INDEX=${EPISODE_INDEX:-0}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
IO_LABEL=mock_pressure
[[ "$RUN_HARDWARE" == "1" ]] && IO_LABEL=hardware
OBS_LABEL=replay_obs
[[ "$LIVE_OBSERVATION" == "1" ]] && OBS_LABEL=live_obs
RUN_LABEL=${RUN_LABEL:-"pressure_state_human_${VLA_BACKEND}_${OBS_LABEL}_${MODE}_${FEEDBACK}_${IO_LABEL}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/artifacts/real_robot/$RUN_LABEL"}
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/smolvla_pressure_state_human_intervention.jsonl"}
EPISODE_SAVE_ROOT=${EPISODE_SAVE_ROOT:-"$OUTPUT_DIR/episodes"}

for required in "$PY" "$CHECKPOINT/config.json" "$KOOPMAN_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "[soft_vla] required asset not found: $required" >&2
    exit 2
  fi
done

# 与 epis_replay_fullA_history_v2.sh 一致：fixed_k_integral 在启动部署前离线生成/加载K。
if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  BUILD_FIXED_K=0
  if [[ -z "$FIXED_K_PATH" ]]; then
    FIXED_K_TMP_DIR=$(mktemp -d /tmp/soft_vla_pressure_state_human_fixed_k.XXXXXX)
    trap 'rm -rf "$FIXED_K_TMP_DIR"' EXIT
    FIXED_K_PATH="$FIXED_K_TMP_DIR/fixed_k_integral.npz"
    BUILD_FIXED_K=1
  elif [[ ! -f "$FIXED_K_PATH" ]]; then
    BUILD_FIXED_K=1
  fi
  if [[ "$BUILD_FIXED_K" == "1" ]]; then
    "$PY" soft_vla/scripts/real_robot/components/build_fixed_k_integral.py \
      --koopman-architecture "$KOOPMAN_ARCHITECTURE" \
      --koopman-checkpoint "$KOOPMAN_CHECKPOINT" \
      --output "$FIXED_K_PATH" \
      --device "$DEVICE" \
      --frequency "$CONTROL_FREQUENCY" \
      --q-tcp6-weight "$Q_TCP6_WEIGHT" \
      --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT" \
      --q-latent-weight "$Q_LATENT_WEIGHT" \
      --q-integral-weight "$Q_INTEGRAL_WEIGHT" \
      --r-weight "$R_WEIGHT"
  fi
fi

args=(
  soft_vla/scripts/real_robot/components/deploy_smolvla_pressure_state_human_intervention.py
  --mode "$MODE"
  --duration-s "$DURATION_S"
  --upper-frequency "$UPPER_FREQUENCY"
  --control-frequency "$CONTROL_FREQUENCY"
  --chunk-size "$CHUNK_SIZE"
  --execution-horizon "$EXECUTION_HORIZON"
  --replan-interval "$REPLAN_INTERVAL"
  --chunk-trigger-margin "$CHUNK_TRIGGER_MARGIN"
  --chunk-expected-stale-steps "$CHUNK_EXPECTED_STALE_STEPS"
  --chunk-worst-stale-steps "$CHUNK_WORST_STALE_STEPS"
  --delta-tcp-scale "$DELTA_TCP_SCALE"
  --pressure-delta-scale "$PRESSURE_DELTA_SCALE"
  --pressure-scale "$PRESSURE_SCALE"
  --feedback "$FEEDBACK"
  --koopman-checkpoint "$KOOPMAN_CHECKPOINT"
  --koopman-architecture "$KOOPMAN_ARCHITECTURE"
  --feedback-gain-scale "$FEEDBACK_GAIN_SCALE"
  --max-integral-error "$MAX_INTEGRAL_ERROR"
  --q-tcp6-weight "$Q_TCP6_WEIGHT"
  --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT"
  --q-latent-weight "$Q_LATENT_WEIGHT"
  --q-integral-weight "$Q_INTEGRAL_WEIGHT"
  --r-weight "$R_WEIGHT"
  --motion-policy-ready-timeout-s "$MOTION_POLICY_READY_TIMEOUT_S"
  --action-print-interval-steps "$ACTION_PRINT_INTERVAL_STEPS"
  --initial-gripper-open "$INITIAL_GRIPPER_OPEN"
  --gripper-close-threshold "$GRIPPER_CLOSE_THRESHOLD"
  --gripper-open-threshold "$GRIPPER_OPEN_THRESHOLD"
  --episode-end-reset-sleep-s "$EPISODE_END_RESET_SLEEP_S"
  --episode-end-reset-zero-packets "$EPISODE_END_RESET_ZERO_PACKETS"
  --first-action-timeout-s "$FIRST_ACTION_TIMEOUT_S"
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATASET_ROOT"
  --repo-id "$REPO_ID"
  --device "$DEVICE"
  --ip "$LUMO_IP"
  --rigid-body-id "$RIGID_BODY_ID"
  --receive-timeout-ms "$RECEIVE_TIMEOUT_MS"
  --port "$PORT"
  --packet-channels 16
  --episode-index "$EPISODE_INDEX"
  --video-backend "$VIDEO_BACKEND"
  --log-jsonl "$LOG_JSONL"
  --task "$TASK"
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
  --cam1-crop-right-fraction "$CAM1_CROP_RIGHT_FRACTION"
  --zed-warmup-usable-frames "$ZED_WARMUP_USABLE_FRAMES"
  --realsense-warmup-usable-frames "$REALSENSE_WARMUP_USABLE_FRAMES"
  --min-realsense-mean "$MIN_REALSENSE_MEAN"
  --camera-preview-scale "$CAMERA_PREVIEW_SCALE"
  --camera-preview-fps "$CAMERA_PREVIEW_FPS"
  --camera-preview-window "$CAMERA_PREVIEW_WINDOW"
)

[[ -n "$FIXED_K_PATH" ]] && args+=(--fixed-k-path "$FIXED_K_PATH")
[[ "$RESET_INTEGRAL_ON_TARGET" == "1" ]] && args+=(--reset-integral-on-target)
[[ -n "$GAMEPAD_DEVICE_PATH" ]] && args+=(--gamepad-device-path "$GAMEPAD_DEVICE_PATH")
[[ -n "$ZED_INDEX" ]] && args+=(--zed-index "$ZED_INDEX")
[[ -n "$MAX_INFERENCE_CHUNKS" ]] && args+=(--max-inference-chunks "$MAX_INFERENCE_CHUNKS")
[[ -n "$REALSENSE_SERIAL_CAM2" ]] && args+=(--realsense-serial-cam2 "$REALSENSE_SERIAL_CAM2")
[[ -n "$REALSENSE_SERIAL_CAM3" ]] && args+=(--realsense-serial-cam3 "$REALSENSE_SERIAL_CAM3")
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
[[ "$GAMEPAD_DEBUG" == "1" ]] && args+=(--print-gamepad-events)

echo "[soft_vla] pressure-state human intervention: VLA_BACKEND=$VLA_BACKEND MODE=$MODE feedback=$FEEDBACK"
echo "[soft_vla] Koopman: architecture=$KOOPMAN_ARCHITECTURE checkpoint=$KOOPMAN_CHECKPOINT reset_integral_on_target=$RESET_INTEGRAL_ON_TARGET"
echo "[soft_vla] arbitration: human replaces delta TCP target and executes pressure_delta=0; VLA pressure stays shadow-only"
echo "[soft_vla] episode data: $EPISODE_SAVE_ROOT/episode_NNNN/data.csv"
"$PY" "${args[@]}"
