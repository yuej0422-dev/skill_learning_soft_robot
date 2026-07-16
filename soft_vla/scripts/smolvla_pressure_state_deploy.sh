#!/usr/bin/env bash
set -euo pipefail

# SmolVLA pressure-state (25D state / 19D action) deployment entry.
# This is intentionally separate from smolvla_deploy.sh.
#
# Data flow at every upper-frequency tick (default 10 Hz):
#   state25 = [state12, gripper, current_normalized_pressure12]
#   action19 = [delta_tcp6, gripper, delta_pressure12]
#   VLA feedforward = clip(current_normalized_pressure12 + delta_pressure12, 0, 1)
# The target state and VLA feedforward are held for all 50 Hz lower-control ticks;
# only the Koopman feedback correction is refreshed at 50 Hz.
#
# Safe mock plumbing smoke (does not load SmolVLA weights or hardware):
#   DEVICE=cpu DURATION_S=2 bash soft_vla/scripts/smolvla_pressure_state_deploy.sh
# Real model + live cameras/state + mock pressure output:
#   VLA_BACKEND=smolvla LIVE_OBSERVATION=1 STATE_HARDWARE=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/smolvla_pressure_state_deploy.sh
# Full hardware (first run with small PRESSURE_DELTA_SCALE and short DURATION_S):
#   RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 PRESSURE_DELTA_SCALE=0.2 DURATION_S=5 bash soft_vla/scripts/smolvla_pressure_state_deploy.sh

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== Runtime mode =====
RUN_HARDWARE=${RUN_HARDWARE:-0}
STATE_HARDWARE=${STATE_HARDWARE:-0}
VLA_BACKEND=${VLA_BACKEND:-mock}       # mock / smolvla
LIVE_OBSERVATION=${LIVE_OBSERVATION:-0}
CAMERA_PREVIEW=${CAMERA_PREVIEW:-0}
WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}
WAIT_FOR_FIRST_ACTION_CHUNK=${WAIT_FOR_FIRST_ACTION_CHUNK:-1}

# ===== Fixed 50 Hz lower loop and configurable upper loop =====
CONTROL_FREQUENCY=${CONTROL_FREQUENCY:-50}
UPPER_FREQUENCY=${UPPER_FREQUENCY:-10}
if [[ "$CONTROL_FREQUENCY" != "50" && "$CONTROL_FREQUENCY" != "50.0" ]]; then
  echo "[soft_vla] pressure-state deployment requires CONTROL_FREQUENCY=50, got $CONTROL_FREQUENCY" >&2
  exit 2
fi

TASK=${TASK:-"pick up the apple and place it on the electronic scale"}
MODE=${MODE:-receding_horizon}
DURATION_S=${DURATION_S:-100}
CHUNK_SIZE=${CHUNK_SIZE:-50}
EXECUTION_HORIZON=${EXECUTION_HORIZON:-10}
REPLAN_INTERVAL=${REPLAN_INTERVAL:-5}
CHUNK_TRIGGER_MARGIN=${CHUNK_TRIGGER_MARGIN:-1}
CHUNK_EXPECTED_STALE_STEPS=${CHUNK_EXPECTED_STALE_STEPS:-2}
CHUNK_WORST_STALE_STEPS=${CHUNK_WORST_STALE_STEPS:-5}
MAX_INFERENCE_CHUNKS=${MAX_INFERENCE_CHUNKS:-}
ACTION_PRINT_INTERVAL_STEPS=${ACTION_PRINT_INTERVAL_STEPS:-10}
FIRST_ACTION_TIMEOUT_S=${FIRST_ACTION_TIMEOUT_S:-120}

# ===== Pressure-state SmolVLA and lower motion control =====
CHECKPOINT=${CHECKPOINT:-"$ROOT/soft_vla/outputs/full_runs/smolvla_pressure_state_bs8_20k_pressure_state_bs8_20k_20260715_161801/checkpoints/010000/pretrained_model"}
# Replay mode only uses recorded images; its observation.state is replaced by
# the runtime's current state12 + gripper + normalized pressure12.
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"}
REPO_ID=${REPO_ID:-local/soft_robot_7_03_1_delta_tcp}
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-"$ROOT/motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"}
FEEDBACK=${FEEDBACK:-fixed_k_integral}
FIXED_K_PATH=${FIXED_K_PATH:-}
DEVICE=${DEVICE:-cuda}
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}
PRESSURE_DELTA_SCALE=${PRESSURE_DELTA_SCALE:-1}
PRESSURE_SCALE=${PRESSURE_SCALE:-1}
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.1}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-0.5}
MOTION_POLICY_READY_TIMEOUT_S=${MOTION_POLICY_READY_TIMEOUT_S:-120}
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50.0}

# ===== Robot and cameras =====
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}
RECEIVE_TIMEOUT_MS=${RECEIVE_TIMEOUT_MS:-1000}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
ZED_INDEX=${ZED_INDEX:-}
ZED_EYE=${ZED_EYE:-left}
ZED_WIDTH=${ZED_WIDTH:-2560}
ZED_HEIGHT=${ZED_HEIGHT:-720}
ZED_FPS=${ZED_FPS:-30}
REALSENSE_SERIAL_CAM2=${REALSENSE_SERIAL_CAM2:-401522072797}
REALSENSE_SERIAL_CAM3=${REALSENSE_SERIAL_CAM3:-408322072769}
ZED_WARMUP_USABLE_FRAMES=${ZED_WARMUP_USABLE_FRAMES:-10}
REALSENSE_WARMUP_USABLE_FRAMES=${REALSENSE_WARMUP_USABLE_FRAMES:-10}
MIN_REALSENSE_MEAN=${MIN_REALSENSE_MEAN:-40}
INITIAL_GRIPPER_OPEN=${INITIAL_GRIPPER_OPEN:-1}
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0.1}
GRIPPER_OPEN_THRESHOLD=${GRIPPER_OPEN_THRESHOLD:-0.995}
CAMERA_PREVIEW_SCALE=${CAMERA_PREVIEW_SCALE:-0.5}
CAMERA_PREVIEW_FPS=${CAMERA_PREVIEW_FPS:-10}
CAMERA_PREVIEW_WINDOW=${CAMERA_PREVIEW_WINDOW:-soft_vla_pressure_state_live_cameras}

# ===== Logging =====
EPISODE_INDEX=${EPISODE_INDEX:-0}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
IO_LABEL=mock_pressure
[[ "$RUN_HARDWARE" == "1" ]] && IO_LABEL=hardware
OBS_LABEL=replay_obs
[[ "$LIVE_OBSERVATION" == "1" ]] && OBS_LABEL=live_obs
RUN_LABEL=${RUN_LABEL:-"pressure_state_1w_${VLA_BACKEND}_${OBS_LABEL}_${MODE}_${FEEDBACK}_${IO_LABEL}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/smolvla_pressure_state_deploy.jsonl"}

for required in "$PY" "$CHECKPOINT/config.json" "$KOOPMAN_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "[soft_vla] required asset not found: $required" >&2
    exit 2
  fi
done

args=(
  soft_vla/scripts/deploy_smolvla_real.py
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
  --vla-action-mode pressure_delta19
  --reference-interpolation zero_order_hold
  --delta-tcp-scale "$DELTA_TCP_SCALE"
  --pressure-delta-scale "$PRESSURE_DELTA_SCALE"
  --pressure-scale "$PRESSURE_SCALE"
  --feedforward external
  --feedback "$FEEDBACK"
  --koopman-checkpoint "$KOOPMAN_CHECKPOINT"
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

[[ -n "$FIXED_K_PATH" ]] && args+=(--fixed-k-path "$FIXED_K_PATH")
[[ -n "$ZED_INDEX" ]] && args+=(--zed-index "$ZED_INDEX")
[[ -n "$MAX_INFERENCE_CHUNKS" ]] && args+=(--max-inference-chunks "$MAX_INFERENCE_CHUNKS")
[[ "$LIVE_OBSERVATION" == "1" ]] && args+=(--live-observation)
[[ "$CAMERA_PREVIEW" == "1" ]] && args+=(--camera-preview)
[[ "$WAIT_FOR_START_KEY" == "1" ]] && args+=(--wait-for-start-key)
[[ "$WAIT_FOR_FIRST_ACTION_CHUNK" != "1" ]] && args+=(--no-wait-for-first-action-chunk)
[[ -n "$REALSENSE_SERIAL_CAM2" ]] && args+=(--realsense-serial-cam2 "$REALSENSE_SERIAL_CAM2")
[[ -n "$REALSENSE_SERIAL_CAM3" ]] && args+=(--realsense-serial-cam3 "$REALSENSE_SERIAL_CAM3")
[[ "$VLA_BACKEND" == "smolvla" ]] && args+=(--real-policy)
[[ "$VLA_BACKEND" != "smolvla" && "$RUN_HARDWARE" == "0" ]] && args+=(--mock)
[[ "$RUN_HARDWARE" == "1" ]] && args+=(--hardware-enabled)
[[ "$STATE_HARDWARE" == "1" ]] && args+=(--state-hardware-enabled)

echo "[soft_vla] pressure-state deployment: checkpoint=$CHECKPOINT"
echo "[soft_vla] frequencies: VLA/target=${UPPER_FREQUENCY}Hz, Koopman/control=${CONTROL_FREQUENCY}Hz (zero-order hold)"
echo "[soft_vla] pressure: feedforward=clip(current+delta,0,1), feedback=$FEEDBACK, gain=$FEEDBACK_GAIN_SCALE"
"$PY" "${args[@]}"
