#!/usr/bin/env bash
set -euo pipefail

# SmolVLA pressure-state 四进程部署入口；与原 smolvla_deploy.sh 相互独立，不覆盖原部署流程。
# 常用方式：
#   1) 纯 mock，不加载 VLA 权重和硬件：DEVICE=cpu DURATION_S=2 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_deploy.sh
#   2) 1w 真实模型 + replay 图像 + mock IO：VLA_BACKEND=smolvla bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_deploy.sh
#   3) 1w 真实模型 + 三相机 + 动捕 + mock 压力：STATE_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_deploy.sh
#   4) 完整实物控制：RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 PRESSURE_DELTA_SCALE=0.2 DURATION_S=5 bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_deploy.sh
#
# pressure-state 数据约定：
#   state25  = [12D 运动状态, 1D 夹爪状态, 当前12D归一化气压]
#   action19 = [6D delta TCP, 1D二值夹爪动作, 12D气压偏差]
#   VLA前馈  = clip(当前12D归一化气压 + PRESSURE_DELTA_SCALE * 12D气压偏差, 0, 1)
#   最终气压 = clip(VLA前馈 + Koopman闭环修正, 0, 1)，随后由底层乘3转换为实物压力。
#
# 频率关系：上层默认10Hz；每次生成的 target state 和 VLA前馈在底层50Hz的5个周期内保持不变；
# Koopman闭环修正仍在每个50Hz周期根据最新状态重新计算。当前使用标准50Hz数据训练的
# Full-A history-v2 模型（history=30）；RESET_INTEGRAL_ON_TARGET=0时积分q跨target保持连续，
# 设为1时每收到一个新target便清零q。
# 该轮训练的 cam_1/ZED 主视角在进入模型前裁掉右侧20%：1280x720 -> 1024x720；
# 推理与 CAMERA_PREVIEW 共用同一裁剪结果，预览显示的就是实际送入推理预处理器的三路图像。
#
# K 的生成与使用：
#   FEEDBACK=fixed_k_integral 且 FIXED_K_PATH 为空时，与 episode 回放一致：先离线求解一次积分LQR的K，
#   写入临时 npz 后由50Hz控制进程加载；整个控制过程固定使用该K，不会在线重复迭代。
#   如果设置 FIXED_K_PATH=/path/to/fixed_k.npz，则复用该文件；路径不存在时会先生成。
# Ctrl+C/SIGTERM 会通知所有子进程退出；50Hz控制进程会连续发送 zero pressure 后再关闭串口。

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== 运行模式 =====
RUN_HARDWARE=${RUN_HARDWARE:-0}          # 1: 打开串口并真实下发压力；0: pressure mock
STATE_HARDWARE=${STATE_HARDWARE:-0}      # 1: 读取真实 LuMo 动捕；RUN_HARDWARE=1 时自动等价开启
VLA_BACKEND=${VLA_BACKEND:-mock}         # mock 或 smolvla
LIVE_OBSERVATION=${LIVE_OBSERVATION:-0}  # 1: 使用实时三相机；0: 使用 replay 图像
CAMERA_PREVIEW=${CAMERA_PREVIEW:-0}      # 1: 显示三相机预览窗口
WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}  # 1: 模型准备完成后按键开始
WAIT_FOR_FIRST_ACTION_CHUNK=${WAIT_FOR_FIRST_ACTION_CHUNK:-1}  # 1: 首段 action chunk 就绪后启动控制

# ===== 固定50Hz底层与可配置上层频率 =====
CONTROL_FREQUENCY=${CONTROL_FREQUENCY:-50}  # 本部署流程固定为50Hz
UPPER_FREQUENCY=${UPPER_FREQUENCY:-10}      # 上层VLA/target-state频率；默认10Hz
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

# ===== Pressure-state SmolVLA 与底层运动控制 =====
CHECKPOINT=${CHECKPOINT:-"$ROOT/soft_vla/outputs/full_runs/smolvla_pressure_state_bs8_20k_pressure_state_bs8_20k_20260715_161801/checkpoints/020000/pretrained_model"}
# replay 模式只使用记录图像；observation.state 会替换为运行时当前运动状态、夹爪和归一化气压。
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"}
REPO_ID=${REPO_ID:-local/soft_robot_7_03_1_delta_tcp}
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-"$ROOT/motion_control_training/koopman/experiments/fullA_history_v2/runs/koopman_pressure16_fullA_history_v2_smoke_model_hparams_fullwindows_epoch1000_wandb_online_20260716/best.pt"}
KOOPMAN_ARCHITECTURE=${KOOPMAN_ARCHITECTURE:-fullA_history_v2}
FEEDBACK=${FEEDBACK:-fixed_k_integral}  # none / integral_lqr / fixed_k_integral
FIXED_K_PATH=${FIXED_K_PATH:-}          # 空: 启动前生成临时离线K；非空: 复用/生成指定npz
RESET_INTEGRAL_ON_TARGET=${RESET_INTEGRAL_ON_TARGET:-0}  # 1: 每个新target清零q；0: 保留q
DEVICE=${DEVICE:-cuda}
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}
PRESSURE_DELTA_SCALE=${PRESSURE_DELTA_SCALE:-1}
PRESSURE_SCALE=${PRESSURE_SCALE:-1}
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-1}
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
CAM1_CROP_RIGHT_FRACTION=${CAM1_CROP_RIGHT_FRACTION:-0.2}  # 与本轮训练一致：裁掉cam_1右侧1/5
REALSENSE_SERIAL_CAM2=${REALSENSE_SERIAL_CAM2:-401522072797}
REALSENSE_SERIAL_CAM3=${REALSENSE_SERIAL_CAM3:-408322072769}
ZED_WARMUP_USABLE_FRAMES=${ZED_WARMUP_USABLE_FRAMES:-10}
REALSENSE_WARMUP_USABLE_FRAMES=${REALSENSE_WARMUP_USABLE_FRAMES:-10}
MIN_REALSENSE_MEAN=${MIN_REALSENSE_MEAN:-40}
INITIAL_GRIPPER_OPEN=${INITIAL_GRIPPER_OPEN:-1}
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0.1}
GRIPPER_OPEN_THRESHOLD=${GRIPPER_OPEN_THRESHOLD:-0.999999}
CAMERA_PREVIEW_SCALE=${CAMERA_PREVIEW_SCALE:-0.5}
CAMERA_PREVIEW_FPS=${CAMERA_PREVIEW_FPS:-10}  # 本模式预览随实际VLA推理帧刷新，不额外抓取原始帧
CAMERA_PREVIEW_WINDOW=${CAMERA_PREVIEW_WINDOW:-soft_vla_pressure_state_live_cameras}

# ===== 数据与日志 =====
EPISODE_INDEX=${EPISODE_INDEX:-0}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
IO_LABEL=mock_pressure
[[ "$RUN_HARDWARE" == "1" ]] && IO_LABEL=hardware
OBS_LABEL=replay_obs
[[ "$LIVE_OBSERVATION" == "1" ]] && OBS_LABEL=live_obs
RUN_LABEL=${RUN_LABEL:-"pressure_state_1w_${VLA_BACKEND}_${OBS_LABEL}_${MODE}_${FEEDBACK}_${IO_LABEL}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/artifacts/real_robot/$RUN_LABEL"}
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/smolvla_pressure_state_deploy.jsonl"}

for required in "$PY" "$CHECKPOINT/config.json" "$KOOPMAN_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "[soft_vla] required asset not found: $required" >&2
    exit 2
  fi
done

if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  BUILD_FIXED_K=0
  if [[ -z "$FIXED_K_PATH" ]]; then
    FIXED_K_TMP_DIR=$(mktemp -d /tmp/soft_vla_pressure_state_fixed_k.XXXXXX)
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
  soft_vla/scripts/real_robot/components/deploy_smolvla_real.py
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
echo "[soft_vla] Koopman: architecture=$KOOPMAN_ARCHITECTURE checkpoint=$KOOPMAN_CHECKPOINT reset_integral_on_target=$RESET_INTEGRAL_ON_TARGET"
echo "[soft_vla] pressure: feedforward=clip(current+delta,0,1), feedback=$FEEDBACK, gain=$FEEDBACK_GAIN_SCALE"
echo "[soft_vla] cam_1 preprocessing: crop_right_fraction=$CAM1_CROP_RIGHT_FRACTION; preview shows exact inference inputs"
"$PY" "${args[@]}"
