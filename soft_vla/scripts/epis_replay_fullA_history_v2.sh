#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"

# 独立的 Full-A history v2 底层运动控制回放入口。
# 不修改 epis_replay.sh 的默认模型和参数。
#
# 默认会连接真实 LuMo 与压力串口；先做离线接线验证时可运行：
#   HARDWARE_ENABLED=0 MAX_FRAMES=2 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 实物小段验证：
#   MAX_FRAMES=2 PRESSURE_SCALE=0.2 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 离线生成临时固定 K 后回放：
#   FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay_fullA_history_v2.sh

cd "$ROOT"

PRESSURE_CHECKPOINT=${PRESSURE_CHECKPOINT:-"$ROOT/motion_control_training/feedforward_pressure/runs/tcp6_target_raw_pressure/best.pt"}
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-"$ROOT/motion_control_training/koopman/experiments/fullA_history_v2/runs/robot_records_7_03_1_delta_tcp_fullA_history_v2_50hz_k50_hist10_epoch3000_wandb_online_20260712_2320/best.pt"}

DATASET_ROOT=${DATASET_ROOT:-"$ROOT/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"}
FEEDBACK=${FEEDBACK:-integral_lqr}  # integral_lqr / fixed_k_integral
RUN_LABEL=${RUN_LABEL:-"pressure_tcp6_fullA_history_v2_${FEEDBACK}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}
EPISODE_INDEX=${EPISODE_INDEX:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}.jsonl"}
PLOT_PATH=${PLOT_PATH:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}.png"}
SUMMARY_PATH=${SUMMARY_PATH:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}_summary.json"}

HARDWARE_ENABLED=${HARDWARE_ENABLED:-1}
DEVICE=${DEVICE:-cuda}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}

# 模型训练频率为 50 Hz；适配器会拒绝频率不一致的部署。
FREQUENCY=${FREQUENCY:-50}
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}
PRESSURE_SCALE=${PRESSURE_SCALE:-1}
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.1}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-0.5}
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50.0}
FIXED_K_PATH=${FIXED_K_PATH:-}

for required in "$PY" "$PRESSURE_CHECKPOINT" "$KOOPMAN_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Required deployment asset not found: $required" >&2
    exit 2
  fi
done

if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  BUILD_FIXED_K=0
  if [[ -z "$FIXED_K_PATH" ]]; then
    FIXED_K_TMP_DIR=$(mktemp -d /tmp/soft_vla_fullA_fixed_k.XXXXXX)
    trap 'rm -rf "$FIXED_K_TMP_DIR"' EXIT
    FIXED_K_PATH="$FIXED_K_TMP_DIR/fixed_k_integral.npz"
    BUILD_FIXED_K=1
  elif [[ ! -f "$FIXED_K_PATH" ]]; then
    BUILD_FIXED_K=1
  fi
  if [[ "$BUILD_FIXED_K" == "1" ]]; then
    "$PY" soft_vla/scripts/build_fixed_k_integral.py \
      --koopman-architecture fullA_history_v2 \
      --koopman-checkpoint "$KOOPMAN_CHECKPOINT" \
      --output "$FIXED_K_PATH" \
      --device "$DEVICE" \
      --dt 0.02 \
      --q-tcp6-weight "$Q_TCP6_WEIGHT" \
      --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT" \
      --q-latent-weight "$Q_LATENT_WEIGHT" \
      --q-integral-weight "$Q_INTEGRAL_WEIGHT" \
      --r-weight "$R_WEIGHT"
  fi
elif [[ "$FEEDBACK" != "integral_lqr" ]]; then
  echo "Unsupported FEEDBACK=$FEEDBACK; use integral_lqr or fixed_k_integral" >&2
  exit 2
fi

args=(
  soft_vla/scripts/replay_episode_real.py
  --dataset-root "$DATASET_ROOT"
  --episode-index "$EPISODE_INDEX"
  --max-frames "$MAX_FRAMES"
  --ip "$LUMO_IP"
  --rigid-body-id "$RIGID_BODY_ID"
  --port "$PORT"
  --packet-channels 16
  --feedforward pressure_model
  --pressure-checkpoint "$PRESSURE_CHECKPOINT"
  --feedback "$FEEDBACK"
  --koopman-architecture fullA_history_v2
  --koopman-checkpoint "$KOOPMAN_CHECKPOINT"
  --delta-tcp-scale "$DELTA_TCP_SCALE"
  --pressure-scale "$PRESSURE_SCALE"
  --feedback-gain-scale "$FEEDBACK_GAIN_SCALE"
  --max-integral-error "$MAX_INTEGRAL_ERROR"
  --q-tcp6-weight "$Q_TCP6_WEIGHT"
  --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT"
  --q-latent-weight "$Q_LATENT_WEIGHT"
  --q-integral-weight "$Q_INTEGRAL_WEIGHT"
  --r-weight "$R_WEIGHT"
  --frequency "$FREQUENCY"
  --device "$DEVICE"
  --log-jsonl "$LOG_JSONL"
  --plot-path "$PLOT_PATH"
  --output-summary "$SUMMARY_PATH"
)

if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  args+=(--fixed-k-path "$FIXED_K_PATH")
fi

if [[ "$HARDWARE_ENABLED" == "1" ]]; then
  args+=(--hardware-enabled)
else
  args+=(--mock)
fi

"$PY" "${args[@]}"
