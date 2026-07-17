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
# 上层每1秒发送一个target、总共发送3次：
#   TARGET_FREQUENCY=1 MAX_FRAMES=3 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 每次切换target前清空积分误差q（默认0，保持积分连续）：
#   RESET_INTEGRAL_ON_TARGET=1 TARGET_FREQUENCY=10 MAX_FRAMES=3 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 上层每10秒发送一个target、总共发送2次：
#   TARGET_FREQUENCY=0.1 MAX_FRAMES=2 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 使用10Hz Full-A Koopman时，需同时覆盖checkpoint与底层频率：
#   KOOPMAN_CHECKPOINT=/path/to/10hz/best.pt FREQUENCY=10 TARGET_FREQUENCY=1 MAX_FRAMES=3 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 实物小段验证：
#   MAX_FRAMES=2 PRESSURE_SCALE=0.2 bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
# 离线生成临时固定 K 后回放：
#   TARGET_FREQUENCY=10 MAX_FRAMES=0 KOOPMAN_CHECKPOINT=/home/cao/skill_learning_soft_robot/motion_control_training/koopman/experiments/fullA_history_v2/runs/robot_records_7_03_1_delta_tcp_fullA_history_v2_10hz_k50_hist10_epoch3000_wandb_online_20260712_2313/best.pt FREQUENCY=10 FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay_fullA_history_v2.sh
#   TARGET_FREQUENCY=10 MAX_FRAMES=0 FREQUENCY=50 FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay_fullA_history_v2.sh



cd "$ROOT"

PRESSURE_CHECKPOINT=${PRESSURE_CHECKPOINT:-"$ROOT/motion_control_training/feedforward_pressure/runs/tcp6_target_raw_pressure/best.pt"}
# 新采集的标准50Hz MAT数据模型：source/target=50Hz、history=30、ksteps=50。
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-"$ROOT/motion_control_training/koopman/experiments/fullA_history_v2/runs/koopman_pressure16_fullA_history_v2_smoke_model_hparams_fullwindows_epoch1000_wandb_online_20260716/best.pt"}

DATASET_ROOT=${DATASET_ROOT:-"$ROOT/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"}
FEEDBACK=${FEEDBACK:-integral_lqr}  # integral_lqr / fixed_k_integral
RUN_LABEL=${RUN_LABEL:-"pressure_tcp6_fullA_history_v2_${FEEDBACK}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}
EPISODE_INDEX=${EPISODE_INDEX:-0}
MAX_FRAMES=${MAX_FRAMES:-0}               # target-state发送次数；<=0表示整个episode
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}.jsonl"}
PLOT_PATH=${PLOT_PATH:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}.png"}
SUMMARY_PATH=${SUMMARY_PATH:-"$OUTPUT_DIR/replay_real_ep${EPISODE_INDEX}_summary.json"}

HARDWARE_ENABLED=${HARDWARE_ENABLED:-1}
DEVICE=${DEVICE:-cuda}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}

# FREQUENCY必须与Full-A Koopman checkpoint的target_hz一致。
FREQUENCY=${FREQUENCY:-50}
TARGET_FREQUENCY=${TARGET_FREQUENCY:-10}  # 上层target-state频率：10 / 1 / 0.1 Hz
RESET_INTEGRAL_ON_TARGET=${RESET_INTEGRAL_ON_TARGET:-0}  # 1: 每个新target前q清零；0: 跨target保留q
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}
PRESSURE_SCALE=${PRESSURE_SCALE:-1}
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-1}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-5}
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50}
FIXED_K_PATH=${FIXED_K_PATH:-}

if [[ "$RESET_INTEGRAL_ON_TARGET" != "0" && "$RESET_INTEGRAL_ON_TARGET" != "1" ]]; then
  echo "RESET_INTEGRAL_ON_TARGET must be 0 or 1, got $RESET_INTEGRAL_ON_TARGET" >&2
  exit 2
fi

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
      --frequency "$FREQUENCY" \
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
  --target-frequency "$TARGET_FREQUENCY"
  --device "$DEVICE"
  --log-jsonl "$LOG_JSONL"
  --plot-path "$PLOT_PATH"
  --output-summary "$SUMMARY_PATH"
)

if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  args+=(--fixed-k-path "$FIXED_K_PATH")
fi

if [[ "$RESET_INTEGRAL_ON_TARGET" == "1" ]]; then
  args+=(--reset-integral-on-target)
fi

if [[ "$HARDWARE_ENABLED" == "1" ]]; then
  args+=(--hardware-enabled)
else
  args+=(--mock)
fi

"$PY" "${args[@]}"
