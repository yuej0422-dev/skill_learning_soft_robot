#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"

# Episode 小幅/全量回放入口。
# 注意：本脚本固定使用 --hardware-enabled，会读取真实 LuMo 并向串口下发真实压力。
# 常用方式：
#   1) 默认回放：bash soft_vla/scripts/epis_replay.sh
#   2) 换前馈：FEEDFORWARD=awac bash soft_vla/scripts/epis_replay.sh
#   3) 离线临时生成固定 K：FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay.sh
#   4) 调 Q/R 后生成固定 K：Q_TCP6_WEIGHT=2 R_WEIGHT=50 FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay.sh
#   5) 改输出目录名：RUN_LABEL=Qlearning FEEDFORWARD=awac bash soft_vla/scripts/epis_replay.sh
#   6) 上层1Hz、发送3次target：TARGET_FREQUENCY=1 MAX_FRAMES=3 bash soft_vla/scripts/epis_replay.sh
#   7) 上层0.1Hz、发送2次target：TARGET_FREQUENCY=0.1 MAX_FRAMES=20 FEEDBACK=fixed_k_integral bash soft_vla/scripts/epis_replay.sh
# 输出图与 Full-A 入口一致：第三个子图为闭环控制律的12路 delta action。

cd "$ROOT"

# ===== 控制器选择 =====
FEEDFORWARD=${FEEDFORWARD:-pressure_model}  # pressure_model / awac
FEEDBACK=${FEEDBACK:-integral_lqr}          # integral_lqr / fixed_k_integral

# ===== 数据与输出 =====
RUN_LABEL=${RUN_LABEL:-"${FEEDFORWARD}_${FEEDBACK}"}  # 输出子目录名
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}
EPISODE_INDEX=${EPISODE_INDEX:-0}           # LeRobot episode index
MAX_FRAMES=${MAX_FRAMES:-0}                 # target-state发送次数；<=0 表示完整 episode
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/soft_vla_replay_real_ep${EPISODE_INDEX}.jsonl"}
PLOT_PATH=${PLOT_PATH:-"$OUTPUT_DIR/soft_vla_replay_real_ep${EPISODE_INDEX}.png"}
SUMMARY_PATH=${SUMMARY_PATH:-"$OUTPUT_DIR/soft_vla_replay_real_ep${EPISODE_INDEX}_summary.json"}

# ===== 运行设备 =====
DEVICE=${DEVICE:-cuda}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}

# ===== 控制频率 =====
FREQUENCY=${FREQUENCY:-50}                  # 底层闭环频率
TARGET_FREQUENCY=${TARGET_FREQUENCY:-10}    # 上层target-state频率：10 / 1 / 0.1 Hz

# ===== 策略输出与压力缩放 =====
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}       # target delta TCP 缩放
PRESSURE_SCALE=${PRESSURE_SCALE:-1}         # 最终压力缩放，1 表示完整 0-3 bar 范围
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.2}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-0.5}

# ===== fixed_k_integral 离线 K 参数 =====
# R_WEIGHT 越大通常越保守；Q_TCP6_WEIGHT/Q_INTEGRAL_WEIGHT 越大越强调跟踪/积分误差。
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50.0}

# fixed_k_integral 的 K 每次临时生成并自动删除；如果需要保留 K 文件，请单独运行 build_fixed_k_integral.py。

if [[ "$FEEDBACK" == "fixed_k_integral" ]]; then
  FIXED_K_TMP_DIR=$(mktemp -d /tmp/soft_vla_fixed_k.XXXXXX)
  trap 'rm -rf "$FIXED_K_TMP_DIR"' EXIT
  FIXED_K_PATH="$FIXED_K_TMP_DIR/fixed_k_integral.npz"
  "$PY" soft_vla/scripts/build_fixed_k_integral.py \
    --output "$FIXED_K_PATH" \
    --device "$DEVICE" \
    --q-tcp6-weight "$Q_TCP6_WEIGHT" \
    --q-state-tail-weight "$Q_STATE_TAIL_WEIGHT" \
    --q-latent-weight "$Q_LATENT_WEIGHT" \
    --q-integral-weight "$Q_INTEGRAL_WEIGHT" \
    --r-weight "$R_WEIGHT"
fi

args=(
  soft_vla/scripts/replay_episode_real.py
  --hardware-enabled
  --ip "$LUMO_IP"
  --rigid-body-id "$RIGID_BODY_ID"
  --port "$PORT"
  --packet-channels 16
  --episode-index "$EPISODE_INDEX"
  --max-frames "$MAX_FRAMES"
  --feedforward "$FEEDFORWARD"
  --feedback "$FEEDBACK"
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

"$PY" "${args[@]}"
