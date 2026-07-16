#!/usr/bin/env bash
set -euo pipefail

# SmolVLA 四进程部署入口。
# 常用方式：
#   1) 纯 mock：bash soft_vla/scripts/smolvla_deploy.sh
#   2) 真实模型 + replay 观测 + mock IO：VLA_BACKEND=smolvla bash soft_vla/scripts/smolvla_deploy.sh
#   3) 真实模型 + 三相机 + 动捕 + mock 压力：STATE_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/smolvla_deploy.sh
#   4) 完整实物控制：RUN_HARDWARE=1 VLA_BACKEND=smolvla LIVE_OBSERVATION=1 CAMERA_PREVIEW=1 bash soft_vla/scripts/smolvla_deploy.sh
# Ctrl+C/SIGTERM 会通知所有子进程退出；50Hz 控制进程会连续发送 zero pressure 后再关闭串口。

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
PY=${PY:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}
export LD_LIBRARY_PATH="/home/cao/miniconda3/envs/soft_vla_cuda/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

# ===== 运行模式 =====
RUN_HARDWARE=${RUN_HARDWARE:-0}          # 1: 打开串口并真实下发压力；0: pressure mock
STATE_HARDWARE=${STATE_HARDWARE:-0}      # 1: 读取真实 LuMo 动捕；RUN_HARDWARE=1 时自动等价开启
VLA_BACKEND=${VLA_BACKEND:-mock}         # mock 或 smolvla
LIVE_OBSERVATION=${LIVE_OBSERVATION:-0}  # 1: 用实时 ZED left + 两路 RealSense；0: 用 replay 观测
CAMERA_PREVIEW=${CAMERA_PREVIEW:-0}      # 1: 另开进程显示三相机 resize mosaic
WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}  # 1: motion policy 和 SmolVLA 权重就绪后，按任意键再开始执行
WAIT_FOR_FIRST_ACTION_CHUNK=${WAIT_FOR_FIRST_ACTION_CHUNK:-1}  # 1: 第一段 action chunk 就绪后再启动 10Hz/50Hz 控制

# ===== 任务与执行器 =====
TASK=${TASK:-"pick up the apple and place it on the electronic scale"}  # SmolVLA language task
MODE=${MODE:-receding_horizon}           # 推荐 receding_horizon；可选 chunk / temporal_ensemble / single_step
# single_step 仅用于 debug，不满足当前 SmolVLA 155-176ms 推理延迟下的 10Hz 实时部署。
# RTC 当前只保留 capability detection/placeholder，本脚本默认不启用。
DURATION_S=${DURATION_S:-100}            # 运行时长；真实 SmolVLA+相机首次启动通常需要十几秒
CHUNK_SIZE=${CHUNK_SIZE:-50}            # SmolVLA 每次输出 action chunk 长度
EXECUTION_HORIZON=${EXECUTION_HORIZON:-10}  # chunk 模式一次消费多少步
REPLAN_INTERVAL=${REPLAN_INTERVAL:-5}   # receding_horizon 每多少个 10Hz step 切换新 chunk
CHUNK_TRIGGER_MARGIN=${CHUNK_TRIGGER_MARGIN:-1}  # 提前重规划余量；10 - 2 - 1 约第 7 步触发
CHUNK_EXPECTED_STALE_STEPS=${CHUNK_EXPECTED_STALE_STEPS:-2}  # p95 约 176ms，对应 2 个 10Hz tick
CHUNK_WORST_STALE_STEPS=${CHUNK_WORST_STALE_STEPS:-5}  # max 约 405ms，对应 5 个 10Hz tick
MAX_INFERENCE_CHUNKS=${MAX_INFERENCE_CHUNKS:-}  # 空表示不限制；smoke 常用 1
ACTION_PRINT_INTERVAL_STEPS=${ACTION_PRINT_INTERVAL_STEPS:-10}  # 每 N 个 10Hz step 打印耗时与 action；<=0 关闭
FIRST_ACTION_TIMEOUT_S=${FIRST_ACTION_TIMEOUT_S:-120}

# ===== 底层 motion policy =====
FEEDFORWARD=${FEEDFORWARD:-pressure_model}  # pressure_model / awac
FEEDBACK=${FEEDBACK:-fixed_k_integral}      # none / integral_lqr / fixed_k_integral
DEVICE=${DEVICE:-cuda}
DELTA_TCP_SCALE=${DELTA_TCP_SCALE:-1}       # SmolVLA delta TCP 缩放
PRESSURE_SCALE=${PRESSURE_SCALE:-1}         # 最终压力缩放，1 表示完整 0-3 bar 范围
FEEDBACK_GAIN_SCALE=${FEEDBACK_GAIN_SCALE:-0.1}
MAX_INTEGRAL_ERROR=${MAX_INTEGRAL_ERROR:-0.5}
MOTION_POLICY_READY_TIMEOUT_S=${MOTION_POLICY_READY_TIMEOUT_S:-120}  # SmolVLA 推理等待 motion policy 初始化的最长时间
Q_TCP6_WEIGHT=${Q_TCP6_WEIGHT:-1.0}
Q_STATE_TAIL_WEIGHT=${Q_STATE_TAIL_WEIGHT:-0.1}
Q_LATENT_WEIGHT=${Q_LATENT_WEIGHT:-0.1}
Q_INTEGRAL_WEIGHT=${Q_INTEGRAL_WEIGHT:-0.5}
R_WEIGHT=${R_WEIGHT:-50.0}

# ===== 动捕、串口、相机 =====
LUMO_IP=${LUMO_IP:-192.168.140.1}
RIGID_BODY_ID=${RIGID_BODY_ID:-1}
RECEIVE_TIMEOUT_MS=${RECEIVE_TIMEOUT_MS:-1000}
PORT=${PORT:-/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0}
ZED_INDEX=${ZED_INDEX:-}                 # 空表示自动找设备名含 ZED 的 video 节点
ZED_EYE=${ZED_EYE:-left}
ZED_WIDTH=${ZED_WIDTH:-2560}             # ZED side-by-side 输入宽度；left half 进入 cam_1
ZED_HEIGHT=${ZED_HEIGHT:-720}
ZED_FPS=${ZED_FPS:-30}
REALSENSE_SERIAL_CAM2=${REALSENSE_SERIAL_CAM2:-401522072797}
REALSENSE_SERIAL_CAM3=${REALSENSE_SERIAL_CAM3:-408322072769}
ZED_WARMUP_USABLE_FRAMES=${ZED_WARMUP_USABLE_FRAMES:-10}
REALSENSE_WARMUP_USABLE_FRAMES=${REALSENSE_WARMUP_USABLE_FRAMES:-10}
MIN_REALSENSE_MEAN=${MIN_REALSENSE_MEAN:-40}
INITIAL_GRIPPER_OPEN=${INITIAL_GRIPPER_OPEN:-1}  # LuMo 无夹爪传感器；state13 最后一维初始用该估计值
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0.1}  # action gripper < 该值 -> close/0
GRIPPER_OPEN_THRESHOLD=${GRIPPER_OPEN_THRESHOLD:-0.995}    # action gripper > 该值 -> open/1；中间区间保持上次状态
CAMERA_PREVIEW_SCALE=${CAMERA_PREVIEW_SCALE:-0.5}
CAMERA_PREVIEW_FPS=${CAMERA_PREVIEW_FPS:-10}
CAMERA_PREVIEW_WINDOW=${CAMERA_PREVIEW_WINDOW:-soft_vla_live_cameras}

# ===== 数据与日志 =====
EPISODE_INDEX=${EPISODE_INDEX:-0}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
REPO_ID=${REPO_ID:-local/soft_robot_7_03_1_delta_tcp}
IO_LABEL=mock_pressure
[[ "$RUN_HARDWARE" == "1" ]] && IO_LABEL=hardware
OBS_LABEL=replay_obs
[[ "$LIVE_OBSERVATION" == "1" ]] && OBS_LABEL=live_obs
RUN_LABEL=${RUN_LABEL:-"${VLA_BACKEND}_${OBS_LABEL}_${MODE}_${FEEDFORWARD}_${FEEDBACK}_${IO_LABEL}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/soft_vla/tests/tmp/$RUN_LABEL"}
LOG_JSONL=${LOG_JSONL:-"$OUTPUT_DIR/smolvla_deploy.jsonl"}

args=(
  soft_vla/scripts/deploy_smolvla_real.py
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
  --action-print-interval-steps "$ACTION_PRINT_INTERVAL_STEPS"
  --initial-gripper-open "$INITIAL_GRIPPER_OPEN"
  --gripper-close-threshold "$GRIPPER_CLOSE_THRESHOLD"
  --gripper-open-threshold "$GRIPPER_OPEN_THRESHOLD"
  --first-action-timeout-s "$FIRST_ACTION_TIMEOUT_S"
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

echo "[soft_vla] requested motion policy: FEEDFORWARD=$FEEDFORWARD FEEDBACK=$FEEDBACK DEVICE=$DEVICE PRESSURE_SCALE=$PRESSURE_SCALE FEEDBACK_GAIN_SCALE=$FEEDBACK_GAIN_SCALE"
"$PY" "${args[@]}"
