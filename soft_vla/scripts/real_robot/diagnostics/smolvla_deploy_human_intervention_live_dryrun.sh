#!/usr/bin/env bash
set -euo pipefail

# 第 2 层：无机械臂压力下发 dry-run。
# 开启：视觉、LuMo 状态、Xbox 手柄、SmolVLA 推理、camera preview、episode 记录。
# 关闭：真实串口压力输出。即 RUN_HARDWARE 固定为 0。
#
# 用法：
#   bash soft_vla/scripts/real_robot/diagnostics/smolvla_deploy_human_intervention_live_dryrun.sh

ROOT=${ROOT:-/home/cao/skill_learning_soft_robot}
cd "$ROOT"

export RUN_HARDWARE=0
export STATE_HARDWARE=${STATE_HARDWARE:-1}
export VLA_BACKEND=${VLA_BACKEND:-smolvla}
export LIVE_OBSERVATION=${LIVE_OBSERVATION:-1}
export CAMERA_PREVIEW=${CAMERA_PREVIEW:-1}
export HUMAN_INTERVENTION=1
export MODE=${MODE:-receding_horizon}
export DURATION_S=${DURATION_S:-0}
export WAIT_FOR_START_KEY=${WAIT_FOR_START_KEY:-1}
export RUN_LABEL=${RUN_LABEL:-human_live_dryrun_no_pressure}

echo "[soft_vla] live dry-run: vision/state/gamepad/SmolVLA enabled, pressure output disabled"
bash soft_vla/scripts/real_robot/deploy/smolvla_deploy_human_intervention.sh
