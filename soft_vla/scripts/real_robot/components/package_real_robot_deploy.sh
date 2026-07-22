#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  soft_vla/scripts/real_robot/components/package_real_robot_deploy.sh [--with-smolvla] [--with-smolvla-extra] [--with-lerobot-videos] [--output PATH]

Build a small deployment tarball for the real soft-robot controller.

Default package:
  - soft_vla source/scripts/configs/docs/validation
  - motion-control runtime dependencies and selected checkpoints
  - LeRobot parquet/meta/raw_pressure data for episode replay

Optional:
  --with-smolvla          include the selected 020000 SmolVLA pretrained_model (~865M)
  --with-smolvla-extra    include two extra SmolVLA pretrained_model checkpoints: 015000 and 010000 (~1.7G)
  --with-lerobot-videos   include LeRobot RGB videos for offline SmolVLA smoke (~699M)

Run this from the repository root: /home/cao/skill_learning_soft_robot
EOF
}

WITH_SMOLVLA=0
WITH_SMOLVLA_EXTRA=0
WITH_LEROBOT_VIDEOS=0
OUTPUT="../skill_learning_soft_robot_real_robot_deploy.tar.gz"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-smolvla)
      WITH_SMOLVLA=1
      shift
      ;;
    --with-smolvla-extra)
      WITH_SMOLVLA=1
      WITH_SMOLVLA_EXTRA=1
      shift
      ;;
    --with-lerobot-videos)
      WITH_LEROBOT_VIDEOS=1
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d soft_vla || ! -d motion_control_training || ! -d lerobot_conversion ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
STAGE="$TMPDIR/skill_learning_soft_robot"
mkdir -p "$STAGE"
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

copy_path() {
  local src="$1"
  local dst="$STAGE/$src"
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    rsync -a \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      "$src/" "$dst/"
  else
    rsync -a \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      "$src" "$dst"
  fi
}

copy_path soft_vla/README.md
copy_path soft_vla/ARCHITECTURE.md
copy_path soft_vla/pyproject.toml
copy_path soft_vla/requirements.txt
copy_path soft_vla/environment.cuda.yml
copy_path soft_vla/src
copy_path soft_vla/scripts
copy_path soft_vla/configs
copy_path soft_vla/docs
copy_path soft_vla/validation

copy_path motion_control_training/feedforward_pressure/infer_pressure.py
copy_path motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt
copy_path motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/config.json
copy_path motion_control_training/KORL
copy_path motion_control_training/koopman/model.py
copy_path motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt
copy_path motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/config.json

copy_path lerobot_conversion/README.md
copy_path lerobot_conversion/add_raw_pressure_sidecar.py
copy_path lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/data
copy_path lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/meta
copy_path lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/raw_pressure
copy_path lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/validation_report.json

if [[ "$WITH_SMOLVLA" -eq 1 ]]; then
  copy_path soft_vla/outputs/full_runs/smolvla_full_full20000_bs8_20260704_180614/checkpoints/020000/pretrained_model
fi

if [[ "$WITH_SMOLVLA_EXTRA" -eq 1 ]]; then
  copy_path soft_vla/outputs/full_runs/smolvla_full_full20000_bs8_20260704_180614/checkpoints/015000/pretrained_model
  copy_path soft_vla/outputs/full_runs/smolvla_full_full20000_bs8_20260704_180614/checkpoints/010000/pretrained_model
fi

if [[ "$WITH_LEROBOT_VIDEOS" -eq 1 ]]; then
  copy_path lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp/videos
fi

mkdir -p "$(dirname "$OUTPUT")"
tar -C "$TMPDIR" -czf "$OUTPUT" skill_learning_soft_robot

echo "Wrote: $OUTPUT"
echo "Staged size:"
du -sh "$STAGE"
echo "Archive size:"
du -h "$OUTPUT"
