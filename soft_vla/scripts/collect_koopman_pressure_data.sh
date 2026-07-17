#!/usr/bin/env bash
set -Eeuo pipefail

# Koopman 16路气压数据采集启动脚本：
#   前12路为本体激励，后4路固定为夹爪关闭。
#   默认采集 8 terms × 20 amplitudes × 400 steps，约 22 分钟。
#   始终使用 --resume：首次运行正常创建，中断后再次执行会自动续采。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/cao/miniconda3/envs/soft_vla_cuda/bin/python}"
PORT="${PORT:-/dev/ttyUSB0}"
LUMO_IP="${LUMO_IP:-192.168.140.1}"
RIGID_BODY_ID="${RIGID_BODY_ID:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Collected_Data/koopman_pressure16}"
TERMS="${TERMS:-0:8}"
AMPLITUDE_INDICES="${AMPLITUDE_INDICES:-0:20}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-400}"
FREQUENCY="${FREQUENCY:-50}"
NOISE_LAMBDA="${NOISE_LAMBDA:-0.2}"
SEED="${SEED:-42}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "错误：找不到 Python 解释器：${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -e "${PORT}" ]]; then
    echo "错误：串口设备不存在：${PORT}" >&2
    echo "可通过 PORT 指定，例如：" >&2
    echo "  PORT=/dev/serial/by-id/你的设备 $0" >&2
    exit 1
fi

"${PYTHON_BIN}" -c "import numpy, scipy, serial" || {
    echo "错误：Python 环境缺少 numpy、scipy 或 pyserial。" >&2
    exit 1
}

echo "开始/继续 Koopman 数据采集"
echo "  LuMo IP:       ${LUMO_IP}"
echo "  rigid body id: ${RIGID_BODY_ID}"
echo "  serial port:   ${PORT}"
echo "  output dir:    ${OUTPUT_DIR}"
echo "  terms:         ${TERMS}"
echo "  amplitudes:    ${AMPLITUDE_INDICES}"
echo "  repeats:       ${REPEATS}"
echo "  steps:         ${STEPS}"
echo "  frequency:     ${FREQUENCY} Hz"
echo "按 Ctrl+C 可安全中断；再次运行本脚本将从已完成 episode 后续采。"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_koopman_pressure_data.py" \
    --hardware-enabled \
    --resume \
    --ip "${LUMO_IP}" \
    --rigid-body-id "${RIGID_BODY_ID}" \
    --port "${PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --terms "${TERMS}" \
    --amplitude-indices "${AMPLITUDE_INDICES}" \
    --repeats "${REPEATS}" \
    --steps "${STEPS}" \
    --frequency "${FREQUENCY}" \
    --noise-lambda "${NOISE_LAMBDA}" \
    --seed "${SEED}"
