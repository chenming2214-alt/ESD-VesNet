#!/bin/bash
set -euo pipefail

# ESD-VesNet training launcher
# Paper: ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for
# Endoscopic Submucosal Dissection with Hard Negative Mining
# 用法示例：
#   bash scripts/train_edl_hnm_fullsam_0105.sh --gpus 3 --devices 0,1,2 --hnm-scan 600 --batch-size 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${PROJ_DIR}/legacy_0105/train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py"

GPUS=1
DEVICES="0"
MASTER_PORT="${MASTER_PORT:-29521}"
NUM_WORKERS=""
HNM_SCAN=""
SKIP_HNM_INIT=0
EVAL_ONLY=0
CKPT=""
BATCH_SIZE=""
GRAD_ACCUM=""
NO_AMP=0

ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --hnm-scan) HNM_SCAN="$2"; shift 2;;
    --skip-hnm-init) SKIP_HNM_INIT=1; shift 1;;
    --eval-only) EVAL_ONLY=1; shift 1;;
    --ckpt) CKPT="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2;;
    --no-amp) NO_AMP=1; shift 1;;
    *) ARGS+=("$1"); shift 1;;
  esac
done

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[error] 找不到训练脚本: ${SCRIPT_PATH}"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${DEVICES}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJ_DIR}/../sam3-main:${PYTHONPATH:-}"

# ---- reduce default log spam (can be overridden by user env) ----
# NCCL_* env vars are deprecated in recent PyTorch; prefer TORCH_NCCL_* to avoid warnings.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
# Default to WARN to avoid thousands of NCCL INFO lines; set NCCL_DEBUG=INFO if you need debugging.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Disable distributed debug spam by default; set TORCH_DISTRIBUTED_DEBUG=DETAIL when debugging DDP.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
# Suppress common warning spam; override via PYTHONWARNINGS if needed.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${NUM_WORKERS}" ]]; then ARGS+=(--num-workers "${NUM_WORKERS}"); fi
if [[ -n "${HNM_SCAN}" ]]; then ARGS+=(--hnm-scan "${HNM_SCAN}"); fi
if [[ "${SKIP_HNM_INIT}" -eq 1 ]]; then ARGS+=(--skip-hnm-init); fi
if [[ "${EVAL_ONLY}" -eq 1 ]]; then ARGS+=(--eval-only); fi
if [[ -n "${CKPT}" ]]; then ARGS+=(--ckpt "${CKPT}"); fi
if [[ -n "${BATCH_SIZE}" ]]; then ARGS+=(--batch-size "${BATCH_SIZE}"); fi
if [[ -n "${GRAD_ACCUM}" ]]; then ARGS+=(--grad-accum "${GRAD_ACCUM}"); fi
if [[ "${NO_AMP}" -eq 1 ]]; then ARGS+=(--no-amp); fi

echo "[run] project=${PROJ_DIR}"
echo "[run] script=${SCRIPT_PATH}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] gpus=${GPUS} master_port=${MASTER_PORT}"

cd "${PROJ_DIR}"

if [[ "${GPUS}" -gt 1 ]]; then
  echo "[run] torchrun --nproc_per_node=${GPUS}"
  torchrun --nproc_per_node="${GPUS}" \
           --master_port="${MASTER_PORT}" \
           "${SCRIPT_PATH}" \
           "${ARGS[@]}"
else
  echo "[run] python"
  python "${SCRIPT_PATH}" "${ARGS[@]}"
fi
