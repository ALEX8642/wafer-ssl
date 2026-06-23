#!/usr/bin/env bash
# run_control_ensemble.sh — From-scratch (no SSL backbone) ensemble CONTROL.
#
# Ablation for Phase Q/R: the 4-model SSL ensemble reached macro-F1 0.9423, but
# single-model SSL came in AT/BELOW the from-scratch baseline (seed42: 0.9057 vs
# 0.9157). So we cannot attribute the ensemble gain to SSL without a control.
# This trains the SAME 4 seeds (42/99/123/456) with the SAME focal+CBAM config
# but RANDOM init (--backbone-ckpt-path ""), then ensembles. If this also lands
# ~0.94, the lever was ENSEMBLING, not SSL.
#
# seed42 reuses the documented Phase F model (outputs/phaseF_backup/best.pt) — the
# authentic from-scratch seed-42 baseline (test 0.9157), already calibrated, so it
# drives the ensemble's thresholds/temperature exactly as the SSL seed-42 did.
# Only seeds 99/123/456 are retrained from scratch. Per-seed patience mirrors the
# SSL run (99->15, 123/456->10) so the ONLY difference vs the SSL ensemble is the
# backbone init.
#
# Writes ONLY to outputs/scratch_seed*/ — does NOT touch the SSL checkpoints or
# diagnostics in outputs/.
#
# Usage:
#   bash scripts/run_control_ensemble.sh /home/alex8642/wafer-classifier/wafer-defect-classifier

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-wafer-defect-classifier>"
  exit 1
fi

WAFER_DIR="$(realpath "$1")"
SSL_DIR="$(realpath "$(dirname "$0")/..")"
PYTHON="${WAFER_DIR}/.venv/bin/python"
YAML="${WAFER_DIR}/configs/baseline.yaml"
SEED42_CKPT="${WAFER_DIR}/outputs/phaseF_backup/best.pt"

echo "=== From-scratch ensemble control (random init, same seeds/config) ==="
echo "  wafer-defect-classifier : ${WAFER_DIR}"
echo "  seed-42 baseline        : ${SEED42_CKPT}"

# --- sanity checks ---
[[ -f "${PYTHON}" ]]      || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${YAML}" ]]        || { echo "ERROR: baseline.yaml not found at ${YAML}"; exit 1; }
[[ -f "${SEED42_CKPT}" ]] || { echo "ERROR: Phase F seed-42 model not found at ${SEED42_CKPT}"; \
                               echo "       (expected the best.pt you moved into outputs/phaseF_backup/)."; exit 1; }

# --- Train seeds 99/123/456 from RANDOM init. Resumable: skip seeds already done.
#     patience mirrors the SSL run exactly (99->15, 123->10, 456->10). ---
train_scratch() {  # train_scratch <seed> <patience>
  local seed="$1" pat="$2" dir="outputs/scratch_seed${seed}"
  echo ""
  if [[ -f "${WAFER_DIR}/${dir}/best.pt" ]]; then
    echo "[control] ${dir}/best.pt exists — skipping (resume)."
    return
  fi
  echo "[control] Training seed ${seed} from scratch (random init, patience ${pat})..."
  (cd "${WAFER_DIR}" && "${PYTHON}" -m wafer.train \
    --seed "${seed}" \
    --patience "${pat}" \
    --backbone-ckpt-path "" \
    --output-dir "${dir}")
}

train_scratch 99  15
train_scratch 123 10
train_scratch 456 10

# --- Ensemble. seed42 (Phase F baseline) is checkpoint[0] so its calibrated
#     temperature.json + thresholds.json drive the decision rule, mirroring the
#     SSL ensemble whose checkpoint[0] was the calibrated seed-42 model. ---
echo ""
echo "[control] From-scratch ensemble evaluation (TTA×8, 4 models)..."
(cd "${SSL_DIR}" && "${PYTHON}" -m wafer_ssl.ensemble \
  --checkpoints \
    "${SEED42_CKPT}" \
    "${WAFER_DIR}/outputs/scratch_seed99/best.pt" \
    "${WAFER_DIR}/outputs/scratch_seed123/best.pt" \
    "${WAFER_DIR}/outputs/scratch_seed456/best.pt" \
  --config "${YAML}" \
  --data-root "${WAFER_DIR}/data/raw")

echo ""
echo "=== Control done. Compare this ensemble macro-F1 to the SSL ensemble's 0.9423."
echo "    If ~equal -> the lever was ensembling, not SSL. Paste the block to Claude. ==="
