#!/usr/bin/env bash
# run_phases_qr.sh — Phase Q (pretrained fine-tune) + Phase R (ensemble seeds)
#
# Run this after SimCLR pretraining completes.
#
# Usage (from wafer-ssl root on the 5090):
#   bash scripts/run_phases_qr.sh /path/to/wafer-defect-classifier
#
# Example:
#   bash scripts/run_phases_qr.sh /home/alex8642/wafer-classifier/wafer-defect-classifier

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-wafer-defect-classifier>"
  exit 1
fi

WAFER_DIR="$(realpath "$1")"
SSL_DIR="$(realpath "$(dirname "$0")/..")"
PYTHON="${WAFER_DIR}/.venv/bin/python"
BACKBONE_SRC="${SSL_DIR}/outputs/pretrained_backbone.pt"
BACKBONE_DST="${WAFER_DIR}/outputs/pretrained_backbone.pt"
YAML="${WAFER_DIR}/configs/baseline.yaml"

echo "=== wafer-ssl Phase Q + R runner ==="
echo "  wafer-defect-classifier : ${WAFER_DIR}"
echo "  backbone source         : ${BACKBONE_SRC}"

# --- sanity checks ---
[[ -f "${BACKBONE_SRC}" ]] || { echo "ERROR: backbone not found at ${BACKBONE_SRC} — run pretrain first."; exit 1; }
[[ -f "${PYTHON}" ]] || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${YAML}" ]] || { echo "ERROR: baseline.yaml not found at ${YAML}"; exit 1; }

# --- Step 1: copy backbone ---
mkdir -p "${WAFER_DIR}/outputs"
echo ""
echo "[Phase Q] Copying backbone..."
cp "${BACKBONE_SRC}" "${BACKBONE_DST}"
echo "  -> ${BACKBONE_DST}"

# --- Step 2: wire backbone into baseline.yaml (idempotent) ---
if grep -q "backbone_ckpt_path: \"\"" "${YAML}"; then
  sed -i 's|backbone_ckpt_path: ""|backbone_ckpt_path: outputs/pretrained_backbone.pt|' "${YAML}"
  echo "[Phase Q] baseline.yaml: backbone_ckpt_path set."
elif grep -q "backbone_ckpt_path: outputs/pretrained_backbone.pt" "${YAML}"; then
  echo "[Phase Q] baseline.yaml: backbone_ckpt_path already set — skipping."
else
  echo "ERROR: unexpected backbone_ckpt_path state in ${YAML}. Edit manually."
  exit 1
fi

# --- Step 3: Phase Q — train seed 42 (default) ---
echo ""
echo "[Phase Q] Training seed 42 with pretrained backbone (~40 min)..."
(cd "${WAFER_DIR}" && "${PYTHON}" -m wafer.train)

echo "[Phase Q] Calibrating..."
(cd "${WAFER_DIR}" && "${PYTHON}" -m wafer.calibrate)

echo "[Phase Q] Evaluating..."
(cd "${WAFER_DIR}" && "${PYTHON}" -m wafer.evaluate)

# --- Step 4: Phase R — three more seeds ---
for SEED in 7 123 456; do
  echo ""
  echo "[Phase R] Training seed ${SEED}..."
  (cd "${WAFER_DIR}" && "${PYTHON}" -m wafer.train \
    --seed "${SEED}" \
    --output-dir "outputs/seed${SEED}")
done

# --- Step 5: Ensemble evaluation ---
echo ""
echo "[Phase R] Ensemble evaluation (TTA×8, 4 models)..."
(cd "${SSL_DIR}" && "${PYTHON}" -m wafer_ssl.ensemble \
  --checkpoints \
    "${WAFER_DIR}/outputs/best.pt" \
    "${WAFER_DIR}/outputs/seed7/best.pt" \
    "${WAFER_DIR}/outputs/seed123/best.pt" \
    "${WAFER_DIR}/outputs/seed456/best.pt" \
  --config "${YAML}" \
  --data-root "${WAFER_DIR}/data/raw")

echo ""
echo "=== Done. Paste the results above back to Claude to update the docs. ==="
