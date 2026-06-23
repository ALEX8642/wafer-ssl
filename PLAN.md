# wafer-ssl — Improvement Plan: Push Toward 0.98 Macro-F1

## Context

Extends [wafer-defect-classifier](https://github.com/alex8642/wafer-defect-classifier),
which achieves test macro-F1 **0.9157** on the WM-811K 9-class problem using
ResNet-18 + focal loss + CBAM attention.

The prior semi-supervised attempt (Phase S, pseudo-labeling at 0.95 confidence)
regressed to 0.9085. Root cause: ~5% label noise on 29k tail-class pseudo-labels
silently degraded Scratch/Loc/Random recall. This repo takes the principled
alternative: **self-supervised contrastive pretraining** on the 638k unlabeled maps —
no labels, no noise.

---

## Honest ceiling check

| Class | Current F1 | Needed for 0.98 macro | Gap |
|---|---|---|---|
| Loc | 0.84 | 0.98 | +14pp |
| Scratch | 0.86 | 0.98 | +12pp |
| Donut | 0.86 | 0.98 | +12pp |
| Edge-Loc | 0.89 | 0.98 | +9pp |
| Random | 0.91 | 0.98 | +7pp |
| Center | 0.95 | 0.98 | +3pp |
| Near-full | 0.95 | 0.98 | +3pp |
| Edge-Ring | 0.99 | — | ≈0 |
| none | 0.99 | — | ≈0 |

**Realistic target after this plan: 0.94–0.96.** We document wherever we land.
→ **Achieved: 0.9423** (4-model SimCLR ensemble). See [Outcome](#outcome) below.

---

## Phase P — SimCLR self-supervised pretraining

**File:** `src/wafer_ssl/pretrain.py`

SimCLR (Chen et al. 2020): for each unlabeled map produce 2 augmented views,
encode both with ResNet-18 backbone, project to 128-dim embedding with a 2-layer
MLP, then minimize NT-Xent loss treating both views of the same map as a positive
pair and all others in the batch as negatives.

**Why this fixes Phase S:** No labels involved — no label noise possible.
The 638k unlabeled maps teach the backbone to discriminate spatial structure
(ring vs. cluster vs. streak) purely from augmentation invariance.

**Augmentations for contrastive pairs:**
- D4 group (4 rotations × 2 flips) — same as supervised training
- Random crop 85–100% + resize back — adds positional variation
- Gaussian blur σ=0.5–2.0 — simulates tool-to-tool map resolution differences

**Training:** 200 epochs, batch_size=256, Adam lr=3e-4, cosine decay.
~3–5 hours on 5090. Saves `outputs/pretrained_backbone.pt`.

---

## Phase Q — Supervised fine-tuning on pretrained backbone

Copy `outputs/pretrained_backbone.pt` to `wafer-defect-classifier/outputs/`.
Set `backbone_ckpt_path: outputs/pretrained_backbone.pt` in `baseline.yaml`
(wafer-defect-classifier v0.1.0+ supports this field).

Same focal+CBAM config as the best Phase F result:
```yaml
loss: focal
cbam: true
num_epochs: 40
patience: 10
backbone_ckpt_path: outputs/pretrained_backbone.pt
```

Then in wafer-defect-classifier:
```bash
.venv/bin/python -m wafer.train && .venv/bin/python -m wafer.calibrate && .venv/bin/python -m wafer.evaluate
```

The pretrained backbone initializes the ResNet-18 stages; CBAM modules remain
randomly initialized and are learned during fine-tuning. `strict=False` loading
handles the CBAM key mismatch cleanly.

---

## Phase R — Ensemble

**File:** `src/wafer_ssl/ensemble.py`

After Phase Q produces a pretrained+fine-tuned checkpoint, retrain 3 more
independent models with different seeds (seeds 7, 123, 456) using the SAME
pretrained backbone — same `backbone_ckpt_path`, different `--seed` and
`--output-dir`:

```bash
.venv/bin/python -m wafer.train --seed 7   --output-dir outputs/seed7
.venv/bin/python -m wafer.train --seed 123 --output-dir outputs/seed123
.venv/bin/python -m wafer.train --seed 456 --output-dir outputs/seed456
```

Evaluate the ensemble from this repo:
```bash
.venv/bin/python -m wafer_ssl.ensemble \
  --checkpoints \
    ../wafer-defect-classifier/outputs/best.pt \
    ../wafer-defect-classifier/outputs/seed7/best.pt \
    ../wafer-defect-classifier/outputs/seed123/best.pt \
    ../wafer-defect-classifier/outputs/seed456/best.pt \
  --config ../wafer-defect-classifier/configs/baseline.yaml \
  --data-root ../wafer-defect-classifier/data/raw
```

Expected gain from variance reduction: **+0.3–0.8pp** macro-F1.
Ensemble inference is 4 × 11M params → still lightweight; ~4× slower
than single model but well under 1 second per wafer on any modern GPU.

---

## Execution timeline

| Step | Command | Where | Est. time |
|---|---|---|---|
| Install | `pip install -e . && pip install -e ../wafer-defect-classifier` | wafer-ssl | 1 min |
| Phase P | `python -m wafer_ssl.pretrain` | wafer-ssl | 3–5 hrs |
| Copy weights | `cp outputs/pretrained_backbone.pt ../wafer-defect-classifier/outputs/` | wafer-ssl | instant |
| Phase Q (seed 42) | `python -m wafer.train && python -m wafer.calibrate && python -m wafer.evaluate` | wafer-defect-classifier | ~40 min |
| Phase R (3 seeds) | 3× `python -m wafer.train --seed N --output-dir outputs/seedN` | wafer-defect-classifier | ~2 hrs |
| Ensemble eval | `python -m wafer_ssl.ensemble --checkpoints ...` | wafer-ssl | ~10 min |

Total: ~8–10 hours unattended on 5090.

---

## Verification checkpoints

1. **Pretraining convergence (read the curve, not the floor):** A *very low, flat*
   NT-Xent loss is a red flag, not success. The first attempt (mild `crop_min: 0.85`)
   sat near ~0.46 from epoch 1 — the augmentations preserved the global wafer outline,
   a near-unique per-map fingerprint, so the task was trivially solvable by matching
   geometry instead of defect structure. Fixed with aggressive crop + cutout
   (`crop_min: 0.2`). A healthy run should *start materially higher* than 0.46 and show
   a clear downward trend as the backbone learns local defect texture. If it still
   starts very low and stays flat, augmentation is still too weak — do not trust the
   backbone; strengthen augmentation and re-run.

2. **Phase Q vs Phase F:** The pretrained backbone should reach ≥0.9265 val F1 (Phase F best)
   in fewer epochs. If it fails to match Phase F, the backbone features aren't transferring —
   investigate by freezing the backbone for epoch 1 and checking feature variance.

3. **Ensemble vs best individual:** `ensemble.py` prints per-model F1 before the ensemble result.
   Confirm ensemble > best individual. If not, the models are too correlated (check seeds differ).

4. **Regression guard:** Run `pytest tests/ -v` in wafer-defect-classifier after adding
   `backbone_ckpt_path` field — all 18 tests should stay green (additive config change).

---

## If we don't reach 0.98

Likely — getting Loc from 0.84 to 0.98 requires near-perfect classification of
a 718-sample class. The honest portfolio story:

> "Self-supervised contrastive pretraining on 638k unlabeled maps achieved
> macro-F1 X.XX on the 9-class imbalanced WM-811K benchmark. This compares to
> NVIDIA Cosmos Reason 1-7B's 96.8% accuracy on 8 balanced classes — a
> fundamentally easier evaluation. The remaining gap is driven by tail-class
> data scarcity (Donut: 111 test samples, Near-full: 30) rather than
> architectural limitations."

What would realistically break through to 0.98+:
- Expert re-annotation of ambiguous Loc/Scratch border cases
- Mixture-of-experts: separate specialist heads for tail vs. common classes
- ViT-based backbone at higher input resolution (52×52 → 224×224 via bilinear)

---

## Outcome

| Configuration | Test macro-F1 | Balanced acc |
|---|---|---|
| Phase F baseline (single model, seed 42) | 0.9157 | 0.9085 |
| From-scratch 4-seed ensemble (control) | 0.9339 | 0.9348 |
| **SimCLR + 4-seed ensemble** | **0.9423** | **0.9427** |

SimCLR pretraining contributed a **modest but consistent +0.8pp** (SimCLR ensemble
0.9423 vs from-scratch control 0.9339; the SimCLR model beat its from-scratch twin on
3 of 4 seeds and was ≥ on every class). The larger lever was ensembling. Single-model
SSL was within noise of the baseline, so the headline gain is honestly attributed:
~+1.8pp ensembling, ~+0.8pp self-supervision. We fell short of the 0.98 stretch goal
(tail-class scarcity, as predicted), landing squarely in the forecast 0.94–0.96 band.

The control matters: without it, the ensemble's 0.9423 could not be attributed to
SSL at all. Running `scripts/run_control_ensemble.sh` (from-scratch, same seeds and
config) is what makes the SSL claim defensible rather than speculative.
