# wafer-ssl

Self-supervised and ensemble extensions for
[wafer-defect-classifier](https://github.com/alex8642/wafer-defect-classifier).

**Goal:** Push the WM-811K 9-class macro-F1 beyond the 0.9157 achieved with
focal loss + CBAM by using the 638k unlabeled production maps for self-supervised
contrastive pretraining (SimCLR), then ensembling independently fine-tuned models.

---

## Motivation

The base project's Phase S (pseudo-labeling) regressed from 0.9157 to 0.9085.
Root cause: hard pseudo-labels at 95% confidence carry ~5% label noise —
enough to silently degrade Scratch/Loc/Random recall on rare tail classes.

**Self-supervised contrastive learning eliminates label noise entirely.**
SimCLR learns representations by contrasting augmented views of the same map
without any labels. The 638k unlabeled maps are used as unlabeled structural
examples, not as labeling candidates.

---

## What's in this repo

| Module | Description |
|---|---|
| `src/wafer_ssl/pretrain.py` | SimCLR pretraining on 638k unlabeled WM-811K maps |
| `src/wafer_ssl/ensemble.py` | Multi-checkpoint ensemble evaluation |
| `configs/pretrain.yaml` | Pretraining hyperparameters |
| `PLAN.md` | Full plan, execution timeline, and honest ceiling analysis |

---

## Setup

```bash
# 1. Install the base package (wafer-defect-classifier must be on disk)
pip install -e /path/to/wafer-defect-classifier

# 2. Install this package
pip install -e .
```

Both packages must be installed in the same virtualenv.

---

## Usage

### Phase P — Pretrain backbone (~3–5 hours on RTX 5090)

```bash
python -m wafer_ssl.pretrain --config configs/pretrain.yaml
```

Edit `configs/pretrain.yaml` first to set `data_root` to your LSWMD.pkl location.

Saves `outputs/pretrained_backbone.pt` after each improvement in NT-Xent loss.
Prints loss every 10 epochs — expect ~0.3–0.5 at epoch 1 dropping to ~0.15–0.20 by epoch 200.
(Wafer maps are binary spatial patterns, not natural images — the contrastive task is easier
than ImageNet, so NT-Xent loss is lower than the ~5.0 quoted in the SimCLR paper.)

### Phase Q — Fine-tune on pretrained backbone

```bash
# Copy backbone to the base repo's outputs directory
cp outputs/pretrained_backbone.pt /path/to/wafer-defect-classifier/outputs/

# In wafer-defect-classifier/configs/baseline.yaml, add/set:
#   backbone_ckpt_path: outputs/pretrained_backbone.pt
#   loss: focal
#   cbam: true
#   num_epochs: 40
#   patience: 10

cd /path/to/wafer-defect-classifier
.venv/bin/python -m wafer.train && \
.venv/bin/python -m wafer.calibrate && \
.venv/bin/python -m wafer.evaluate
```

### Phase R — Ensemble (3 additional seeds)

```bash
# In wafer-defect-classifier (same backbone_ckpt_path, different seeds):
.venv/bin/python -m wafer.train --seed 7   --output-dir outputs/seed7
.venv/bin/python -m wafer.train --seed 123 --output-dir outputs/seed123
.venv/bin/python -m wafer.train --seed 456 --output-dir outputs/seed456

# Evaluate ensemble from this repo:
python -m wafer_ssl.ensemble \
  --checkpoints \
    /path/to/wafer-defect-classifier/outputs/best.pt \
    /path/to/wafer-defect-classifier/outputs/seed7/best.pt \
    /path/to/wafer-defect-classifier/outputs/seed123/best.pt \
    /path/to/wafer-defect-classifier/outputs/seed456/best.pt \
  --config /path/to/wafer-defect-classifier/configs/baseline.yaml \
  --data-root /path/to/wafer-defect-classifier/data/raw
```

---

## Results (to be updated)

| Experiment | Val macro-F1 | Test macro-F1 |
|---|---|---|
| Phase F baseline (focal+CBAM) | 0.9265 | **0.9157** |
| Phase P+Q (SimCLR + fine-tune) | — | — |
| Phase R (4-model ensemble) | — | — |

---

## References

- Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Woo et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV 2018.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
- Wu et al. (2015). *Wafer Map Failure Pattern Recognition.* IEEE Trans. Semiconductor Manufacturing.
