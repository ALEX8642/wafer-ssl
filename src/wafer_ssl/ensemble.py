"""
ensemble.py — Multi-model ensemble evaluation for wafer-defect-classifier checkpoints.

Loads N independently trained checkpoints, runs TTA×8 on each, averages the
probability matrices, then reports per-class F1 and macro-F1. Ensemble variance
reduction typically yields +0.3–0.8pp macro-F1 over the best single model.

Why ensemble works:
    Each training run (different seed) converges to a different local optimum. The
    models make different errors — one may confidently misclassify a particular Scratch
    as Loc while another correctly classifies it. Averaging softmax probabilities lets
    the consensus win and the outlier lose weight.

Usage:
    python -m wafer_ssl.ensemble \\
        --checkpoints /path/to/best.pt /path/to/seed7/best.pt /path/to/seed123/best.pt \\
        --data-root /path/to/wafer-defect-classifier/data/raw \\
        --config /path/to/wafer-defect-classifier/configs/baseline.yaml

    All --config fields can be overridden as CLI args (same pattern as wafer.evaluate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report

from wafer.config import WaferConfig, build_arg_parser
from wafer.data import get_dataloaders, CLASS_NAMES
from wafer.evaluate import tta_predict, apply_thresholds
from wafer.model import build_model


def evaluate_ensemble(cfg: WaferConfig, checkpoint_paths: list[Path]) -> None:
    """
    Load all checkpoints, run TTA on each, average probabilities, report metrics.

    Each checkpoint may have been trained with different architecture flags (e.g.
    different seeds but same focal+CBAM config). Architecture flags are read from
    each checkpoint's saved cfg to avoid key mismatches.
    """
    if not checkpoint_paths:
        raise ValueError("Provide at least one checkpoint path.")

    print(f"Ensemble evaluation: {len(checkpoint_paths)} model(s)")

    # Load calibration temperature from the first checkpoint's output dir
    # (all models in an ensemble should share the same calibration)
    temperature = 1.0
    temp_path = Path(checkpoint_paths[0]).parent / "temperature.json"
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = float(json.load(f)["temperature"])
        print(f"Temperature: T={temperature:.4f}  (from {temp_path})")
    else:
        print("Temperature: T=1.0  (temperature.json not found — run wafer.calibrate)")

    # Load per-class thresholds from the first checkpoint's output dir
    thresholds: dict = {}
    thresh_path = Path(checkpoint_paths[0]).parent / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        print(f"Thresholds: {len(thresholds)} class thresholds loaded")

    # Load each model
    models = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        saved_cfg = ckpt.get("cfg", {})
        cfg.cbam = bool(saved_cfg.get("cbam", cfg.cbam))
        cfg.cbam_reduction = int(saved_cfg.get("cbam_reduction", cfg.cbam_reduction))
        class_to_idx = ckpt["class_to_idx"]
        num_classes = len(class_to_idx)

        model = build_model(cfg, num_classes=num_classes).to(cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded: {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
              f"val F1 {ckpt.get('val_macro_f1', float('nan')):.4f})")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    _, _, test_loader, _, _ = get_dataloaders(cfg)

    # Collect per-model probabilities
    all_targets: list[int] = []
    per_model_probs: list[list[np.ndarray]] = [[] for _ in models]

    for inputs, targets in test_loader:
        all_targets.extend(targets.numpy())
        for m_idx, model in enumerate(models):
            probs = tta_predict(model, inputs, cfg.device, temperature)
            per_model_probs[m_idx].append(probs)

    # Stack each model's probabilities → (N, num_classes)
    stacked = [np.vstack(p) for p in per_model_probs]

    # Per-model F1 (individual baselines for comparison)
    print(f"\n{'Model':<55}  {'Macro-F1':>10}")
    print("-" * 68)
    for path, probs_arr in zip(checkpoint_paths, stacked):
        preds = (apply_thresholds(probs_arr, thresholds, class_names)
                 if thresholds else probs_arr.argmax(axis=1))
        report = classification_report(
            all_targets, preds, target_names=class_names,
            zero_division=0, output_dict=True,
        )
        f1 = report["macro avg"]["f1-score"]
        print(f"  {str(path)[-50:]:<52}  {f1:.4f}")

    # Ensemble: average probabilities across all models
    ensemble_probs = np.mean(stacked, axis=0)
    targets_arr = np.array(all_targets)
    preds = (apply_thresholds(ensemble_probs, thresholds, class_names)
             if thresholds else ensemble_probs.argmax(axis=1))

    macro_f1 = float(
        classification_report(targets_arr, preds, target_names=class_names,
                               zero_division=0, output_dict=True)["macro avg"]["f1-score"]
    )
    bal_acc = balanced_accuracy_score(targets_arr, preds)

    print("\n" + "=" * 64)
    print(f"ENSEMBLE RESULTS  [{len(models)} models, TTA×8, per-class τ]")
    print("=" * 64)
    print(f"  Macro-F1          : {macro_f1:.4f}  ← headline metric")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print("  (Plain accuracy suppressed — misleading under 85 % class imbalance.)\n")
    print(classification_report(targets_arr, preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    parser = build_arg_parser("wafer-ssl ensemble")
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", required=True,
        help="Paths to .pt checkpoint files to ensemble (space-separated)",
    )
    args = parser.parse_args()
    cfg = WaferConfig.from_yaml_and_args(args.config, args)
    evaluate_ensemble(cfg, args.checkpoints)
