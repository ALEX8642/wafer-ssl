"""
pretrain.py — SimCLR self-supervised backbone pretraining on 638k unlabeled WM-811K maps.

Extends wafer-defect-classifier (https://github.com/alex8642/wafer-defect-classifier)
with self-supervised contrastive pretraining. The pretrained backbone replaces random
initialization in the supervised fine-tuning step, giving the model a head start from
unlabeled production wafer data.

Why SimCLR instead of pseudo-labeling (Phase S):
    Phase S used hard pseudo-labels at 0.95 confidence. Even at 95% confidence, 5%
    label noise on 29k tail-class samples was enough to degrade Scratch/Loc/Random F1.
    SimCLR uses NO labels — it learns representations by contrasting augmented views of
    the same map, so there is no label noise. The contrastive signal is purely structural:
    "two augmented views of the same wafer map should be closer than views of different maps."

Algorithm (Chen et al. 2020, "A Simple Framework for Contrastive Learning of Visual Representations"):
    1. For each map in the batch, produce two independently augmented views (view1, view2)
    2. Encode each view: h = backbone(view)         — 512-dim representation
    3. Project: z = projector(h)                    — 128-dim contrastive embedding
    4. NT-Xent loss: treat (z1_i, z2_i) as a positive pair; all 2(N-1) others as negatives
    5. After pretraining: discard projector, save backbone weights

Augmentations for contrastive pairs (stronger than supervised training):
    - Random D4 rotation (0/90/180/270°) + random horizontal/vertical flip
    - Random crop 85–100% of map + resize back to input_size (positional variation)
    - Gaussian blur σ=0.5–2.0 (simulates map resolution differences across tools)
    No color jitter: wafer maps are 3-channel binary tensors, not RGB.

Usage:
    python -m wafer_ssl.pretrain --config configs/pretrain.yaml
    python -m wafer_ssl.pretrain --config configs/pretrain.yaml --epochs 100  # quick test

Output:
    outputs/pretrained_backbone.pt — saved after each improvement in NT-Xent loss.
    Contains {"backbone_state_dict": ..., "epoch": ..., "loss": ...}.
    Copy this to wafer-defect-classifier/outputs/ and set backbone_ckpt_path in
    baseline.yaml to use it for fine-tuning.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wafer.config import REPO_ROOT as WAFER_REPO_ROOT
from wafer.data import _is_labeled, encode_map, resize_map
from wafer.model import build_model, WaferConfig as _WaferConfig


# ---------------------------------------------------------------------------
# Contrastive augmentation — stronger than supervised training
# ---------------------------------------------------------------------------

class _WaferAugment:
    """
    Contrastive augmentation for one-hot encoded wafer maps.

    Returns a randomly augmented view of the input (3, H, W) tensor.
    Designed to produce diverse views that share global defect structure
    but differ in orientation, scale, and sharpness — forcing the backbone
    to learn the structural signature rather than memorising pixel positions.
    """

    def __init__(self, input_size: int = 224, crop_min: float = 0.85,
                 blur_prob: float = 0.5) -> None:
        self.input_size = input_size
        self.crop_min = crop_min
        self.blur_prob = blur_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # D4: random 90° rotation
        k = random.randint(0, 3)
        if k:
            tensor = torch.rot90(tensor, k, dims=[1, 2])

        # D4: horizontal flip
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])

        # D4: vertical flip
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[1])

        # Random crop (85–100% of image) → resize back
        # crop_min=0.85 avoids cutting ring/edge patterns that extend to the border
        if random.random() < 0.8:
            H, W = tensor.shape[1], tensor.shape[2]
            scale = random.uniform(self.crop_min, 1.0)
            ch = max(1, int(H * scale))
            cw = max(1, int(W * scale))
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            tensor = tensor[:, top : top + ch, left : left + cw]
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)

        # Gaussian blur — mild spatial smoothing
        if random.random() < self.blur_prob:
            sigma = random.uniform(0.5, 2.0)
            tensor = _gaussian_blur(tensor, sigma)

        return tensor


def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply per-channel Gaussian blur to a (C, H, W) float tensor."""
    ks = max(3, 2 * int(math.ceil(2 * sigma)) + 1)
    x = torch.arange(ks, dtype=torch.float32, device=tensor.device) - ks // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)
    C = tensor.shape[0]
    kernel = kernel.expand(C, 1, ks, ks)
    return F.conv2d(
        tensor.unsqueeze(0), kernel, padding=ks // 2, groups=C
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset: returns two independent augmented views per map
# ---------------------------------------------------------------------------

class ContrastivePairDataset(Dataset):
    """
    Wraps the unlabeled WM-811K maps and returns two independently augmented
    views per sample for NT-Xent contrastive training.

    Each call to __getitem__ applies the augmentation pipeline twice with
    independent random seeds — the two views will differ in rotation, crop,
    and blur but share the global defect structure.
    """

    def __init__(self, maps: list, input_size: int = 224,
                 crop_min: float = 0.85, blur_prob: float = 0.5) -> None:
        self.maps = maps
        self.input_size = input_size
        self.aug = _WaferAugment(input_size, crop_min, blur_prob)

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base = resize_map(encode_map(self.maps[idx]), self.input_size)
        return self.aug(base.clone()), self.aug(base.clone())


# ---------------------------------------------------------------------------
# SimCLR model: backbone + projection head
# ---------------------------------------------------------------------------

class _ProjectionHead(nn.Module):
    """
    2-layer MLP projection head (SimCLR convention).

    Projecting to a lower-dimensional space with a non-linear head before
    computing the contrastive loss is a key SimCLR finding: the representation
    h (before the head) transfers better to downstream tasks than z (after).
    After pretraining, we save h (backbone outputs) and discard this head.

    BatchNorm between layers stabilises training in large-batch regimes
    (batch_size=256 with 638k maps).
    """

    def __init__(self, in_dim: int = 512, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRNet(nn.Module):
    """ResNet-18 backbone with projection head for contrastive pretraining."""

    def __init__(self, backbone: nn.Module, proj_hidden: int = 512,
                 proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        in_dim = 512 if "resnet18" in type(backbone).__name__.lower() else 2048
        # Detect in_dim from the FC layer we replaced with Identity
        # (the backbone's original fc.in_features was saved before replacement)
        if hasattr(backbone, "_ssl_feature_dim"):
            in_dim = backbone._ssl_feature_dim
        self.projector = _ProjectionHead(in_dim, proj_hidden, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)    # (B, in_dim)
        return self.projector(h)  # (B, proj_dim)


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy loss (Chen et al. 2020).

    Given batch of B samples with 2 views each:
    - Concatenate z1 and z2 → z of shape (2B, D)
    - For each anchor i, the positive is at index i+B (or i-B)
    - All other 2B-2 samples in the batch are negatives
    - Loss = -log(sim(i, positive) / sum_negatives)

    Args:
        z1: (B, D) — projection vectors for view 1 (NOT pre-normalized)
        z2: (B, D) — projection vectors for view 2 (NOT pre-normalized)
        temperature: scaling factor; lower = sharper distribution over negatives

    Returns:
        scalar mean loss over all 2B anchors
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature             # (2B, 2B)

    # Mask the diagonal (self-similarity = trivial positive, not meaningful)
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag_mask, -float("inf"))

    # For anchor i in [0, B): positive is at i+B
    # For anchor i in [B, 2B): positive is at i-B
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------

def pretrain(
    data_root: Path,
    output_dir: Path,
    arch: str = "resnet18",
    input_size: int = 224,
    device: str = "auto",
    num_workers: int = 4,
    batch_size: int = 256,
    epochs: int = 200,
    temperature: float = 0.07,
    proj_dim: int = 128,
    proj_hidden: int = 512,
    crop_min: float = 0.85,
    blur_prob: float = 0.5,
    seed: int = 42,
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load unlabeled maps ---
    pkl_path = Path(data_root) / "LSWMD.pkl"
    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    unlabeled_mask = ~df["failureType"].apply(_is_labeled)
    maps = df[unlabeled_mask]["waferMap"].tolist()
    print(f"Unlabeled maps: {len(maps):,}  (labeled: {(~unlabeled_mask).sum():,})")

    ds = ContrastivePairDataset(maps, input_size, crop_min, blur_prob)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,   # NT-Xent assumes fixed batch size
    )

    # --- build backbone (no CBAM, no classification head) ---
    # Reuse build_model for architecture consistency; strip the FC head.
    cfg_stub = _WaferConfig(
        data_root=data_root,
        output_dir=output_dir,
        device=device,
        arch=arch,
        cbam=False,         # pretraining without CBAM — task-agnostic features
        pretrained=False,   # train from scratch on wafer domain
    )
    backbone = build_model(cfg_stub, num_classes=1)
    feature_dim = backbone.fc.in_features   # 512 for ResNet-18
    backbone._ssl_feature_dim = feature_dim
    backbone.fc = nn.Identity()

    model = SimCLRNet(backbone, proj_hidden, proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    print(f"\nSimCLR pretraining: {len(ds):,} maps | batch={batch_size} | "
          f"epochs={epochs} | T={temperature} | proj_dim={proj_dim}")
    print(f"Device: {device}  |  arch: {arch}  |  feature_dim: {feature_dim}\n")
    print(f"{'Epoch':>6}  {'NT-Xent loss':>14}  {'LR':>10}")
    print("-" * 36)

    best_loss = float("inf")
    ckpt_path = output_dir / "pretrained_backbone.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for view1, view2 in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=device_type == "cuda"):
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:6d}  {avg_loss:14.4f}  {lr:10.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "projector_state_dict": model.projector.state_dict(),
                    "loss": avg_loss,
                    "arch": arch,
                    "feature_dim": feature_dim,
                },
                ckpt_path,
            )

    print(f"\nPretraining complete. Best NT-Xent loss: {best_loss:.4f}")
    print(f"Backbone saved → {ckpt_path}")
    print(
        "\nNext steps:"
        "\n  1. Copy outputs/pretrained_backbone.pt to wafer-defect-classifier/outputs/"
        "\n  2. In wafer-defect-classifier/configs/baseline.yaml set:"
        "\n       backbone_ckpt_path: outputs/pretrained_backbone.pt"
        "\n       loss: focal"
        "\n       cbam: true"
        "\n       num_epochs: 40"
        "\n       patience: 10"
        "\n  3. Run: python -m wafer.train && python -m wafer.calibrate && python -m wafer.evaluate"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wafer-ssl: SimCLR backbone pretraining")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "pretrain.yaml",
        help="YAML config for pretraining (default: configs/pretrain.yaml)",
    )
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None, dest="batch_size")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--proj-dim",    type=int,   default=None, dest="proj_dim")
    parser.add_argument("--data-root",   type=Path,  default=None, dest="data_root")
    parser.add_argument("--output-dir",  type=Path,  default=None, dest="output_dir")
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI args override YAML
    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    cfg.update(overrides)

    pretrain(
        data_root=Path(cfg.get("data_root", "data/raw")),
        output_dir=Path(cfg.get("output_dir", "outputs")),
        arch=cfg.get("arch", "resnet18"),
        input_size=int(cfg.get("input_size", 224)),
        device=cfg.get("device", "auto"),
        num_workers=int(cfg.get("num_workers", 4)),
        batch_size=int(cfg.get("batch_size", 256)),
        epochs=int(cfg.get("pretrain_epochs", 200)),
        temperature=float(cfg.get("temperature", 0.07)),
        proj_dim=int(cfg.get("proj_dim", 128)),
        proj_hidden=int(cfg.get("proj_hidden", 512)),
        crop_min=float(cfg.get("crop_min", 0.85)),
        blur_prob=float(cfg.get("blur_prob", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
