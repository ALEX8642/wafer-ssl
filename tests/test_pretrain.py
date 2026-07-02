"""Smoke tests for wafer_ssl.pretrain — no GPU or LSWMD.pkl required."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wafer_ssl.pretrain import (
    ContrastivePairDataset,
    SimCLRNet,
    _WaferAugment,
    _gaussian_blur,
    nt_xent_loss,
)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def test_nt_xent_single_pair_is_zero():
    # B=1: after masking self-similarity, the positive is the only candidate,
    # so cross-entropy is exactly 0 regardless of the embedding values.
    z = torch.randn(1, 8)
    loss = nt_xent_loss(z, z.clone())
    assert loss.item() == 0.0


def test_nt_xent_orthogonal_embeddings_match_uniform():
    # All 2B embeddings mutually orthogonal → every similarity is 0 → softmax
    # is uniform over the 2B-1 candidates → loss = ln(2B-1).
    eye = torch.eye(8)
    z1, z2 = eye[:4], eye[4:]
    loss = nt_xent_loss(z1, z2, temperature=0.07)
    assert torch.isclose(loss, torch.log(torch.tensor(7.0)), atol=1e-5)


def test_nt_xent_identical_views_near_zero():
    # Perfectly aligned positives dominate random negatives at T=0.07.
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    loss = nt_xent_loss(z, z.clone(), temperature=0.07)
    assert loss.item() < 0.01


def test_nt_xent_mismatched_views_higher_than_matched():
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = z1 + 0.01 * torch.randn_like(z1)          # matched views
    z2_shuf = z2[torch.randperm(8)]                # positives scrambled
    assert nt_xent_loss(z1, z2_shuf) > nt_xent_loss(z1, z2)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _random_one_hot(h: int = 32, w: int = 32) -> torch.Tensor:
    idx = torch.randint(0, 3, (h, w))
    return F.one_hot(idx, num_classes=3).permute(2, 0, 1).float()


def test_augment_preserves_one_hot_encoding():
    # With blur disabled, every op (D4, nearest-resize crop, cutout) must keep
    # the tensor a valid one-hot map: binary values, channels summing to 1.
    torch.manual_seed(0)
    import random as _random
    _random.seed(0)
    aug = _WaferAugment(input_size=32, crop_min=0.2, blur_prob=0.0)
    for _ in range(10):
        out = aug(_random_one_hot())
        assert out.shape == (3, 32, 32)
        assert torch.all((out == 0) | (out == 1))
        assert torch.allclose(out.sum(dim=0), torch.ones(32, 32))


def test_gaussian_blur_normalized_kernel_and_shape():
    ones = torch.ones(3, 16, 16)
    out = _gaussian_blur(ones, sigma=1.0)
    assert out.shape == ones.shape
    # Normalized kernel → interior of a constant map stays constant
    # (borders lose mass to zero padding).
    assert torch.allclose(out[:, 8, 8], torch.ones(3), atol=1e-5)


def test_contrastive_pair_dataset_returns_two_views():
    rng = np.random.default_rng(0)
    maps = [rng.integers(0, 3, size=(26, 26)) for _ in range(3)]
    ds = ContrastivePairDataset(maps, input_size=64, crop_min=0.2, blur_prob=0.0)
    assert len(ds) == 3
    v1, v2 = ds[0]
    assert v1.shape == (3, 64, 64)
    assert v2.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# SimCLR net
# ---------------------------------------------------------------------------

class _TinyBackbone(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_simclr_net_output_shape():
    backbone = _TinyBackbone(dim=32)
    backbone._ssl_feature_dim = 32
    net = SimCLRNet(backbone, proj_hidden=16, proj_dim=8).eval()
    out = net(torch.randn(4, 3, 16, 16))
    assert out.shape == (4, 8)
