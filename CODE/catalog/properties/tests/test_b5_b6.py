"""Tests pour B5 BlockDiagonal-distance et B6 Banded-distance."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_b_structural.b5_block_diagonal_distance import (
    B5BlockDiagonalDistance,
)
from catalog.properties.family_b_structural.b6_banded_distance import (
    B6BandedDistance,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# B5 Block-diagonal
# -----------------------------------------------------------------------------


def test_b5_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B5BlockDiagonalDistance()
    out = prop.compute(A, ctx)
    assert "epsilon_best" in out
    assert "block_size_best" in out
    assert out["block_size_best"] >= 1


def test_b5_pure_block_diagonal_matrix_gives_zero_eps() -> None:
    """Si A est exactement block-diagonale avec bs=4, ε ≈ 0 pour bs=4."""
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    A[0, 0, 0:4, 0:4] = torch.eye(4)
    A[0, 0, 4:8, 4:8] = torch.eye(4) * 2
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B5BlockDiagonalDistance(block_sizes=[4])
    out = prop.compute(A, ctx)
    assert out["epsilon_best"] < 1e-10
    assert out["block_size_best"] == 4


def test_b5_registered() -> None:
    assert REGISTRY.get("B5_block_diagonal_distance") is B5BlockDiagonalDistance


# -----------------------------------------------------------------------------
# B6 Banded
# -----------------------------------------------------------------------------


def test_b6_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B6BandedDistance()
    out = prop.compute(A, ctx)
    assert "epsilon_best" in out
    assert "bandwidth_best" in out
    assert out["bandwidth_best"] >= 0


def test_b6_pure_banded_matrix_gives_zero_eps() -> None:
    n = 8
    A_full = torch.randn(n, n, dtype=torch.float64)
    # Garde seulement la bande w=2
    i = torch.arange(n).view(-1, 1)
    j = torch.arange(n).view(1, -1)
    mask = (i - j).abs() <= 2
    A_banded = torch.where(mask, A_full, torch.zeros_like(A_full))
    A = A_banded.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B6BandedDistance(bandwidths=[2])
    out = prop.compute(A, ctx)
    assert out["epsilon_best"] < 1e-10
    assert out["bandwidth_best"] == 2


def test_b6_random_softmax_high_eps_at_small_bandwidth() -> None:
    """Softmax aléatoire (non focalisée) a ε > 0.5 même au best bandwidth small."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 24, 24, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B6BandedDistance(bandwidths=[1, 2, 4])
    out = prop.compute(A, ctx)
    # Best bandwidth (≤ 4) ne capture pas tout, ε > 0.3
    assert out["epsilon_best"] > 0.3


def test_b6_registered() -> None:
    assert REGISTRY.get("B6_banded_distance") is B6BandedDistance
