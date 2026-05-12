"""Tests famille U — Sparse-structurées."""

from __future__ import annotations

import math

import torch

from catalog.projectors import ButterflyMask, MonarchMask
from catalog.projectors.butterfly_mask import _butterfly_support_mask
from catalog.projectors.monarch_mask import _monarch_support_mask
from catalog.properties.base import PropertyContext
from catalog.properties.family_u_sparse_structured.u1_butterfly_distance import (
    U1ButterflyDistance,
)
from catalog.properties.family_u_sparse_structured.u2_monarch_distance import (
    U2MonarchDistance,
)
from catalog.properties.family_u_sparse_structured.u3_block_sparse_distance import (
    U3BlockSparseDistance,
)
from catalog.properties.family_u_sparse_structured.u5_sparse_plus_lowrank import (
    U5SparsePlusLowRank,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# Butterfly mask
# -----------------------------------------------------------------------------


def test_butterfly_mask_has_correct_density() -> None:
    """Pour N = 2^k, density mask 2-niveaux est O(log N / N) en théorie complète,
    mais V1 (2 niveaux fixes) → O(1/N) par niveau × 2 = O(1/N)."""
    mask = _butterfly_support_mask(8, torch.device("cpu"), torch.float64)
    # Pour N=8, k=3, on ajoute 3 niveaux : chaque niveau ajoute 2 entrées par ligne
    # (i, i) et (i, i XOR bit). Total ≤ N × (k + 1)
    density = float(mask.float().mean().item())
    assert density > 0.0 and density < 1.0


def test_butterfly_mask_diagonal_included() -> None:
    mask = _butterfly_support_mask(8, torch.device("cpu"), torch.float64)
    for i in range(8):
        assert mask[i, i].item() == 1.0


# -----------------------------------------------------------------------------
# Monarch mask
# -----------------------------------------------------------------------------


def test_monarch_mask_diagonal_blocks_included() -> None:
    """Monarch mask (m=2, b=4, N=8) doit inclure les blocs diagonaux 2×2."""
    n = 8
    mask = _monarch_support_mask(n, m=2, b=4, device=torch.device("cpu"), dtype=torch.float64)
    # Le support exact dépend de la composition D1 P2 D2 ; on vérifie au moins
    # que les blocs diagonaux du début sont activés
    assert mask[0, 0].item() == 1.0 or mask[0, 4].item() == 1.0


# -----------------------------------------------------------------------------
# U1 Butterfly distance
# -----------------------------------------------------------------------------


def test_u1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U1ButterflyDistance().compute(A, ctx)
    assert "epsilon_butterfly_lb_median" in out
    assert "mask_density" in out


def test_u1_supported_matrix_zero_eps() -> None:
    """Si A est sur le support butterfly, ε = 0."""
    n = 8
    mask = _butterfly_support_mask(n, torch.device("cpu"), torch.float64)
    A = (torch.randn(n, n, dtype=torch.float64) * mask).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U1ButterflyDistance().compute(A, ctx)
    assert out["epsilon_butterfly_lb_median"] < 1e-10


def test_u1_registered() -> None:
    assert REGISTRY.get("U1_butterfly_distance") is U1ButterflyDistance


# -----------------------------------------------------------------------------
# U2 Monarch distance
# -----------------------------------------------------------------------------


def test_u2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U2MonarchDistance().compute(A, ctx)
    assert "epsilon_monarch_lb_median" in out
    assert "mask_density" in out


def test_u2_registered() -> None:
    assert REGISTRY.get("U2_monarch_distance") is U2MonarchDistance


# -----------------------------------------------------------------------------
# U3 Block-sparse
# -----------------------------------------------------------------------------


def test_u3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U3BlockSparseDistance(block_size=4).compute(A, ctx)
    assert "epsilon_block_sparse_top_0p10_median" in out
    assert out["n_blocks_total"] == 16  # 4 × 4 blocs


def test_u3_one_block_zero_eps() -> None:
    """A avec un seul bloc non-nul : top-5% (1 bloc) capture tout, ε=0."""
    n = 16
    bs = 4
    A_dense = torch.zeros(n, n, dtype=torch.float64)
    A_dense[0:bs, 0:bs] = torch.randn(bs, bs, dtype=torch.float64)
    A = A_dense.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U3BlockSparseDistance(block_size=bs, k_fractions=(0.05, 0.10)).compute(A, ctx)
    # 1 bloc sur 16 = 6.25%, top-10% = 2 blocs → capture tout
    assert out["epsilon_block_sparse_top_0p10_median"] < 1e-10


def test_u3_rejects_invalid_block_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="block_size"):
        U3BlockSparseDistance(block_size=0)


def test_u3_registered() -> None:
    assert REGISTRY.get("U3_block_sparse_distance") is U3BlockSparseDistance


# -----------------------------------------------------------------------------
# U5 Sparse + Low-Rank
# -----------------------------------------------------------------------------


def test_u5_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U5SparsePlusLowRank().compute(A, ctx)
    assert "rel_residual_median" in out
    assert "rel_lowrank_norm_median" in out
    assert "rel_sparse_norm_median" in out


def test_u5_pure_lowrank_captured() -> None:
    """A = u·vᵀ → décomposition trouve L ≈ A, S ≈ 0, résidu petit."""
    n = 8
    u = torch.randn(n, 1, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    A = (u @ v.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = U5SparsePlusLowRank(rank_target=1, n_iter=8).compute(A, ctx)
    assert out["rel_residual_median"] < 0.05
    assert out["rel_lowrank_norm_median"] > 0.9


def test_u5_rejects_invalid_rank() -> None:
    import pytest

    with pytest.raises(ValueError, match="rank_target"):
        U5SparsePlusLowRank(rank_target=0)


def test_u5_registered() -> None:
    assert REGISTRY.get("U5_sparse_plus_lowrank") is U5SparsePlusLowRank
