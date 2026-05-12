"""Tests pour B4 SparseFraction et B7 TropicalDegeneracy."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_b_structural.b4_sparse_fraction import (
    B4SparseFraction,
)
from catalog.properties.family_b_structural.b7_tropical_degeneracy import (
    B7TropicalDegeneracy,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# B4 SparseFraction
# -----------------------------------------------------------------------------


def test_b4_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B4SparseFraction()
    out = prop.compute(A, ctx)
    # 3 epsilons × 2 stats + top-k × 3 × 2 + gini ×4 + n_matrices
    assert "sparse_frac_eps_0p01_median" in out
    assert "top1_concentration_median" in out
    assert "gini_median" in out
    assert out["n_matrices"] == 6


def test_b4_one_hot_matrix_has_sparse_frac_near_one() -> None:
    """Matrice one-hot par ligne : sparsité maximale, gini ≈ 1."""
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for i in range(n):
        A[0, 0, i, (i + 3) % n] = 1.0  # une seule entrée par ligne
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B4SparseFraction().compute(A, ctx)
    # n entrées non-zéro sur n² → fraction zeros = (n²-n)/n² = (n-1)/n ≈ 0.875 pour n=8
    assert out["sparse_frac_eps_0p01_median"] > 0.85
    # Gini quasi 1 sur distribution Pareto extrême
    assert out["gini_median"] > 0.85
    # top-1 capture toute la masse (n=8 lignes, mais on flatten N² → top-1 = 1/n)
    # Σ all = n, top-1 = 1 → top-1 conc = 1/n = 0.125
    assert abs(out["top1_concentration_median"] - 1.0 / n) < 0.01


def test_b4_uniform_matrix_has_zero_gini() -> None:
    """Matrice uniforme |A| constant : sparsité 0, gini 0."""
    A = torch.ones(1, 1, 16, 16, dtype=torch.float64) / 16.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B4SparseFraction().compute(A, ctx)
    # sparse_frac : aucune entrée < ε × max (max == toutes) → 0
    assert out["sparse_frac_eps_0p10_median"] == 0.0
    assert abs(out["gini_median"]) < 1e-6


def test_b4_registered() -> None:
    assert REGISTRY.get("B4_sparse_fraction") is B4SparseFraction


# -----------------------------------------------------------------------------
# B7 TropicalDegeneracy
# -----------------------------------------------------------------------------


def test_b7_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B7TropicalDegeneracy()
    out = prop.compute(A, ctx)
    assert "log_gap_top1_top2_median" in out
    assert "tropical_rank_proxy_delta_1p0_median" in out
    assert "argmax_unique_fraction" in out
    assert "top1_softmax_mass_median" in out


def test_b7_peaked_softmax_gives_large_log_gap() -> None:
    """Softmax piqué (logits écartés) → log_gap ≫ 1."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16, dtype=torch.float64) * 10  # piqué
    A = torch.softmax(logits, dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B7TropicalDegeneracy().compute(A, ctx)
    # log gap top1 ≫ top2 quand logits ont gros écart
    assert out["log_gap_top1_top2_median"] > 2.0
    # top1 mass capture la quasi-totalité
    assert out["top1_softmax_mass_median"] > 0.7


def test_b7_uniform_softmax_gives_zero_log_gap() -> None:
    """Softmax uniforme (logits égaux) → log_gap ≈ 0."""
    A = torch.ones(1, 1, 16, 16, dtype=torch.float64) / 16.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = B7TropicalDegeneracy().compute(A, ctx)
    assert abs(out["log_gap_top1_top2_median"]) < 1e-6
    # Top-1 mass = 1/n
    assert abs(out["top1_softmax_mass_median"] - 1.0 / 16.0) < 1e-6


def test_b7_registered() -> None:
    assert REGISTRY.get("B7_tropical_degeneracy") is B7TropicalDegeneracy
