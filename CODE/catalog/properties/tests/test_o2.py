"""Tests pour O2 CauchyDisplacementRank."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_o_displacement.o2_cauchy_displacement_rank import (
    O2CauchyDisplacementRank,
    _equispace_xy,
)
from catalog.properties.registry import REGISTRY


def test_o2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = O2CauchyDisplacementRank().compute(A, ctx)
    assert "best_r_eff_median" in out
    assert "best_rank_strict_median" in out
    assert "fraction_best_rank_le_1_strict" in out
    assert out["n_matrices"] == 6


def test_o2_cauchy_matrix_has_rank_1() -> None:
    """C[i,j] = 1/(y_i − x_j) avec x, y equispace : rang(D_y C − C D_x) = 1."""
    n = 8
    offset = 0.5
    x, y = _equispace_xy(n, offset=offset, device="cpu", dtype=torch.float64)
    # Construit la matrice de Cauchy exacte pour cette paire
    C = torch.empty(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            C[i, j] = 1.0 / (y[i] - x[j])
    A = C.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    # Force la paire (x, y) qui matche la construction
    out = O2CauchyDisplacementRank(offsets=(offset,), include_chebyshev=False, rank_atol=1e-8).compute(A, ctx)
    # Min rang strict doit être 1
    assert out["best_rank_strict_median"] == 1.0
    assert out["fraction_best_rank_le_1_strict"] == 1.0


def test_o2_random_matrix_has_high_rank() -> None:
    """Matrice aléatoire générique → rang ≈ N pour toute paire (x, y)."""
    torch.manual_seed(0)
    A = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = O2CauchyDisplacementRank().compute(A, ctx)
    # Le min rang sur la grille doit rester élevé (proche de N)
    assert out["best_rank_strict_median"] >= 8


def test_o2_registered() -> None:
    assert REGISTRY.get("O2_cauchy_displacement_rank") is O2CauchyDisplacementRank
