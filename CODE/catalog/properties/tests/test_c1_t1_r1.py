"""Tests C1 + T1 + R1."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_c_token_stats.c1_kl_baseline import C1KLBaseline
from catalog.properties.family_r_rkhs.r1_mercer_psd import R1MercerPSD
from catalog.properties.family_t_equivariance.t1_permutation_invariance import (
    T1PermutationInvariance,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# C1 KLBaseline
# -----------------------------------------------------------------------------


def test_c1_constant_attention_zero_kl() -> None:
    """Toutes les lignes identiques → baseline = ligne → KL=0."""
    n = 8
    row = torch.softmax(torch.randn(n, dtype=torch.float64), dim=-1)
    A = row.unsqueeze(0).expand(n, n).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C1KLBaseline().compute(A, ctx)
    assert out["kl_baseline_median"] < 1e-10


def test_c1_diverse_rows_nonzero_kl() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C1KLBaseline().compute(A, ctx)
    assert out["kl_baseline_mean"] > 0.0


def test_c1_registered() -> None:
    assert REGISTRY.get("C1_kl_baseline") is C1KLBaseline


# -----------------------------------------------------------------------------
# T1 PermutationInvariance
# -----------------------------------------------------------------------------


def test_t1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = T1PermutationInvariance(n_permutations=4).compute(A, ctx)
    assert "epsilon_perm_median" in out
    assert out["n_permutations"] == 4


def test_t1_constant_matrix_zero_eps() -> None:
    """A = constant partout → π A π^T = A → ε = 0 pour toute π."""
    A = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = T1PermutationInvariance(n_permutations=4).compute(A, ctx)
    assert out["epsilon_perm_median"] < 1e-10


def test_t1_random_matrix_positive_eps() -> None:
    torch.manual_seed(0)
    A = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = T1PermutationInvariance(n_permutations=4).compute(A, ctx)
    assert out["epsilon_perm_median"] > 0.5


def test_t1_rejects_invalid_n_perm() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_permutations"):
        T1PermutationInvariance(n_permutations=0)


def test_t1_registered() -> None:
    assert REGISTRY.get("T1_permutation_invariance") is T1PermutationInvariance


# -----------------------------------------------------------------------------
# R1 MercerPSD
# -----------------------------------------------------------------------------


def test_r1_psd_matrix_passes() -> None:
    """A = M M^T → PSD strict."""
    n = 8
    M = torch.randn(n, n, dtype=torch.float64)
    A_psd = (M @ M.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = R1MercerPSD().compute(A_psd, ctx)
    assert out["psd_fraction"] == 1.0
    assert out["min_eigenvalue_median"] >= -1e-10


def test_r1_non_psd_matrix_flagged() -> None:
    """Matrice antisymétrique → eigvalsh sur (A+A^T)/2 = 0 → PSD vacuously."""
    # Pour un cas vraiment non-PSD, construire A_sym avec eigvals négatives
    n = 8
    M = torch.randn(n, n, dtype=torch.float64)
    A_sym = (M + M.T) / 2  # symétrique mais pas forcément PSD
    # Force au moins une valeur propre négative en soustrayant identité * λ_min + ε
    eigs = torch.linalg.eigvalsh(A_sym)
    if eigs.min() > 0:
        A_sym = A_sym - (eigs.min() + 0.5) * torch.eye(n, dtype=torch.float64)
    A = A_sym.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = R1MercerPSD().compute(A, ctx)
    assert out["psd_fraction"] == 0.0
    assert out["min_eigenvalue_median"] < -0.1


def test_r1_registered() -> None:
    assert REGISTRY.get("R1_mercer_psd") is R1MercerPSD
