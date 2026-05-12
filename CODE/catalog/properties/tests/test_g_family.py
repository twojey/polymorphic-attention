"""Tests pour famille G — G1, G2, G3."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_g_algebraic.g1_trace_det import G1TraceDet
from catalog.properties.family_g_algebraic.g2_symmetry import G2Symmetry
from catalog.properties.family_g_algebraic.g3_idempotence import G3Idempotence
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# G1 TraceDet
# -----------------------------------------------------------------------------


def test_g1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G1TraceDet().compute(A, ctx)
    assert "trace_median" in out
    assert "logabsdet_median" in out
    assert "eig_max_abs_median" in out
    assert out["n_matrices"] == 6


def test_g1_identity_matrix_invariants() -> None:
    """I_n : tr = n, det = 1, eig = 1 partout."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G1TraceDet().compute(A, ctx)
    assert abs(out["trace_median"] - n) < 1e-10
    assert abs(out["logabsdet_median"] - 0.0) < 1e-10  # log(1) = 0
    assert abs(out["eig_max_abs_median"] - 1.0) < 1e-10


def test_g1_softmax_trace_in_range() -> None:
    """Softmax rows somment à 1 → tr ∈ [0, N]."""
    n = 16
    A = torch.softmax(torch.randn(1, 1, n, n, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G1TraceDet().compute(A, ctx)
    assert 0.0 <= out["trace_median"] <= float(n)


def test_g1_registered() -> None:
    assert REGISTRY.get("G1_trace_det") is G1TraceDet


# -----------------------------------------------------------------------------
# G2 Symmetry
# -----------------------------------------------------------------------------


def test_g2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G2Symmetry().compute(A, ctx)
    assert "sym_fraction_median" in out
    assert "anti_fraction_median" in out
    assert "epsilon_asymmetry_median" in out


def test_g2_symmetric_matrix_has_anti_fraction_zero() -> None:
    """A symétrique → anti_fraction = 0, sym_fraction = 1."""
    M = torch.randn(8, 8, dtype=torch.float64)
    A_sym = (M + M.T) / 2
    A = A_sym.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G2Symmetry().compute(A, ctx)
    assert out["sym_fraction_median"] > 0.999
    assert out["anti_fraction_median"] < 1e-10
    assert out["epsilon_asymmetry_median"] < 1e-10


def test_g2_antisymmetric_matrix_has_sym_fraction_zero() -> None:
    M = torch.randn(8, 8, dtype=torch.float64)
    A_anti = (M - M.T) / 2
    A = A_anti.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G2Symmetry().compute(A, ctx)
    assert out["anti_fraction_median"] > 0.999
    assert out["sym_fraction_median"] < 1e-10


def test_g2_sum_fractions_equals_one() -> None:
    """Décomposition orthogonale : sym + anti = 1."""
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G2Symmetry().compute(A, ctx)
    s = out["sym_fraction_median"] + out["anti_fraction_median"]
    # Median of sum != sum of medians en général, mais sur 6 matrices très similaires
    # c'est typiquement vrai à 1e-2 près. Test plus solide : mean (additive).
    s_mean = out["sym_fraction_mean"] + out["anti_fraction_mean"]
    # FP32 cast en sortie → tol ~1e-7
    assert abs(s_mean - 1.0) < 1e-6


def test_g2_registered() -> None:
    assert REGISTRY.get("G2_symmetry") is G2Symmetry


# -----------------------------------------------------------------------------
# G3 Idempotence
# -----------------------------------------------------------------------------


def test_g3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G3Idempotence(max_power=4).compute(A, ctx)
    assert "epsilon_idempotence_median" in out
    assert "power_diff_k2_median" in out
    assert "power_diff_k3_median" in out


def test_g3_identity_is_idempotent() -> None:
    """I² = I → ε = 0."""
    A = torch.eye(8, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G3Idempotence().compute(A, ctx)
    assert out["epsilon_idempotence_median"] < 1e-10
    assert out["power_diff_k2_median"] < 1e-10


def test_g3_orthogonal_projector_idempotent() -> None:
    """Projecteur orthogonal P = U U^T (U orthonormal) → P² = P."""
    n = 8
    k = 3
    M = torch.randn(n, k, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    P = Q @ Q.T
    A = P.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G3Idempotence().compute(A, ctx)
    assert out["epsilon_idempotence_median"] < 1e-10


def test_g3_random_softmax_not_idempotent() -> None:
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = G3Idempotence().compute(A, ctx)
    assert out["epsilon_idempotence_median"] > 0.1


def test_g3_registered() -> None:
    assert REGISTRY.get("G3_idempotence") is G3Idempotence
