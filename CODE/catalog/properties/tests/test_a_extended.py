"""Tests pour famille A étendue — A3, A4, A5, A6."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_a_spectral.a3_condition_number import (
    A3ConditionNumber,
)
from catalog.properties.family_a_spectral.a4_spectral_entropy import (
    A4SpectralEntropy,
)
from catalog.properties.family_a_spectral.a5_spectral_decay import (
    A5SpectralDecay,
)
from catalog.properties.family_a_spectral.a6_participation_ratio import (
    A6ParticipationRatio,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# A3 ConditionNumber
# -----------------------------------------------------------------------------


def test_a3_identity_has_kappa_one() -> None:
    """I_n a tous σ = 1 → κ = 1, log10 κ = 0."""
    A = torch.eye(8, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A3ConditionNumber().compute(A, ctx)
    assert abs(out["log10_kappa_median"]) < 1e-10
    assert out["fraction_singular"] == 0.0


def test_a3_singular_matrix_flagged() -> None:
    """A rang ≤ N → σ_min ≈ 0 → fraction_singular > 0."""
    n = 8
    M = torch.randn(n, 3, dtype=torch.float64)
    A = (M @ M.T).unsqueeze(0).unsqueeze(0)  # rank 3
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A3ConditionNumber().compute(A, ctx)
    assert out["fraction_singular"] == 1.0


def test_a3_registered() -> None:
    assert REGISTRY.get("A3_condition_number") is A3ConditionNumber


# -----------------------------------------------------------------------------
# A4 SpectralEntropy
# -----------------------------------------------------------------------------


def test_a4_identity_max_entropy() -> None:
    """I_n : σ uniformes → H_spec = log K, H_norm = 1."""
    A = torch.eye(8, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A4SpectralEntropy().compute(A, ctx)
    assert abs(out["spectral_entropy_norm_median"] - 1.0) < 1e-6


def test_a4_rank_1_zero_entropy() -> None:
    """A rank-1 (uv^T) : un seul σ non nul → H_spec ≈ 0."""
    n = 8
    u = torch.randn(n, 1, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    A = (u @ v.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A4SpectralEntropy().compute(A, ctx)
    assert out["spectral_entropy_norm_median"] < 1e-3


def test_a4_registered() -> None:
    assert REGISTRY.get("A4_spectral_entropy") is A4SpectralEntropy


# -----------------------------------------------------------------------------
# A5 SpectralDecay
# -----------------------------------------------------------------------------


def test_a5_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A5SpectralDecay().compute(A, ctx)
    assert "decay_alpha_median" in out
    assert "decay_r_squared_median" in out


def test_a5_steep_decay_high_alpha() -> None:
    """Spectre rapidement décroissant (rank-1+bruit) → α élevé."""
    n = 32
    # Construire matrice avec σ_i ∝ 1/(i+1)^2 (décroissance rapide)
    U, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    sigma = 1.0 / torch.arange(1, n + 1, dtype=torch.float64).pow(2)
    A = (U @ torch.diag(sigma) @ V.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A5SpectralDecay(k_fit_max=16).compute(A, ctx)
    # alpha théorique = 2 (puisque σ_i = 1/i² → log σ = -2 log i)
    assert abs(out["decay_alpha_median"] - 2.0) < 0.1
    assert out["decay_r_squared_median"] > 0.99


def test_a5_flat_spectrum_alpha_near_zero() -> None:
    """Identité (σ uniformes) → α ≈ 0."""
    n = 16
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A5SpectralDecay(k_fit_max=8).compute(A, ctx)
    assert abs(out["decay_alpha_median"]) < 1e-6


def test_a5_rejects_invalid_k_fit() -> None:
    import pytest

    with pytest.raises(ValueError, match="k_fit_max"):
        A5SpectralDecay(k_fit_max=2)


def test_a5_registered() -> None:
    assert REGISTRY.get("A5_spectral_decay") is A5SpectralDecay


# -----------------------------------------------------------------------------
# A6 ParticipationRatio
# -----------------------------------------------------------------------------


def test_a6_identity_pr_equals_k() -> None:
    """I_n : tous σ égaux à 1 → PR = K (uniforme)."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A6ParticipationRatio().compute(A, ctx)
    assert abs(out["participation_ratio_median"] - n) < 1e-6
    assert abs(out["participation_ratio_norm_median"] - 1.0) < 1e-6


def test_a6_rank_1_pr_one() -> None:
    """A rank-1 : PR = 1 (un seul mode)."""
    n = 8
    u = torch.randn(n, 1, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    A = (u @ v.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = A6ParticipationRatio().compute(A, ctx)
    assert abs(out["participation_ratio_median"] - 1.0) < 1e-6
    assert out["fraction_pr_below_3"] == 1.0


def test_a6_registered() -> None:
    assert REGISTRY.get("A6_participation_ratio") is A6ParticipationRatio
