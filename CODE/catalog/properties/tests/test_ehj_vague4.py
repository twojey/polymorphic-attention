"""Tests vague 4 — E1, E2, H2, H4, J2, J4."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_e_information.e1_mutual_info import E1MutualInfo
from catalog.properties.family_e_information.e2_compressibility import (
    E2Compressibility,
)
from catalog.properties.family_h_cross_layer.h2_layer_composition import (
    H2LayerComposition,
)
from catalog.properties.family_h_cross_layer.h4_layer_convergence import (
    H4LayerConvergence,
)
from catalog.properties.family_j_markov.j2_stationary_distribution import (
    J2StationaryDistribution,
)
from catalog.properties.family_j_markov.j4_reversibility import J4Reversibility
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# E1 MutualInfo
# -----------------------------------------------------------------------------


def test_e1_uniform_zero_mi() -> None:
    """A uniforme partout → distribs conditionnelles ≈ marginale → MI=0."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = E1MutualInfo().compute(A, ctx)
    assert abs(out["mutual_info_median"]) < 1e-10
    assert out["mutual_info_norm_median"] < 1e-10


def test_e1_identity_max_mi() -> None:
    """A = I_n : chaque query pointe sur 1 key unique → MI = log(N) (max)."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = E1MutualInfo().compute(A, ctx)
    assert abs(out["mutual_info_norm_median"] - 1.0) < 1e-3


def test_e1_registered() -> None:
    assert REGISTRY.get("E1_mutual_info") is E1MutualInfo


# -----------------------------------------------------------------------------
# E2 Compressibility
# -----------------------------------------------------------------------------


def test_e2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = E2Compressibility().compute(A, ctx)
    assert "compression_lowrank_median" in out
    assert "compression_quantized_entropy_median" in out
    assert "compression_combined_median" in out


def test_e2_rank_1_high_compressibility() -> None:
    """A rank-1 → r_eff/N petit → compression_lowrank ≈ 1/N."""
    n = 16
    u = torch.randn(n, 1, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    A = (u @ v.T).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = E2Compressibility().compute(A, ctx)
    assert out["compression_lowrank_median"] < 0.10


def test_e2_rejects_invalid_n_bins() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_bins"):
        E2Compressibility(n_bins=1)


def test_e2_registered() -> None:
    assert REGISTRY.get("E2_compressibility") is E2Compressibility


# -----------------------------------------------------------------------------
# H2 LayerComposition
# -----------------------------------------------------------------------------


def test_h2_basic_output_shape() -> None:
    layers = [
        torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
        for _ in range(3)
    ]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H2LayerComposition().compute(layers, ctx)
    assert "r_eff_product_global_median" in out
    assert "r_eff_product_layer_0_1_median" in out


def test_h2_identical_layers_product_idempotent() -> None:
    """Si A_ℓ = A_ℓ₊₁ = projecteur → A·A = A, cos = 1."""
    n = 8
    M = torch.randn(n, 3, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    P = Q @ Q.T  # projecteur orthogonal
    A = P.unsqueeze(0).unsqueeze(0)
    layers = [A.clone(), A.clone()]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H2LayerComposition().compute(layers, ctx)
    assert out["cos_prod_left_global_mean"] > 0.999


def test_h2_registered() -> None:
    assert REGISTRY.get("H2_layer_composition") is H2LayerComposition


# -----------------------------------------------------------------------------
# H4 LayerConvergence
# -----------------------------------------------------------------------------


def test_h4_basic_output_shape() -> None:
    layers = [
        torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
        for _ in range(4)
    ]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H4LayerConvergence().compute(layers, ctx)
    assert "plateau_layer" in out
    assert "eps_to_infinity_layer_0_median" in out
    assert "eps_to_infinity_layer_3_median" in out


def test_h4_identical_layers_plateau_immediate() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    layers = [A.clone() for _ in range(4)]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H4LayerConvergence().compute(layers, ctx)
    assert out["plateau_layer"] == 0
    assert out["plateau_reached"] is True


def test_h4_registered() -> None:
    assert REGISTRY.get("H4_layer_convergence") is H4LayerConvergence


# -----------------------------------------------------------------------------
# J2 StationaryDistribution
# -----------------------------------------------------------------------------


def test_j2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J2StationaryDistribution().compute(A, ctx)
    assert "lambda1_abs_median" in out
    assert "spectral_gap_median" in out
    assert "mixing_scale_median" in out


def test_j2_row_stochastic_lambda1_one() -> None:
    """Pour A row-stochastique : λ_1 = 1 exactement (Perron-Frobenius)."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J2StationaryDistribution().compute(A, ctx)
    assert abs(out["lambda1_abs_median"] - 1.0) < 1e-6


def test_j2_uniform_matrix_lambda2_zero() -> None:
    """A = 1/N partout → rank 1 → λ_2 = 0 → gap = 1."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J2StationaryDistribution().compute(A, ctx)
    assert out["lambda2_abs_median"] < 1e-6
    assert abs(out["spectral_gap_median"] - 1.0) < 1e-6


def test_j2_registered() -> None:
    assert REGISTRY.get("J2_stationary_distribution") is J2StationaryDistribution


# -----------------------------------------------------------------------------
# J4 Reversibility
# -----------------------------------------------------------------------------


def test_j4_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J4Reversibility().compute(A, ctx)
    assert "epsilon_detailed_balance_median" in out


def test_j4_symmetric_doubly_stochastic_reversible() -> None:
    """A symétrique row-stochastique → detailed balance satisfait (π uniforme)."""
    n = 8
    M = torch.exp(torch.randn(n, n, dtype=torch.float64))
    M = (M + M.T) / 2
    # Normaliser row stochastic ne préserve pas la symétrie en général,
    # mais pour une matrice symétrique doublement stochastique on est OK.
    # Pour le test on construit doublement stochastique via Sinkhorn léger :
    for _ in range(50):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
        M = (M + M.T) / 2
    M = M / M.sum(dim=-1, keepdim=True)
    A = M.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J4Reversibility().compute(A, ctx)
    # π uniforme par symétrie doublement stochastique → diag(π) A = (1/N) A symétrique
    assert out["epsilon_detailed_balance_median"] < 0.05


def test_j4_registered() -> None:
    assert REGISTRY.get("J4_reversibility") is J4Reversibility
