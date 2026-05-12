"""Tests pour famille K — K1, K3."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_k_graph.k1_laplacian_spectrum import (
    K1LaplacianSpectrum,
)
from catalog.properties.family_k_graph.k3_pagerank_centrality import (
    K3PageRankCentrality,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# K1 LaplacianSpectrum
# -----------------------------------------------------------------------------


def test_k1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K1LaplacianSpectrum().compute(A, ctx)
    assert "lambda_1_median" in out
    assert "lambda_2_fiedler_median" in out
    assert "spectral_gap_median" in out


def test_k1_lambda1_zero_for_connected_graph() -> None:
    """A_sym ≥ 0 sur graphe connexe → λ_1 = 0."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K1LaplacianSpectrum().compute(A, ctx)
    # λ_1 doit être ~ 0 (peut-être très petit nombre numérique)
    assert abs(out["lambda_1_median"]) < 1e-6


def test_k1_normalized_lambda_max_at_most_two() -> None:
    """Pour Laplacien normalisé, λ_max ≤ 2."""
    A = torch.softmax(torch.randn(2, 2, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K1LaplacianSpectrum(normalize=True).compute(A, ctx)
    assert out["lambda_max_median"] <= 2.0 + 1e-6


def test_k1_registered() -> None:
    assert REGISTRY.get("K1_laplacian_spectrum") is K1LaplacianSpectrum


# -----------------------------------------------------------------------------
# K3 PageRankCentrality
# -----------------------------------------------------------------------------


def test_k3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K3PageRankCentrality().compute(A, ctx)
    assert "pi_entropy_norm_median" in out
    assert "pi_max_mass_median" in out
    assert "power_iter_residual_max" in out


def test_k3_uniform_matrix_gives_uniform_pi() -> None:
    """A = 1/N partout → π = uniforme → entropy_norm = 1, gini = 0."""
    n = 16
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K3PageRankCentrality(damping=1.0).compute(A, ctx)
    assert abs(out["pi_entropy_norm_median"] - 1.0) < 1e-3
    assert out["pi_gini_median"] < 1e-3


def test_k3_converges_on_random_softmax() -> None:
    """Power iteration converge à tol < 1e-4 sur softmax aléatoire."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = K3PageRankCentrality(max_iter=200, tol=1e-8).compute(A, ctx)
    assert out["power_iter_residual_max"] < 1e-4


def test_k3_rejects_invalid_damping() -> None:
    import pytest

    with pytest.raises(ValueError, match="damping"):
        K3PageRankCentrality(damping=1.5)
    with pytest.raises(ValueError, match="damping"):
        K3PageRankCentrality(damping=0.0)


def test_k3_registered() -> None:
    assert REGISTRY.get("K3_pagerank_centrality") is K3PageRankCentrality
