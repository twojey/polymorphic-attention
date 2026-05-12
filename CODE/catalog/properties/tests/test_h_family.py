"""Tests pour famille H — cross-layer H1 + H3."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_h_cross_layer.h1_layer_residual import H1LayerResidual
from catalog.properties.family_h_cross_layer.h3_reff_trajectory import H3REffTrajectory
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# H1 LayerResidual
# -----------------------------------------------------------------------------


def test_h1_requires_at_least_2_layers() -> None:
    import pytest

    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="au moins 2 couches"):
        H1LayerResidual().compute([A], ctx)


def test_h1_identical_layers_zero_residual() -> None:
    """Si toutes les couches sont identiques, ε = 0."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    layers = [A.clone() for _ in range(4)]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H1LayerResidual().compute(layers, ctx)
    assert out["epsilon_consecutive_median"] < 1e-10
    assert out["epsilon_first_last_median"] < 1e-10
    assert out["n_layer_pairs"] == 3


def test_h1_different_layers_nonzero() -> None:
    torch.manual_seed(0)
    layers = [
        torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
        for _ in range(4)
    ]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H1LayerResidual().compute(layers, ctx)
    assert out["epsilon_consecutive_median"] > 0.1
    assert "epsilon_layer_0_to_1_median" in out
    assert out["n_layers"] == 4


def test_h1_per_pair_keys_exist() -> None:
    layers = [
        torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
        for _ in range(3)
    ]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H1LayerResidual().compute(layers, ctx)
    assert "epsilon_layer_0_to_1_median" in out
    assert "epsilon_layer_1_to_2_median" in out


def test_h1_registered() -> None:
    assert REGISTRY.get("H1_layer_residual") is H1LayerResidual
    assert H1LayerResidual.scope == "per_regime_layers"


# -----------------------------------------------------------------------------
# H3 REffTrajectory
# -----------------------------------------------------------------------------


def test_h3_basic_output_shape() -> None:
    layers = [
        torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
        for _ in range(3)
    ]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H3REffTrajectory().compute(layers, ctx)
    assert "r_eff_layer_0_median" in out
    assert "r_eff_layer_1_median" in out
    assert "r_eff_layer_2_median" in out
    assert "r_eff_layer_argmax" in out
    assert out["n_layers"] == 3


def test_h3_constant_rank_layers() -> None:
    """Toutes layers même rang → range = 0."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # rank N
    layers = [A.clone() for _ in range(3)]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = H3REffTrajectory().compute(layers, ctx)
    assert out["r_eff_layer_range"] == 0.0


def test_h3_rejects_invalid_theta() -> None:
    import pytest

    with pytest.raises(ValueError, match="theta"):
        H3REffTrajectory(theta_cumulative=0.0)
    with pytest.raises(ValueError, match="theta"):
        H3REffTrajectory(theta_cumulative=1.5)


def test_h3_registered() -> None:
    assert REGISTRY.get("H3_reff_trajectory") is H3REffTrajectory
    assert H3REffTrajectory.scope == "per_regime_layers"
