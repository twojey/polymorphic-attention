"""Tests pour B1 Toeplitz-distance Property."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_b_structural.b1_toeplitz_distance import (
    B1ToeplitzDistance,
)
from catalog.properties.registry import REGISTRY


def test_b1_basic_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B1ToeplitzDistance()
    out = prop.compute(A, ctx)

    assert "epsilon_median" in out
    assert "epsilon_min" in out
    assert "fraction_below_0p30" in out
    assert out["n_matrices"] == 6  # B=2 * H=3
    assert 0.0 <= out["epsilon_min"] <= out["epsilon_max"] <= 1.0


def test_b1_zero_on_pure_toeplitz_matrix() -> None:
    """Si A est exactement Toeplitz, ε_T ≈ 0."""
    n = 5
    c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[0, 0, i, j] = c[abs(i - j)]
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B1ToeplitzDistance()
    out = prop.compute(A, ctx)
    assert out["epsilon_max"] < 1e-10


def test_b1_high_epsilon_on_random_softmax() -> None:
    """Attention softmax aléatoire est globalement non-Toeplitz."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 32, 32, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B1ToeplitzDistance()
    out = prop.compute(A, ctx)
    assert out["epsilon_min"] > 0.3, "matrice random softmax devrait être non-Toeplitz"


def test_b1_caches_projection() -> None:
    """Le cache PropertyContext stocke bien la projection Toeplitz."""
    A = torch.softmax(torch.randn(2, 2, 6, 6, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B1ToeplitzDistance()
    _ = prop.compute(A, ctx)
    assert any("projection_toeplitz" in k for k in ctx.cache.keys())


def test_b1_registered_in_global_registry() -> None:
    cls = REGISTRY.get("B1_toeplitz_distance")
    assert cls is B1ToeplitzDistance


def test_b1_filter_by_family_b() -> None:
    b_props = REGISTRY.filter(family="B")
    names = {c.name for c in b_props}
    assert "B1_toeplitz_distance" in names


def test_b1_fraction_below_threshold() -> None:
    """fraction_below_0p30 doit être dans [0, 1] et cohérente."""
    A = torch.softmax(torch.randn(4, 4, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = B1ToeplitzDistance()
    out = prop.compute(A, ctx)
    f = out["fraction_below_0p30"]
    assert 0.0 <= f <= 1.0
