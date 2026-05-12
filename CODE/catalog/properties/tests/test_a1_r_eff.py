"""Tests pour A1 r_eff Property."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_a_spectral.a1_r_eff import (
    A1ReffTheta99,
    _r_eff_from_singular_values,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# Math primitive _r_eff_from_singular_values
# -----------------------------------------------------------------------------


def test_r_eff_rank_one_singular_values() -> None:
    s = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    r = _r_eff_from_singular_values(s, theta=0.95)
    assert r.item() == 1


def test_r_eff_uniform_singular_values_full_rank() -> None:
    s = torch.ones(1, 10, dtype=torch.float64)
    r = _r_eff_from_singular_values(s, theta=0.95)
    # 9 valeurs sur 10 = 90 % → r_eff = 10
    assert r.item() >= 9


def test_r_eff_all_zero_returns_zero() -> None:
    s = torch.zeros(1, 4, dtype=torch.float64)
    r = _r_eff_from_singular_values(s, theta=0.99)
    assert r.item() == 0


# -----------------------------------------------------------------------------
# Property A1
# -----------------------------------------------------------------------------


def test_a1_property_basic_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = A1ReffTheta99(theta=0.99)
    out = prop.compute(A, ctx)
    assert "r_eff_median" in out
    assert "r_eff_mean" in out
    assert "r_eff_max" in out
    assert out["n_matrices"] == 6  # B=2 * H=3
    assert 1 <= out["r_eff_min"] <= out["r_eff_max"]


def test_a1_property_caches_svd() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    prop = A1ReffTheta99()

    # Premier compute → cache rempli
    _ = prop.compute(A, ctx)
    cache_keys_after_first = list(ctx.cache.keys())
    assert any("svd_singular_values" in k for k in cache_keys_after_first)

    # Deuxième compute sur le même A → cache hit, on vérifie via mock du SVD
    original_svdvals = torch.linalg.svdvals
    n_svd_calls = 0

    def counting_svdvals(*args, **kwargs):
        nonlocal n_svd_calls
        n_svd_calls += 1
        return original_svdvals(*args, **kwargs)

    torch.linalg.svdvals = counting_svdvals
    try:
        _ = prop.compute(A, ctx)
    finally:
        torch.linalg.svdvals = original_svdvals
    assert n_svd_calls == 0, "SVD recalculée alors qu'elle devrait être cachée"


def test_a1_property_rejects_wrong_shape() -> None:
    prop = A1ReffTheta99()
    ctx = PropertyContext()
    A_bad = torch.randn(8, 8)  # 2D, pas 4D
    import pytest
    with pytest.raises(ValueError, match="\\(B, H, N, N\\)"):
        prop.compute(A_bad, ctx)


def test_a1_registered_in_global_registry() -> None:
    """Le décorateur @register_property a bien enregistré A1."""
    cls = REGISTRY.get("A1_r_eff_theta099")
    assert cls is A1ReffTheta99


def test_a1_filtered_by_family_a() -> None:
    a_props = REGISTRY.filter(family="A")
    names = {c.name for c in a_props}
    assert "A1_r_eff_theta099" in names


def test_a1_filter_cost_class_max_excludes_higher() -> None:
    cheap = REGISTRY.filter(cost_class_max=1)
    names = {c.name for c in cheap}
    # A1 a cost_class=2 → exclu
    assert "A1_r_eff_theta099" not in names

    medium = REGISTRY.filter(cost_class_max=2)
    names = {c.name for c in medium}
    assert "A1_r_eff_theta099" in names


def test_a1_theta_validation() -> None:
    import pytest
    with pytest.raises(ValueError, match="theta"):
        A1ReffTheta99(theta=1.5)
    with pytest.raises(ValueError, match="theta"):
        A1ReffTheta99(theta=-0.1)
