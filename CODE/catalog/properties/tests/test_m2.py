"""Tests pour M2 StressVariation (cross_regime)."""

from __future__ import annotations

import torch

from catalog.oracles.base import AttentionDump
from catalog.properties.base import PropertyContext
from catalog.properties.family_m_conditional.m2_stress_variation import (
    M2StressVariation,
    _spearman_rho,
)
from catalog.properties.registry import REGISTRY


def _make_dump(omega: float, delta: float, attn_shape: tuple[int, ...] = (4, 2, 8, 8), seed: int = 0) -> AttentionDump:
    """Construit un AttentionDump minimaliste avec stress params donnés."""
    torch.manual_seed(seed)
    B, H, N, _ = attn_shape
    attn_layers = [
        torch.softmax(torch.randn(*attn_shape, dtype=torch.float64), dim=-1)
        for _ in range(3)  # 3 layers
    ]
    return AttentionDump(
        attn=attn_layers,
        omegas=torch.full((B,), omega, dtype=torch.float64),
        deltas=torch.full((B,), delta, dtype=torch.float64),
        entropies=torch.zeros(B, dtype=torch.float64),
        tokens=torch.zeros(B, N, dtype=torch.long),
        query_pos=torch.zeros(B, dtype=torch.long),
        metadata={"regime": (omega, delta, 0.0)},
    )


def test_spearman_perfect_correlation() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    assert abs(_spearman_rho(x, y) - 1.0) < 1e-10


def test_spearman_negative_correlation() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 8.0, 6.0, 4.0])
    assert abs(_spearman_rho(x, y) - (-1.0)) < 1e-10


def test_spearman_handles_singleton() -> None:
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    import math
    assert math.isnan(_spearman_rho(x, y))


def test_m2_basic_output_shape() -> None:
    dumps = {
        (0,): _make_dump(0.0, 16.0, seed=0),
        (1,): _make_dump(1.0, 16.0, seed=1),
        (2,): _make_dump(2.0, 16.0, seed=2),
    }
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = M2StressVariation().compute(dumps, ctx)
    assert "rho_spearman_r_eff_vs_omega" in out
    assert "rho_spearman_H_spec_norm_vs_omega" in out
    assert out["n_regimes"] == 3


def test_m2_rejects_empty_dict() -> None:
    import pytest

    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="dict non-vide"):
        M2StressVariation().compute({}, ctx)


def test_m2_registered() -> None:
    assert REGISTRY.get("M2_stress_variation") is M2StressVariation
    assert M2StressVariation.scope == "cross_regime"
