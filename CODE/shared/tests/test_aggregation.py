"""Tests aggregation helpers."""

from __future__ import annotations

import numpy as np

from shared.aggregation import aggregate_by_regime, aggregate_by_regime_2d, regime_stats


def test_regime_stats_uniform() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=10000)
    stats = regime_stats(values)
    assert abs(stats.mean - 0.5) < 0.05
    assert abs(stats.median - 0.5) < 0.05
    assert 0.30 < stats.iqr < 0.55  # IQR uniforme [0,1] = 0.5
    assert 0.05 < stats.p10 < 0.15
    assert 0.85 < stats.p90 < 0.95


def test_regime_stats_empty() -> None:
    stats = regime_stats(np.array([]))
    assert stats.n == 0


def test_aggregate_by_regime_monovariate() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    omega = rng.choice([1, 2, 4], size=n)
    values = omega * 0.5 + rng.normal(0, 0.1, size=n)
    aggregated = aggregate_by_regime(values, omega=omega)
    assert ("omega", 1.0) in aggregated
    assert ("omega", 4.0) in aggregated
    # mean croît avec omega
    assert aggregated[("omega", 1.0)].mean < aggregated[("omega", 4.0)].mean


def test_aggregate_2d() -> None:
    rng = np.random.default_rng(0)
    n = 800
    omega = rng.choice([1, 2], size=n)
    delta = rng.choice([10, 20], size=n)
    values = omega * delta + rng.normal(0, 0.1, size=n)
    out = aggregate_by_regime_2d(values, axis1_name="omega", axis1=omega.astype(float),
                                  axis2_name="delta", axis2=delta.astype(float))
    assert (1.0, 10.0) in out
    assert (2.0, 20.0) in out
    assert out[(1.0, 10.0)].mean < out[(2.0, 20.0)].mean
