"""Tests pour famille C — C2, C3, C4, C5."""

from __future__ import annotations

import math

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_c_token_stats.c2_kl_uniform import C2KLUniform
from catalog.properties.family_c_token_stats.c3_shannon_entropy import C3ShannonEntropy
from catalog.properties.family_c_token_stats.c4_renyi_entropy import C4RenyiEntropy
from catalog.properties.family_c_token_stats.c5_row_variance import C5RowVariance
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# C2 KL Uniform
# -----------------------------------------------------------------------------


def test_c2_uniform_distribution_zero_kl() -> None:
    """A uniforme partout → KL vs uniform = 0."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C2KLUniform(causal=False).compute(A, ctx)
    assert abs(out["kl_uniform_median"]) < 1e-6


def test_c2_one_hot_max_kl_log_n() -> None:
    """A one-hot → KL = log N."""
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, t] = 1.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C2KLUniform(causal=False).compute(A, ctx)
    # KL vs uniform = log N − H = log N − 0 = log N
    assert abs(out["kl_uniform_median"] - math.log(n)) < 1e-3


def test_c2_registered() -> None:
    assert REGISTRY.get("C2_kl_uniform") is C2KLUniform


# -----------------------------------------------------------------------------
# C3 Shannon
# -----------------------------------------------------------------------------


def test_c3_uniform_entropy_log_n() -> None:
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C3ShannonEntropy().compute(A, ctx)
    assert abs(out["shannon_entropy_median"] - math.log(n)) < 1e-3
    assert abs(out["shannon_entropy_norm_median"] - 1.0) < 1e-3


def test_c3_one_hot_entropy_zero() -> None:
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, (t + 2) % n] = 1.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C3ShannonEntropy().compute(A, ctx)
    assert out["shannon_entropy_median"] < 1e-6
    assert out["fraction_rows_norm_below_0p20"] == 1.0


def test_c3_registered() -> None:
    assert REGISTRY.get("C3_shannon_entropy") is C3ShannonEntropy


# -----------------------------------------------------------------------------
# C4 Rényi
# -----------------------------------------------------------------------------


def test_c4_uniform_renyi_log_n() -> None:
    """Toute α : pour uniform p_i = 1/N, H_α = log N."""
    n = 16
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C4RenyiEntropy(alphas=(2.0, 0.5)).compute(A, ctx)
    assert abs(out["renyi_alpha_2p0_median"] - math.log(n)) < 1e-3
    assert abs(out["renyi_alpha_0p5_median"] - math.log(n)) < 1e-3
    # min-entropy aussi = log N pour uniform
    assert abs(out["min_entropy_median"] - math.log(n)) < 1e-3


def test_c4_one_hot_min_entropy_zero() -> None:
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, t] = 1.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C4RenyiEntropy(alphas=(2.0,)).compute(A, ctx)
    # H_2 = -log Σ p² = -log 1 = 0 pour one-hot
    assert abs(out["renyi_alpha_2p0_median"]) < 1e-6
    # H_∞ = 0 aussi
    assert abs(out["min_entropy_median"]) < 1e-6


def test_c4_invalid_alpha_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="alpha=1.0 invalide"):
        C4RenyiEntropy(alphas=(1.0,))


def test_c4_registered() -> None:
    assert REGISTRY.get("C4_renyi_entropy") is C4RenyiEntropy


# -----------------------------------------------------------------------------
# C5 RowVariance
# -----------------------------------------------------------------------------


def test_c5_uniform_zero_variance() -> None:
    n = 16
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C5RowVariance().compute(A, ctx)
    assert out["row_variance_median"] < 1e-15
    # Purity uniforme = 1/N
    assert abs(out["purity_median"] - 1.0 / n) < 1e-6


def test_c5_one_hot_purity_one() -> None:
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, t] = 1.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C5RowVariance().compute(A, ctx)
    assert abs(out["purity_median"] - 1.0) < 1e-10
    # Variance: E[X²] = 1/N, E[X]² = 1/N², Var = 1/N − 1/N² = (N-1)/N²
    expected_var = (n - 1) / (n * n)
    assert abs(out["row_variance_median"] - expected_var) < 1e-6


def test_c5_registered() -> None:
    assert REGISTRY.get("C5_row_variance") is C5RowVariance
