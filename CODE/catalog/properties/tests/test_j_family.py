"""Tests pour famille J — J1, J3."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_j_markov.j1_markov_convergence import (
    J1MarkovConvergence,
)
from catalog.properties.family_j_markov.j3_mixing_time import J3MixingTime
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# J1 MarkovConvergence
# -----------------------------------------------------------------------------


def test_j1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J1MarkovConvergence(powers=(2, 4)).compute(A, ctx)
    assert "rank1_residual_k2_median" in out
    assert "rank1_residual_k4_median" in out
    assert "sigma2_over_sigma1_k2_median" in out


def test_j1_uniform_matrix_immediate_rank1() -> None:
    """A = 1/N partout → A^k = 1/N partout → rank-1 résidu ≈ 0."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J1MarkovConvergence(powers=(2, 4)).compute(A, ctx)
    assert out["rank1_residual_k2_median"] < 1e-10
    assert out["rank1_residual_k4_median"] < 1e-10


def test_j1_random_softmax_residual_decreases() -> None:
    """Pour A row-stochastique, residual_k devrait baisser monotonement avec k."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J1MarkovConvergence(powers=(2, 4, 8, 16)).compute(A, ctx)
    r2 = out["rank1_residual_k2_median"]
    r4 = out["rank1_residual_k4_median"]
    r8 = out["rank1_residual_k8_median"]
    r16 = out["rank1_residual_k16_median"]
    # Décroissance monotone
    assert r2 >= r4 >= r8 >= r16


def test_j1_registered() -> None:
    assert REGISTRY.get("J1_markov_convergence") is J1MarkovConvergence


# -----------------------------------------------------------------------------
# J3 MixingTime
# -----------------------------------------------------------------------------


def test_j3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J3MixingTime(max_iter=20).compute(A, ctx)
    assert "tmix_eps_0p50_median" in out
    assert "tmix_eps_0p01_median" in out


def test_j3_uniform_matrix_mixed_at_t_one() -> None:
    """A uniform → A^1 = A = π → tmix(ε) = 1 pour tout ε > 0."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J3MixingTime(epsilons=(0.5, 0.1), max_iter=20).compute(A, ctx)
    assert out["tmix_eps_0p50_median"] == 1.0
    assert out["tmix_eps_0p10_median"] == 1.0


def test_j3_softmax_eventually_mixes() -> None:
    """softmax(N(0,1)) row-stochastique → mixing fini pour ε modéré."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = J3MixingTime(epsilons=(0.25, 0.05), max_iter=40).compute(A, ctx)
    assert out["tmix_eps_0p25_saturated_fraction"] == 0.0


def test_j3_rejects_invalid_epsilon() -> None:
    import pytest

    with pytest.raises(ValueError, match="epsilon"):
        J3MixingTime(epsilons=(0.0,))


def test_j3_registered() -> None:
    assert REGISTRY.get("J3_mixing_time") is J3MixingTime
