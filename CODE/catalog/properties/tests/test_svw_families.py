"""Tests familles S (tenseurs), V (opérateurs), W (logique)."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_s_tensors.s1_tucker_rank import S1TuckerRank
from catalog.properties.family_v_operators.v2_cz_decay import V2CZDecay
from catalog.properties.family_w_logic.w1_pattern_complexity import W1PatternComplexity
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# S1 Tucker rank
# -----------------------------------------------------------------------------


def test_s1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = S1TuckerRank().compute(A, ctx)
    assert "tucker_rank_batch" in out
    assert "tucker_rank_head" in out
    assert "tucker_rank_row" in out
    assert "tucker_rank_col" in out


def test_s1_constant_tensor_rank_1_on_all_modes() -> None:
    """A constant partout : tous les rangs sont 1."""
    A = torch.full((2, 3, 8, 8), 0.5, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = S1TuckerRank().compute(A, ctx)
    assert out["tucker_rank_batch"] == 1
    assert out["tucker_rank_head"] == 1
    assert out["tucker_rank_row"] == 1
    assert out["tucker_rank_col"] == 1


def test_s1_registered() -> None:
    assert REGISTRY.get("S1_tucker_rank") is S1TuckerRank


# -----------------------------------------------------------------------------
# V2 CZ decay
# -----------------------------------------------------------------------------


def test_v2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(2, 3, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = V2CZDecay().compute(A, ctx)
    assert "cz_decay_alpha_median" in out
    assert "cz_decay_r_squared_median" in out


def test_v2_polynomial_decay_detected() -> None:
    """A[i,j] = 1 / (1 + |i-j|)^2 → alpha ≈ 2."""
    n = 32
    i = torch.arange(n).view(-1, 1)
    j = torch.arange(n).view(1, -1)
    A = 1.0 / ((1.0 + (i - j).abs()) ** 2)
    A = A.unsqueeze(0).unsqueeze(0).to(torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = V2CZDecay().compute(A, ctx)
    assert abs(out["cz_decay_alpha_median"] - 2.0) < 0.3
    assert out["cz_decay_r_squared_median"] > 0.9


def test_v2_rejects_invalid_min_lag() -> None:
    import pytest

    with pytest.raises(ValueError, match="min_lag"):
        V2CZDecay(min_lag=0)


def test_v2_registered() -> None:
    assert REGISTRY.get("V2_cz_decay") is V2CZDecay


# -----------------------------------------------------------------------------
# W1 Pattern complexity
# -----------------------------------------------------------------------------


def test_w1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = W1PatternComplexity().compute(A, ctx)
    assert "pattern_unique_rows_median" in out
    assert "pattern_entropy_median" in out


def test_w1_constant_matrix_unique_one() -> None:
    """A constant → toutes lignes pareilles → unique = 1."""
    A = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float64)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = W1PatternComplexity().compute(A, ctx)
    assert out["pattern_unique_rows_median"] == 1.0
    assert out["pattern_entropy_median"] < 1e-10


def test_w1_identity_high_unique() -> None:
    """I_n : chaque ligne est différente → unique = N."""
    n = 8
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = W1PatternComplexity().compute(A, ctx)
    assert out["pattern_unique_rows_median"] == n


def test_w1_rejects_invalid_threshold() -> None:
    import pytest

    with pytest.raises(ValueError, match="threshold_ratio"):
        W1PatternComplexity(threshold_ratio=0.0)
    with pytest.raises(ValueError, match="threshold_ratio"):
        W1PatternComplexity(threshold_ratio=1.0)


def test_w1_registered() -> None:
    assert REGISTRY.get("W1_pattern_complexity") is W1PatternComplexity
