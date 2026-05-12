"""Tests vague finale — C7 Fisher, C9 Wasserstein, I3 head clustering."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_c_token_stats.c7_fisher_information import (
    C7FisherInformation,
)
from catalog.properties.family_c_token_stats.c9_wasserstein import C9Wasserstein
from catalog.properties.family_i_cross_head.i3_head_clustering import (
    I3HeadClustering,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# C7 FisherInformation
# -----------------------------------------------------------------------------


def test_c7_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C7FisherInformation().compute(A, ctx)
    assert "fisher_trace_median" in out
    assert "fisher_max_p_median" in out


def test_c7_one_hot_has_zero_trace() -> None:
    """A one-hot : Σ p² = 1 → trace(F) = 0."""
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, t] = 1.0
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C7FisherInformation().compute(A, ctx)
    assert out["fisher_trace_median"] < 1e-10
    assert abs(out["fisher_max_p_median"] - 1.0) < 1e-10


def test_c7_uniform_trace_n_minus_1_over_n() -> None:
    """A uniforme p=1/N : Σ p² = 1/N → trace(F) = 1 - 1/N."""
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C7FisherInformation().compute(A, ctx)
    expected = 1.0 - 1.0 / n
    assert abs(out["fisher_trace_median"] - expected) < 1e-6


def test_c7_registered() -> None:
    assert REGISTRY.get("C7_fisher_information") is C7FisherInformation


# -----------------------------------------------------------------------------
# C9 Wasserstein
# -----------------------------------------------------------------------------


def test_c9_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C9Wasserstein().compute(A, ctx)
    assert "wasserstein_median" in out


def test_c9_identical_rows_zero_distance() -> None:
    """Toutes les lignes identiques : W_1 = 0 sur toutes paires."""
    n = 16
    row = torch.softmax(torch.randn(n, dtype=torch.float64), dim=-1)
    A = row.unsqueeze(0).expand(n, n).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = C9Wasserstein().compute(A, ctx)
    assert out["wasserstein_median"] < 1e-10


def test_c9_rejects_invalid_n_pairs() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_pairs"):
        C9Wasserstein(n_pairs=0)


def test_c9_registered() -> None:
    assert REGISTRY.get("C9_wasserstein") is C9Wasserstein


# -----------------------------------------------------------------------------
# I3 HeadClustering
# -----------------------------------------------------------------------------


def test_i3_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 4, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I3HeadClustering().compute(A, ctx)
    assert "diameter" in out
    assert "n_clusters_at_threshold_0p50" in out


def test_i3_identical_heads_single_cluster() -> None:
    """Toutes les têtes identiques → 1 cluster même au seuil le plus bas."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    A = A.repeat(1, 4, 1, 1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I3HeadClustering().compute(A, ctx)
    assert out["n_clusters_at_threshold_0p25"] == 1
    assert out["n_clusters_at_threshold_0p50"] == 1


def test_i3_distinct_heads_more_clusters_at_low_threshold() -> None:
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 4, 8, 8, dtype=torch.float64) * 5, dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I3HeadClustering().compute(A, ctx)
    # 4 têtes très différentes : au seuil bas (25% diamètre), plus de 1 cluster
    assert out["n_clusters_at_threshold_0p25"] >= 1


def test_i3_registered() -> None:
    assert REGISTRY.get("I3_head_clustering") is I3HeadClustering
