"""Tests pour familles D, F, I (quick wins)."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_d_geometric.d1_head_cosine import D1HeadCosine
from catalog.properties.family_d_geometric.d2_distance_baselines import (
    D2DistanceBaselines,
)
from catalog.properties.family_d_geometric.d3_subspace_angles import D3SubspaceAngles
from catalog.properties.family_f_dynamic.f2_temporal_stability import (
    F2TemporalStability,
)
from catalog.properties.family_i_cross_head.i1_head_diversity import I1HeadDiversity
from catalog.properties.family_i_cross_head.i2_head_specialization import (
    I2HeadSpecialization,
)
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# D1 HeadCosine
# -----------------------------------------------------------------------------


def test_d1_identical_heads_cosine_one() -> None:
    """Toutes les têtes identiques → cos = 1 sur toutes paires."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    A = A.repeat(1, 4, 1, 1)  # 4 têtes identiques
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = D1HeadCosine().compute(A, ctx)
    assert abs(out["cosine_median"] - 1.0) < 1e-6
    assert out["fraction_redundant"] == 1.0


def test_d1_single_head_degenerate() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = D1HeadCosine().compute(A, ctx)
    assert out["n_head_pairs"] == 0


def test_d1_registered() -> None:
    assert REGISTRY.get("D1_head_cosine") is D1HeadCosine


# -----------------------------------------------------------------------------
# D2 DistanceBaselines
# -----------------------------------------------------------------------------


def test_d2_uniform_matrix_zero_distance_uniform() -> None:
    n = 8
    A = torch.ones(1, 1, n, n, dtype=torch.float64) / n
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = D2DistanceBaselines().compute(A, ctx)
    assert out["distance_uniform_median"] < 1e-10


def test_d2_identity_zero_distance_identity() -> None:
    A = torch.eye(8, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = D2DistanceBaselines().compute(A, ctx)
    assert out["distance_identity_median"] < 1e-10


def test_d2_registered() -> None:
    assert REGISTRY.get("D2_distance_baselines") is D2DistanceBaselines


# -----------------------------------------------------------------------------
# D3 SubspaceAngles
# -----------------------------------------------------------------------------


def test_d3_identical_heads_aligned() -> None:
    """Têtes identiques → angle_max ≈ 0."""
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    A = A.repeat(1, 4, 1, 1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = D3SubspaceAngles(top_r=3).compute(A, ctx)
    # cos sigmas ≈ 1, angles ≈ 0
    assert out["angle_max_median_deg"] < 1.0
    assert out["fraction_aligned"] == 1.0


def test_d3_registered() -> None:
    assert REGISTRY.get("D3_subspace_angles") is D3SubspaceAngles


# -----------------------------------------------------------------------------
# F2 TemporalStability
# -----------------------------------------------------------------------------


def test_f2_constant_attention_zero_delta() -> None:
    """Toutes lignes identiques → δ = 0."""
    n = 8
    row = torch.softmax(torch.randn(n, dtype=torch.float64), dim=-1)
    A = row.unsqueeze(0).expand(n, n).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = F2TemporalStability().compute(A, ctx)
    assert out["delta_median"] < 1e-10


def test_f2_random_softmax_nonzero_delta() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = F2TemporalStability().compute(A, ctx)
    assert out["delta_median"] > 0.01


def test_f2_registered() -> None:
    assert REGISTRY.get("F2_temporal_stability") is F2TemporalStability


# -----------------------------------------------------------------------------
# I1 HeadDiversity
# -----------------------------------------------------------------------------


def test_i1_identical_heads_zero_variance() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    A = A.repeat(1, 4, 1, 1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I1HeadDiversity().compute(A, ctx)
    assert out["var_h_median"] < 1e-10


def test_i1_different_heads_positive_variance() -> None:
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(1, 4, 8, 8, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I1HeadDiversity().compute(A, ctx)
    assert out["var_h_median"] > 0.0


def test_i1_registered() -> None:
    assert REGISTRY.get("I1_head_diversity") is I1HeadDiversity


# -----------------------------------------------------------------------------
# I2 HeadSpecialization
# -----------------------------------------------------------------------------


def test_i2_identical_heads_zero_specialization() -> None:
    A = torch.softmax(torch.randn(1, 1, 8, 8, dtype=torch.float64), dim=-1)
    A = A.repeat(2, 4, 1, 1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I2HeadSpecialization().compute(A, ctx)
    assert out["specialization_score_max"] < 1e-6


def test_i2_specialized_heads_high_score() -> None:
    """Têtes "spécialisées" : chacune a un pattern différent."""
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(4, 4, 8, 8, dtype=torch.float64) * 5, dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = I2HeadSpecialization().compute(A, ctx)
    assert out["specialization_score_max"] > 0.0


def test_i2_registered() -> None:
    assert REGISTRY.get("I2_head_specialization") is I2HeadSpecialization
