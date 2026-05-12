"""Tests familles Q (hiérarchiques) + P (réalisation Ho-Kalman)."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_p_realization.p1_hankel_realization import (
    P1HankelRealization,
)
from catalog.properties.family_q_hierarchical.q1_hmatrix_structure import (
    Q1HMatrixStructure,
)
from catalog.properties.family_q_hierarchical.q2_hss_rank import Q2HSSRank
from catalog.properties.registry import REGISTRY


# -----------------------------------------------------------------------------
# Q1 H-matrix structure
# -----------------------------------------------------------------------------


def test_q1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = Q1HMatrixStructure(n_levels=2).compute(A, ctx)
    assert "r_eff_offdiag_level_1_median" in out
    assert "r_eff_offdiag_level_2_median" in out
    assert "r_eff_offdiag_global_median" in out


def test_q1_low_rank_blocks_have_low_r_eff() -> None:
    """Construire A avec off-diagonal rank-1 par bloc → r_eff = 1."""
    n = 16
    A = torch.zeros(n, n, dtype=torch.float64)
    # Diagonale full rank
    A[:8, :8] = torch.eye(8)
    A[8:, 8:] = torch.eye(8)
    # Off-diagonal rank-1
    u = torch.randn(8, 1, dtype=torch.float64)
    v = torch.randn(1, 8, dtype=torch.float64)
    A[:8, 8:] = u @ v
    u2 = torch.randn(8, 1, dtype=torch.float64)
    v2 = torch.randn(1, 8, dtype=torch.float64)
    A[8:, :8] = u2 @ v2
    A_bat = A.unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = Q1HMatrixStructure(n_levels=1).compute(A_bat, ctx)
    # Level 1 : 2 blocs off-diagonaux 8×8, tous deux rank-1
    assert out["r_eff_offdiag_level_1_max"] == 1


def test_q1_rejects_invalid_levels() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_levels"):
        Q1HMatrixStructure(n_levels=0)


def test_q1_registered() -> None:
    assert REGISTRY.get("Q1_hmatrix_structure") is Q1HMatrixStructure


# -----------------------------------------------------------------------------
# Q2 HSS rank
# -----------------------------------------------------------------------------


def test_q2_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = Q2HSSRank(n_splits=4).compute(A, ctx)
    assert "hss_rank_median" in out
    assert "n_separated_blocs" in out


def test_q2_rejects_invalid_n_splits() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_splits"):
        Q2HSSRank(n_splits=1)


def test_q2_registered() -> None:
    assert REGISTRY.get("Q2_hss_rank") is Q2HSSRank


# -----------------------------------------------------------------------------
# P1 Hankel realization
# -----------------------------------------------------------------------------


def test_p1_basic_output_shape() -> None:
    A = torch.softmax(torch.randn(1, 1, 16, 16, dtype=torch.float64), dim=-1)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = P1HankelRealization().compute(A, ctx)
    assert "hankel_rank_median" in out
    assert "hsv_sigma2_over_sigma1_median" in out


def test_p1_constant_rows_low_rank() -> None:
    """Si toutes les lignes de A sont identiques → Hankel rang 1."""
    n = 16
    row = torch.randn(n, dtype=torch.float64)
    A = row.unsqueeze(0).expand(n, n).unsqueeze(0).unsqueeze(0)
    ctx = PropertyContext(device="cpu", dtype=torch.float64)
    out = P1HankelRealization().compute(A, ctx)
    # Hankel block construite à partir d'une seule ligne distincte → rang 1
    assert out["hankel_rank_max"] == 1


def test_p1_registered() -> None:
    assert REGISTRY.get("P1_hankel_realization") is P1HankelRealization
