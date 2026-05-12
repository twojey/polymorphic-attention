"""Tests math pour BlockDiagonal et Banded projectors."""

from __future__ import annotations

import pytest
import torch

from catalog.projectors import Banded, BlockDiagonal


# -----------------------------------------------------------------------------
# BlockDiagonal
# -----------------------------------------------------------------------------


def test_block_diagonal_keeps_block_diagonal_matrix() -> None:
    n, bs = 6, 3
    A = torch.zeros(n, n, dtype=torch.float64)
    A[0:3, 0:3] = torch.eye(3) * 2.0
    A[3:6, 3:6] = torch.eye(3) * 3.0
    proj = BlockDiagonal(block_size=bs).project(A)
    assert torch.allclose(proj, A, atol=1e-10)


def test_block_diagonal_zeros_off_block() -> None:
    n, bs = 4, 2
    A = torch.ones(n, n, dtype=torch.float64)
    proj = BlockDiagonal(block_size=bs).project(A)
    # Off-block elements (e.g., [0,2], [0,3], [1,2], [1,3], [2,0], [2,1], etc.)
    assert proj[0, 2].item() == 0.0
    assert proj[0, 3].item() == 0.0
    assert proj[2, 0].item() == 0.0
    # In-block preserved
    assert proj[0, 0].item() == 1.0
    assert proj[0, 1].item() == 1.0


def test_block_diagonal_invalid_block_size_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        BlockDiagonal(block_size=0)


def test_block_diagonal_with_block_size_one_equals_identity_pattern() -> None:
    """block_size=1 → seulement la diagonale principale."""
    n = 5
    A = torch.randn(n, n, dtype=torch.float64)
    proj = BlockDiagonal(block_size=1).project(A)
    diag = torch.diag(torch.diagonal(A))
    assert torch.allclose(proj, diag, atol=1e-10)


def test_block_diagonal_batched() -> None:
    A = torch.randn(2, 3, 6, 6, dtype=torch.float64)
    proj = BlockDiagonal(block_size=2).project(A)
    assert proj.shape == A.shape


# -----------------------------------------------------------------------------
# Banded
# -----------------------------------------------------------------------------


def test_banded_keeps_diagonal_with_zero_bandwidth() -> None:
    n = 5
    A = torch.randn(n, n, dtype=torch.float64)
    proj = Banded(bandwidth=0).project(A)
    expected = torch.diag(torch.diagonal(A))
    assert torch.allclose(proj, expected, atol=1e-12)


def test_banded_keeps_full_matrix_with_large_bandwidth() -> None:
    n = 5
    A = torch.randn(n, n, dtype=torch.float64)
    proj = Banded(bandwidth=n - 1).project(A)
    assert torch.allclose(proj, A, atol=1e-12)


def test_banded_zeros_far_off_diagonal() -> None:
    n = 8
    A = torch.ones(n, n, dtype=torch.float64)
    proj = Banded(bandwidth=2).project(A)
    # |i - j| > 2 zéros
    assert proj[0, 5].item() == 0.0
    assert proj[5, 0].item() == 0.0
    # |i - j| ≤ 2 préservé
    assert proj[0, 2].item() == 1.0
    assert proj[3, 5].item() == 1.0


def test_banded_invalid_bandwidth_raises() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        Banded(bandwidth=-1)


def test_banded_batched() -> None:
    A = torch.randn(2, 3, 7, 7, dtype=torch.float64)
    proj = Banded(bandwidth=2).project(A)
    assert proj.shape == A.shape


def test_banded_residual_orthogonal_to_projection() -> None:
    """Math : <A - P_B(A), P_B(A)> = 0."""
    A = torch.randn(6, 6, dtype=torch.float64)
    b = Banded(bandwidth=2)
    proj = b.project(A)
    res = b.residual(A)
    inner = (res * proj).sum().item()
    assert abs(inner) < 1e-10
