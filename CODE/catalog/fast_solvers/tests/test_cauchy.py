"""Tests matrice Cauchy."""

from __future__ import annotations

import pytest
import torch

from catalog.fast_solvers import cauchy_matrix, cauchy_matvec_naive, cauchy_solve


def test_cauchy_matrix_basic() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
    C = cauchy_matrix(x, y)
    # C[0, 0] = 1 / (1 - 0.5) = 2.0
    assert abs(C[0, 0].item() - 2.0) < 1e-12
    # C[2, 1] = 1 / (3 - 1.5) = 1/1.5
    assert abs(C[2, 1].item() - 1 / 1.5) < 1e-12


def test_cauchy_matrix_singular_raises() -> None:
    """x_i == y_j → ValueError."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([1.0, 3.0], dtype=torch.float64)  # x[0] == y[0]
    with pytest.raises(ValueError, match="singulière"):
        cauchy_matrix(x, y)


def test_cauchy_matvec_consistency() -> None:
    x = torch.linspace(0.1, 1.0, 6, dtype=torch.float64)
    y = torch.linspace(2.0, 3.0, 6, dtype=torch.float64)
    v = torch.randn(6, dtype=torch.float64)
    C = cauchy_matrix(x, y)
    y_dense = C @ v
    y_solver = cauchy_matvec_naive(x, y, v)
    assert torch.allclose(y_dense, y_solver, atol=1e-12)


def test_cauchy_solve_roundtrip() -> None:
    """C·u = b → u, vérifier C·u = b.

    Note : Cauchy mal-conditionnée, on accepte 1e-3 (κ(C) ~ 10^10 pour N=8).
    """
    torch.manual_seed(0)
    x = torch.linspace(0.1, 1.0, 8, dtype=torch.float64)
    y = torch.linspace(2.0, 3.0, 8, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64)
    u = cauchy_solve(x, y, b)
    C = cauchy_matrix(x, y)
    err = (C @ u - b).norm() / b.norm()
    assert err.item() < 1e-3


def test_cauchy_solve_multi_rhs() -> None:
    x = torch.linspace(0.1, 1.0, 6, dtype=torch.float64)
    y = torch.linspace(2.0, 3.0, 6, dtype=torch.float64)
    B = torch.randn(6, 3, dtype=torch.float64)
    U = cauchy_solve(x, y, B)
    C = cauchy_matrix(x, y)
    err = (C @ U - B).norm() / B.norm()
    assert err.item() < 1e-6
