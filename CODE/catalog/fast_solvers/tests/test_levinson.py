"""Tests Levinson-Durbin Toeplitz."""

from __future__ import annotations

import torch

from catalog.fast_solvers import (
    levinson_durbin_solve,
    toeplitz_from_first_row_col,
    toeplitz_matvec,
)


def _make_toeplitz_spd(N: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Génère une Toeplitz symétrique définie positive."""
    torch.manual_seed(seed)
    # SPD via auto-corrélation (théorème Wiener-Khinchin discret)
    x = torch.randn(N, dtype=torch.float64)
    # ACF discrète
    acf = torch.zeros(N, dtype=torch.float64)
    for k in range(N):
        acf[k] = (x[: N - k] * x[k:]).sum() / N
    # Symétrise + ajoute regularisation pour SPD strict
    acf[0] += 1.0
    return acf, acf  # row == col pour symétrique


def test_toeplitz_construction() -> None:
    row = torch.tensor([1.0, 2.0, 3.0])
    col = torch.tensor([1.0, 4.0, 5.0])
    T = toeplitz_from_first_row_col(row, col)
    # T[0, :] = row, T[:, 0] = col
    assert torch.allclose(T[0, :], row)
    assert torch.allclose(T[:, 0], col)
    # T[1, 2] = row[1] (j-i = 1)
    assert T[1, 2].item() == 2.0


def test_toeplitz_matvec_consistency() -> None:
    """T·x calculé via matvec == produit dense."""
    row, col = _make_toeplitz_spd(N=8, seed=0)
    T = toeplitz_from_first_row_col(row, col)
    x = torch.randn(8, dtype=torch.float64)
    y_dense = T @ x
    y_solver = toeplitz_matvec(row, col, x)
    assert torch.allclose(y_dense, y_solver, atol=1e-10)


def test_levinson_solve_spd_8() -> None:
    """Levinson sur Toeplitz SPD N=8 : x reconstruit à 1e-8."""
    row, col = _make_toeplitz_spd(N=8, seed=42)
    T = toeplitz_from_first_row_col(row, col)
    b = torch.randn(8, dtype=torch.float64)
    x_levinson = levinson_durbin_solve(row, col, b)
    x_solve = torch.linalg.solve(T, b)
    err = (x_levinson - x_solve).norm() / x_solve.norm()
    assert err.item() < 1e-6


def test_levinson_solve_n16() -> None:
    """Levinson N=16 : précision."""
    row, col = _make_toeplitz_spd(N=16, seed=1)
    T = toeplitz_from_first_row_col(row, col)
    b = torch.randn(16, dtype=torch.float64)
    x_levinson = levinson_durbin_solve(row, col, b)
    err = (T @ x_levinson - b).norm() / b.norm()
    assert err.item() < 1e-6


def test_levinson_multi_rhs() -> None:
    """Levinson avec b ∈ ℝ^{N×K} (plusieurs RHS)."""
    row, col = _make_toeplitz_spd(N=8, seed=2)
    T = toeplitz_from_first_row_col(row, col)
    B_rhs = torch.randn(8, 3, dtype=torch.float64)
    X_lev = levinson_durbin_solve(row, col, B_rhs)
    X_solve = torch.linalg.solve(T, B_rhs)
    err = (X_lev - X_solve).norm() / X_solve.norm()
    assert err.item() < 1e-6


def test_levinson_n1_edge() -> None:
    """N=1 cas dégénéré."""
    row = torch.tensor([2.0])
    col = torch.tensor([2.0])
    b = torch.tensor([6.0])
    x = levinson_durbin_solve(row, col, b)
    assert abs(x.item() - 3.0) < 1e-10
