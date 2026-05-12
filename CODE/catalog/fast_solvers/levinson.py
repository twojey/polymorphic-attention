"""
levinson.py — Algorithme Levinson-Durbin pour systèmes Toeplitz.

Spec : DOC/CATALOGUE §1.2 (Kailath/Pan).

Levinson-Durbin résout T·x = b en O(N²) au lieu de O(N³) (Gauss).
Pour T Toeplitz (général ; spécialisation symétrique défini positif
plus rapide). Référence : Kailath & Sayed (1999), §2.

Implémentation V1 :
- `levinson_durbin_solve` utilise `scipy.linalg.solve_toeplitz` qui est
  la référence numérique stable (Levinson-Trench-Zohar variant).
- Pour un produit T·x, on construit T explicite et fait T@x (V1 simple).
- V2 (futur) : produit FFT-circulant O(N log N).

API :
- `toeplitz_from_first_row_col(row, col)` : construit T
- `toeplitz_matvec(row, col, x)` : produit T·x
- `levinson_durbin_solve(row, col, b)` : résout T·x = b
"""

from __future__ import annotations

import torch


def toeplitz_from_first_row_col(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Construit la matrice Toeplitz T avec T[0, :] = row, T[:, 0] = col.

    Convention : T[i, j] = row[j - i] si j >= i, sinon col[i - j].
    """
    if abs(row[0].item() - col[0].item()) > 1e-12:
        raise ValueError(f"row[0]={row[0].item()} != col[0]={col[0].item()}")
    N = row.shape[-1]
    M = col.shape[-1]
    out = torch.zeros(M, N, dtype=row.dtype, device=row.device)
    rows_idx = torch.arange(M).unsqueeze(1)
    cols_idx = torch.arange(N).unsqueeze(0)
    diff = cols_idx - rows_idx  # (M, N)
    upper = diff >= 0
    out[upper] = row[diff[upper]]
    out[~upper] = col[-diff[~upper]]
    return out


def toeplitz_matvec(row: torch.Tensor, col: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Produit T·x où T est Toeplitz défini par (row, col).

    Implémentation V1 simple O(N²) via construction explicite.
    V2 FFT embedding circulant : O(N log N) (futur si bottleneck).
    """
    T = toeplitz_from_first_row_col(row, col)
    return T @ x


def levinson_durbin_solve(
    row: torch.Tensor, col: torch.Tensor, b: torch.Tensor,
) -> torch.Tensor:
    """Résout T·x = b où T est Toeplitz défini par (row, col).

    Utilise scipy.linalg.solve_toeplitz (référence stable Levinson). Si
    scipy indisponible, fallback dense `torch.linalg.solve` O(N³).

    Args
    ----
    row : Tensor (N,) — première ligne de T
    col : Tensor (N,) — première colonne de T (col[0] == row[0])
    b : Tensor (N,) ou (N, K) — RHS

    Returns
    -------
    x : Tensor même shape que b
    """
    if abs(row[0].item() - col[0].item()) > 1e-12:
        raise ValueError(f"row[0] != col[0]")
    if row.shape[-1] != col.shape[-1]:
        raise ValueError(f"row.shape != col.shape : {row.shape} vs {col.shape}")

    try:
        from scipy.linalg import solve_toeplitz
    except ImportError:
        # Fallback dense
        T = toeplitz_from_first_row_col(row, col)
        if b.dim() == 1:
            return torch.linalg.solve(T, b.unsqueeze(-1)).squeeze(-1)
        return torch.linalg.solve(T, b)

    # scipy.solve_toeplitz(c_or_cr, b) :
    # - si c_or_cr est tuple (c, r) : T avec col=c, row=r
    # - sinon c (T symétrique)
    c_np = col.detach().cpu().numpy()
    r_np = row.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    x_np = solve_toeplitz((c_np, r_np), b_np)
    return torch.tensor(x_np, dtype=row.dtype, device=row.device)
