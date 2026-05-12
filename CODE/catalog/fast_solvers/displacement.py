"""
displacement.py — Utilitaires rang de déplacement Sylvester (Kailath/Pan).

Spec : DOC/CATALOGUE §1.2.

Pour deux opérateurs (M, N) et une matrice A, le **déplacement Sylvester**
est :
    ∇_{M, N}(A) = M·A − A·N

Si rang(∇_{M, N}(A)) = r, on peut factoriser :
    ∇_{M, N}(A) = G · Bᵀ,  G ∈ ℝ^{N×r}, B ∈ ℝ^{N×r}
G et B sont les **générateurs**, et toute la structure de A se concentre
dedans (paramétrisation O(rN) au lieu de O(N²)).

API :
- `sylvester_displacement(A, M, N)` : calcule ∇_{M, N}(A)
- `extract_displacement_generators(A, M, N, r)` : SVD du déplacement → (G, B)
- `displacement_residual_norm(A, M, N, G, B)` : ‖∇(A) − G Bᵀ‖_F / ‖∇(A)‖_F

Réutilisé par Properties O1, O2, O4 pour validation cross-méthode.
"""

from __future__ import annotations

import torch


def sylvester_displacement(
    A: torch.Tensor, M: torch.Tensor, N: torch.Tensor
) -> torch.Tensor:
    """Calcule ∇_{M, N}(A) = M·A − A·N. Supporte batched (B, H, n, n)."""
    return M @ A - A @ N


def extract_displacement_generators(
    A: torch.Tensor, M: torch.Tensor, N: torch.Tensor, r: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Extrait (G, B) tels que ∇_{M,N}(A) ≈ G · Bᵀ, rang r.

    Args
    ----
    A : Tensor (..., n, n)
    M, N : Tensor (n, n) — opérateurs de déplacement
    r : rang cible

    Returns
    -------
    G : Tensor (..., n, r) — colonnes générateurs
    B : Tensor (..., n, r) — colonnes co-générateurs (Bᵀ apparaît dans la décomp)
    eps : erreur relative reconstruction
    """
    nabla = sylvester_displacement(A, M, N)  # (..., n, n)
    U, S, Vh = torch.linalg.svd(nabla, full_matrices=False)
    r = min(r, S.shape[-1])
    sqrt_S = S[..., :r].clamp_min(0.0).sqrt()
    G = U[..., :, :r] * sqrt_S.unsqueeze(-2)  # (..., n, r)
    B = Vh[..., :r, :].transpose(-2, -1) * sqrt_S.unsqueeze(-2)  # (..., n, r)
    # Erreur relative
    recon = G @ B.transpose(-2, -1)
    nabla_norm = nabla.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
    err = (nabla - recon).flatten(start_dim=-2).norm(dim=-1)
    eps = float((err / nabla_norm).float().median().item())
    return G, B, eps


def displacement_residual_norm(
    A: torch.Tensor, M: torch.Tensor, N: torch.Tensor,
    G: torch.Tensor, B: torch.Tensor,
) -> float:
    """Calcule ‖∇_{M, N}(A) − G Bᵀ‖_F / ‖∇(A)‖_F."""
    nabla = sylvester_displacement(A, M, N)
    recon = G @ B.transpose(-2, -1)
    nabla_norm = nabla.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
    err = (nabla - recon).flatten(start_dim=-2).norm(dim=-1)
    return float((err / nabla_norm).float().median().item())


def shift_down_operator(n: int, dtype: torch.dtype = torch.float64,
                       device: str = "cpu") -> torch.Tensor:
    """Shift-down Z[i, j] = 1 si i = j+1, 0 sinon. Toeplitz nabla operator."""
    Z = torch.zeros(n, n, dtype=dtype, device=device)
    if n > 1:
        idx = torch.arange(n - 1, device=device)
        Z[idx + 1, idx] = 1.0
    return Z
