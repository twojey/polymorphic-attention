"""
u5_sparse_plus_lowrank.py — Property U5 : décomposition Sparse + Low-Rank (RPCA-light).

Spec : DOC/CATALOGUE §U5 "Sparse + low-rank".

A = L + S où L est de rang ≤ r et S est sparse. Méthode V1 simplifiée :
itération de soft-thresholding alternée :
1. L ← SVD top-r de (A - S)
2. S ← soft-threshold(A - L, τ)
Répéter quelques itérations.

V1 paramétré par rank_target=3, n_iter=5, threshold_ratio=0.05.
Retour : ratio de norme résiduelle, ‖L‖/‖A‖, ‖S‖/‖A‖, density(S).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _soft_threshold(X: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft-thresholding élément par élément : sign(x) · max(|x|-τ, 0)."""
    return torch.sign(X) * (X.abs() - tau).clamp_min(0.0)


@register_property
class U5SparsePlusLowRank(Property):
    """U5 — décomposition A ≈ L + S via PCP (Principal Component Pursuit) light."""

    name = "U5_sparse_plus_lowrank"
    family = "U"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        rank_target: int = 3,
        n_iter: int = 5,
        threshold_ratio: float = 0.05,
        eps_floor: float = 1e-30,
    ) -> None:
        if rank_target < 1:
            raise ValueError("rank_target ≥ 1")
        if n_iter < 1:
            raise ValueError("n_iter ≥ 1")
        self.rank_target = rank_target
        self.n_iter = n_iter
        self.threshold_ratio = threshold_ratio
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        r = min(self.rank_target, N, N2)

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        tau = self.threshold_ratio * A_work.abs().flatten(start_dim=-2).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)

        S = torch.zeros_like(A_work)
        L = torch.zeros_like(A_work)
        for _ in range(self.n_iter):
            # L step : SVD top-r de (A - S)
            X = A_work - S
            U, sigmas, Vh = torch.linalg.svd(X, full_matrices=False)
            sigmas_truncated = sigmas.clone()
            sigmas_truncated[..., r:] = 0.0
            L = U @ torch.diag_embed(sigmas_truncated) @ Vh
            # S step : soft-threshold(A - L, τ)
            S = _soft_threshold(A_work - L, tau.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1))

        # Résidu = A - L - S
        resid = A_work - L - S
        resid_norm = resid.flatten(start_dim=-2).norm(dim=-1)
        L_norm = L.flatten(start_dim=-2).norm(dim=-1)
        S_norm = S.flatten(start_dim=-2).norm(dim=-1)

        rel_resid = (resid_norm / A_norm).float().flatten()
        rel_L = (L_norm / A_norm).float().flatten()
        rel_S = (S_norm / A_norm).float().flatten()
        # Density of S : fraction d'entrées non-nulles
        S_density = (S.abs() > 1e-10).float().flatten(start_dim=-2).mean(dim=-1).float().flatten()

        return {
            "rel_residual_median": float(rel_resid.median().item()),
            "rel_residual_mean": float(rel_resid.mean().item()),
            "rel_lowrank_norm_median": float(rel_L.median().item()),
            "rel_sparse_norm_median": float(rel_S.median().item()),
            "S_density_median": float(S_density.median().item()),
            "rank_target": r,
            "threshold_ratio": self.threshold_ratio,
            "n_iter": self.n_iter,
            "n_matrices": int(B * H),
        }
