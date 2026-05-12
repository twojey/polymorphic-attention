"""
j4_reversibility.py — Property J4 : detailed balance / réversibilité Markov.

Spec : DOC/CATALOGUE §J4 "π_i A_{ij} ≈ π_j A_{ji}".

Une chaîne de Markov stationnaire (π, A) est réversible (vérifie le
detailed balance) ssi pour tout (i, j) : π_i · A_{ij} = π_j · A_{ji}.
Équivalent : la matrice diag(π) · A est symétrique.

Métrique : ε_DB(A) = ‖diag(π) A − A^T diag(π)‖_F / ‖diag(π) A‖_F

Cohérent avec G2 sur diag(π)·A. Distingue les chaînes "réversibles" (ex:
diffusion, marches aléatoires symétriques) des "irréversibles" (ex: shift).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class J4Reversibility(Property):
    """J4 — distance au detailed balance π_i A_ij = π_j A_ji."""

    name = "J4_reversibility"
    family = "J"
    cost_class = 3  # power iter + matmuls
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, pi_max_iter: int = 50, eps_floor: float = 1e-30) -> None:
        self.pi_max_iter = pi_max_iter
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum

        # π via power iter rapide
        pi = torch.ones(B, H, N, device=A_work.device, dtype=A_work.dtype) / N
        for _ in range(self.pi_max_iter):
            new_pi = torch.einsum("bhi,bhij->bhj", pi, P)
            new_pi = new_pi / new_pi.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
            if (new_pi - pi).abs().sum(dim=-1).max() < 1e-7:
                pi = new_pi
                break
            pi = new_pi

        # M = diag(π) · A
        # Vectorisé : M[b, h, i, j] = π[b, h, i] * P[b, h, i, j]
        M = pi.unsqueeze(-1) * P  # (B, H, N, N)
        M_T = M.transpose(-1, -2)
        diff = M - M_T
        diff_norm = diff.flatten(start_dim=-2).norm(dim=-1)
        M_norm = M.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        eps_db = (diff_norm / M_norm).float().flatten()

        return {
            "epsilon_detailed_balance_median": float(eps_db.median().item()),
            "epsilon_detailed_balance_mean": float(eps_db.mean().item()),
            "epsilon_detailed_balance_p90": float(eps_db.quantile(0.90).item()),
            "fraction_quasi_reversible_below_0p10": float(
                (eps_db < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
