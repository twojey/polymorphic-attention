"""
k4_modularity.py — Property K4 : modularité Newman (proxy communautés Louvain).

Spec : DOC/CATALOGUE §K4.

La modularité Q d'une partition C d'un graphe pondéré est :

    Q = (1 / 2m) Σ_{ij} [A_ij − k_i k_j / 2m] · δ(c_i, c_j)

Une partition non-triviale avec Q élevée (> 0.3) indique structure de
communautés. V1 simple : on utilise une partition naïve par seuil binaire
sur top-eigenvector du Laplacien (spectral bisection). Pas Louvain
itératif (V2), mais ça donne un ordre de grandeur.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class K4Modularity(Property):
    """K4 — modularité de la partition par bisection spectrale (proxy Louvain)."""

    name = "K4_modularity"
    family = "K"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2 or N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_sym = 0.5 * (A_work + A_work.transpose(-1, -2))

        # Bisection spectrale : 2ème vec propre (Fiedler) du Laplacien
        deg = A_sym.sum(dim=-1)
        D = torch.diag_embed(deg)
        L = D - A_sym
        # eigvecsh : retourne (eigenvalues, eigenvectors)
        eigvals, eigvecs = torch.linalg.eigh(L)
        # 2ème (Fiedler) : index 1 (0 est le constant)
        fiedler = eigvecs[..., :, 1]  # (B, H, N)
        partition = (fiedler > 0).long()  # (B, H, N), {0, 1}

        # Calculer Q
        total_weight = A_sym.sum(dim=(-1, -2)).clamp_min(self.eps_floor)  # 2m
        # k_i k_j / 2m
        k_outer = deg.unsqueeze(-1) * deg.unsqueeze(-2) / total_weight.unsqueeze(-1).unsqueeze(-1)
        modularity_kernel = A_sym - k_outer  # (B, H, N, N)
        # δ(c_i, c_j) = 1 si même communauté
        delta = (partition.unsqueeze(-1) == partition.unsqueeze(-2)).to(A_work.dtype)
        Q = (modularity_kernel * delta).sum(dim=(-1, -2)) / total_weight

        # Taille des communautés
        size_0 = (partition == 0).float().sum(dim=-1)  # (B, H)
        size_1 = (partition == 1).float().sum(dim=-1)
        balance = torch.minimum(size_0, size_1) / N  # ∈ [0, 0.5]

        Q_flat = Q.float().flatten()
        bal_flat = balance.float().flatten()
        return {
            "modularity_median": float(Q_flat.median().item()),
            "modularity_mean": float(Q_flat.mean().item()),
            "modularity_max": float(Q_flat.max().item()),
            "partition_balance_median": float(bal_flat.median().item()),
            "fraction_strong_community_Q_above_0p3": float(
                (Q_flat > 0.3).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
