"""
k1_laplacian_spectrum.py — Property K1 : spectre du Laplacien graphe.

Spec : DOC/CATALOGUE §K1 (anciennement DOC/00b §K1).

L'attention A peut être vue comme la matrice d'adjacence d'un graphe pondéré
dirigé. Pour étudier sa structure topologique, on regarde le Laplacien :

    L = D − A_sym, avec A_sym = (A + Aᵀ) / 2, D = diag(Σ_j A_sym_{ij})

Le Laplacien normalisé est :

    L_norm = I − D^{-1/2} A_sym D^{-1/2}

Le spectre de L donne :
- λ_1 = 0 (par construction si A_sym ≥ 0 et connexe)
- λ_2 (Fiedler) : "algebraic connectivity", caractérise la difficulté à
  partitionner le graphe (mesure de regroupements)
- λ_max : excentricité spectrale
- Spectral gap λ_2 / λ_max : indicateur "mixing"

Cost class 3 — eigvals sur (B·H) matrices N×N.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class K1LaplacianSpectrum(Property):
    """K1 — spectre du Laplacien normalisé / non-normalisé du graphe A_sym."""

    name = "K1_laplacian_spectrum"
    family = "K"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, normalize: bool = True, eps_floor: float = 1e-12) -> None:
        self.normalize = normalize
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Symétriser : A_sym = (A + Aᵀ)/2 (option courante pour graphes dirigés)
        A_sym = 0.5 * (A_work + A_work.transpose(-1, -2))
        # Degrés
        deg = A_sym.sum(dim=-1)  # (B, H, N)
        deg_safe = deg.clamp_min(self.eps_floor)

        if self.normalize:
            # L_norm = I − D^{-1/2} A_sym D^{-1/2}
            d_inv_sqrt = deg_safe.rsqrt()  # (B, H, N)
            # outer : D^{-1/2} A D^{-1/2}
            scaled = A_sym * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
            eye = torch.eye(N, device=A_work.device, dtype=A_work.dtype)
            L = eye - scaled
        else:
            # L = D − A_sym
            D = torch.diag_embed(deg)
            L = D - A_sym

        # eigvalsh : Laplacien est symétrique semi-définie positive
        eigs = torch.linalg.eigvalsh(L)  # (B, H, N), triés croissants

        # λ_1 (= 0 si A_sym ≥ 0 connexe), λ_2 (Fiedler), λ_max
        lambda_1 = eigs[..., 0]
        lambda_2 = eigs[..., 1]
        lambda_max = eigs[..., -1]
        # Spectral gap : λ_2 / λ_max (ratio "mixing speed")
        gap = lambda_2 / lambda_max.clamp_min(self.eps_floor)

        lambda_1_flat = lambda_1.float().flatten()
        lambda_2_flat = lambda_2.float().flatten()
        lambda_max_flat = lambda_max.float().flatten()
        gap_flat = gap.float().flatten()

        # Effective spectral rank : nb eigvals > tol
        rank = (eigs > 1e-8).sum(dim=-1).float().flatten()

        return {
            "lambda_1_median": float(lambda_1_flat.median().item()),
            "lambda_2_fiedler_median": float(lambda_2_flat.median().item()),
            "lambda_2_fiedler_mean": float(lambda_2_flat.mean().item()),
            "lambda_max_median": float(lambda_max_flat.median().item()),
            "spectral_gap_median": float(gap_flat.median().item()),
            "spectral_gap_mean": float(gap_flat.mean().item()),
            "spectral_rank_median": float(rank.median().item()),
            "normalize": str(self.normalize),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
