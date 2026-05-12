"""
b8_sylvester_rank.py — Property B8 : rang Sylvester (GCD polynomial implicite).

Spec : DOC/CATALOGUE §B8.

Pour deux polynômes p, q de degrés n et m respectivement, leur **matrice de
Sylvester** S(p, q) est (n+m) × (n+m) et son rang détermine le degré du
PGCD(p, q). En particulier : rang(S) < n+m ⟺ p et q ont un facteur commun.

V1 simple : on considère chaque ligne de A[t,:] comme coefficients d'un
polynôme. Pour des paires de lignes consécutives, on construit la matrice
de Sylvester et on calcule son rang effectif. Un rang typiquement bas
indique structure polynomiale partagée entre lignes consécutives.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _sylvester_matrix(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Matrice de Sylvester de deux polynômes (coefficients en ordre croissant ou décroissant)."""
    n = p.shape[-1] - 1  # degré de p
    m = q.shape[-1] - 1  # degré de q
    size = n + m
    S = torch.zeros(*p.shape[:-1], size, size, device=p.device, dtype=p.dtype)
    for i in range(m):
        S[..., i, i: i + n + 1] = p
    for j in range(n):
        S[..., m + j, j: j + m + 1] = q
    return S


@register_property
class B8SylvesterRank(Property):
    """B8 — rang effectif de la matrice Sylvester pour lignes consécutives."""

    name = "B8_sylvester_rank"
    family = "B"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30, max_seq_len: int = 16) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor
        self.max_seq_len = max_seq_len  # cap pour éviter Sylvester O((2N)²)

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N2 > self.max_seq_len:
            # Sub-sample pour limiter coût
            stride = N2 // self.max_seq_len
            A_sub = A[..., :, ::stride][..., :self.max_seq_len]
        else:
            A_sub = A
        N_eff = A_sub.shape[-1]
        if N < 2 or N_eff < 2:
            return {"n_matrices": int(B * H), "skip_reason": "too small"}

        A_work = A_sub.to(device=ctx.device, dtype=ctx.dtype)
        # Pour chaque paire (ligne t, ligne t+1), construire Sylvester
        ranks: list[torch.Tensor] = []
        for t in range(min(N - 1, 8)):  # limite à 8 paires pour le coût
            p = A_work[..., t, :]
            q = A_work[..., t + 1, :]
            S = _sylvester_matrix(p, q)  # (..., 2*N_eff-2, 2*N_eff-2) wait
            # Actually size = n+m = (N_eff-1) + (N_eff-1) = 2(N_eff-1)
            sigmas = torch.linalg.svdvals(S)
            s2 = sigmas.pow(2)
            cumsum = s2.cumsum(dim=-1)
            total = cumsum[..., -1:].clamp_min(self.eps_floor)
            ratio = cumsum / total
            r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1
            ranks.append(r_eff.float().flatten())

        all_ranks = torch.cat(ranks)
        full_rank = 2 * (N_eff - 1)
        return {
            "sylvester_rank_eff_median": float(all_ranks.median().item()),
            "sylvester_rank_eff_mean": float(all_ranks.mean().item()),
            "sylvester_rank_eff_max": float(all_ranks.max().item()),
            "fraction_rank_deficient": float(
                (all_ranks < full_rank).float().mean().item()
            ),
            "full_rank_theoretical": full_rank,
            "n_pairs": int(min(N - 1, 8)),
            "n_matrices": int(B * H),
            "seq_len_effective": int(N_eff),
        }
