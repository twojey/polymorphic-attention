"""
s3_hierarchical_tucker.py — Property S3 : Hierarchical Tucker (HT) decomposition.

Spec : DOC/CATALOGUE §S3 "décomposition Hierarchical Tucker : pour
tensor 4D (B, H, N, N), on construit un arbre dyadique de splits
{B|H|N|N} et chaque nœud interne a un rang HT".

V1 simplifié sur tensor (n_examples, n_heads, N, N) :
- Arbre dyadique : (B|H), (N|N) → 2 nœuds internes + 1 racine
- Rang à chaque nœud = rang de la matricisation
  - Pour racine (BH | NN) : matricize en (BH × NN), rang
  - Pour nœud (B|H) : matricize en (B × H·NN), rang
  - Pour nœud (N|N) : matricize en (BH·N × N), rang

Sortie : rangs HT à chaque nœud + bound max + ratio compression
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff_matrix(M: torch.Tensor, theta: float, eps_floor: float) -> int:
    """Rang effectif d'une matrice 2D."""
    sigmas = torch.linalg.svdvals(M)
    s2 = sigmas.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(eps_floor)
    ratio = cumsum / total
    return int((ratio >= theta).float().argmax(dim=-1).item()) + 1


@register_property
class S3HierarchicalTucker(Property):
    """S3 — rangs Hierarchical Tucker via matricizations dyadiques."""

    name = "S3_hierarchical_tucker"
    family = "S"
    cost_class = 4
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Matricizations
        # Racine : (BH × NN)
        root_mat = A_work.reshape(B * H, N * N)
        r_root = _r_eff_matrix(root_mat, self.theta, self.eps_floor)

        # Nœud (B|H) : (B × H·NN) — slice par B
        bh_mat = A_work.reshape(B, H * N * N)
        r_bh = _r_eff_matrix(bh_mat, self.theta, self.eps_floor)

        # Nœud (N|N) : on aplatit B et H ensemble, puis sépare les 2 N
        nn_mat = A_work.reshape(B * H * N, N)
        r_nn = _r_eff_matrix(nn_mat, self.theta, self.eps_floor)

        # Mode-2 (B × HNN), mode-3 (H × BNN), mode-4 (N × BHN), mode-5 (N × BHN)
        mode_B = A_work.permute(0, 1, 2, 3).reshape(B, H * N * N)
        mode_H = A_work.permute(1, 0, 2, 3).reshape(H, B * N * N)
        mode_N1 = A_work.permute(2, 0, 1, 3).reshape(N, B * H * N)
        mode_N2 = A_work.permute(3, 0, 1, 2).reshape(N, B * H * N)

        r_mode_B = _r_eff_matrix(mode_B, self.theta, self.eps_floor)
        r_mode_H = _r_eff_matrix(mode_H, self.theta, self.eps_floor)
        r_mode_N1 = _r_eff_matrix(mode_N1, self.theta, self.eps_floor)
        r_mode_N2 = _r_eff_matrix(mode_N2, self.theta, self.eps_floor)

        # Bound max et "compression ratio" : produit / N_total
        ranks = [r_root, r_bh, r_nn, r_mode_B, r_mode_H, r_mode_N1, r_mode_N2]
        rank_max = max(ranks)
        N_total = B * H * N * N
        compression = float(rank_max) ** 4 / max(float(N_total), 1.0)

        return {
            "ht_rank_root_BH_NN": int(r_root),
            "ht_rank_B_H": int(r_bh),
            "ht_rank_N_N": int(r_nn),
            "ht_rank_mode_B": int(r_mode_B),
            "ht_rank_mode_H": int(r_mode_H),
            "ht_rank_mode_N1": int(r_mode_N1),
            "ht_rank_mode_N2": int(r_mode_N2),
            "ht_rank_max": int(rank_max),
            "ht_compression_ratio_proxy": compression,
            "n_examples": int(B),
            "n_heads": int(H),
            "seq_len": int(N),
        }
