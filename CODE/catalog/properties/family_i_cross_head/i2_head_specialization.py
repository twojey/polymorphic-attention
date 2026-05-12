"""
i2_head_specialization.py — Property I2 : spécialisation des têtes.

Spec : DOC/CATALOGUE §I2 "entropie inter-heads, cluster-ability".

Pour chaque tête h, calcule son "signature" : moyenne par batch elem
de log A[b, h, t, t'] (ou directement la moyenne A[b, h]). Puis :
- entropy des signatures sur la dim h (à quel point les têtes diffèrent
  de la moyenne globale)
- spread Frobenius entre têtes (mean inter-head distance)

Forte spécialisation → entropies hautes (chaque tête sort de la moyenne).
Forte redondance → entropies basses.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class I2HeadSpecialization(Property):
    """I2 — score de spécialisation par tête."""

    name = "I2_head_specialization"
    family = "I"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if H < 2:
            return {
                "n_heads": H,
                "specialization_score_mean": 0.0,
                "head_spread_median": 0.0,
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Moyenne par tête sur batch → "signature head" (H, N, N2)
        head_sig = A_work.mean(dim=0)
        # Moyenne globale (N, N2)
        global_mean = head_sig.mean(dim=0, keepdim=True)
        # Écart de chaque tête à la moyenne globale (H, N, N2)
        diff = head_sig - global_mean

        # Score de spécialisation : norm Frobenius / norm globale
        diff_norm = diff.flatten(start_dim=-2).norm(dim=-1)  # (H,)
        global_norm = global_mean.flatten().norm().clamp_min(1e-30)
        spec_score = diff_norm / global_norm  # (H,)
        spec_flat = spec_score.float()

        # Head spread : matrice de distances Frobenius entre paires
        sig_flat = head_sig.flatten(start_dim=-2)  # (H, N*N)
        # Pairwise euclidean distances
        diffs = sig_flat.unsqueeze(0) - sig_flat.unsqueeze(1)  # (H, H, N*N)
        pair_dist = diffs.norm(dim=-1)  # (H, H)
        # Stats sur triangle supérieur strict (paires distinctes)
        triu = torch.triu_indices(H, H, offset=1)
        pair_distances = pair_dist[triu[0], triu[1]].float()

        return {
            "specialization_score_max": float(spec_flat.max().item()),
            "specialization_score_mean": float(spec_flat.mean().item()),
            "specialization_score_min": float(spec_flat.min().item()),
            "head_spread_median": float(pair_distances.median().item()),
            "head_spread_mean": float(pair_distances.mean().item()),
            "head_spread_max": float(pair_distances.max().item()),
            "n_heads": H,
            "n_head_pairs": int(pair_distances.numel()),
            "n_matrices": int(B * H),
        }
