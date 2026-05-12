"""
d1_head_cosine.py — Property D1 : similarité cosinus entre têtes (pairs).

Spec : DOC/CATALOGUE §D1.

Pour chaque paire (h_1, h_2) avec h_1 < h_2, calcule cos(A[h_1].flat,
A[h_2].flat) — similarité globale entre matrices d'attention de deux têtes.

Métriques exposées :
- Médiane / mean sur toutes les paires
- Fraction de paires "quasi-identiques" (cos > 0.95) → redondance
- Fraction de paires "anti-corrélées" (cos < -0.5) → opposition

Cost 1 : matmul O(H² N²) par batch elem.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D1HeadCosine(Property):
    """D1 — cosine sim entre paires de têtes, vue 'inter-heads matrix'."""

    name = "D1_head_cosine"
    family = "D"
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
            # Pas de paires possibles si une seule tête
            return {
                "n_head_pairs": 0,
                "cosine_median": 1.0,
                "cosine_mean": 1.0,
                "fraction_redundant": 0.0,
                "fraction_anticorrelated": 0.0,
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Flatten matrices : (B, H, N*N)
        flat = A_work.flatten(start_dim=-2)
        # Normalise par norme L2 par head
        norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        flat_n = flat / norms  # (B, H, N²) unit-norm

        # Cosine matrix C : (B, H, H) où C[b, h1, h2] = <flat_n[b,h1], flat_n[b,h2]>
        cos_mat = torch.einsum("bhi,bgi->bhg", flat_n, flat_n)

        # Garder les paires h1 < h2
        triu_indices = torch.triu_indices(H, H, offset=1)
        pair_cos = cos_mat[:, triu_indices[0], triu_indices[1]]  # (B, n_pairs)

        cos_flat = pair_cos.float().flatten()
        return {
            "cosine_median": float(cos_flat.median().item()),
            "cosine_mean": float(cos_flat.mean().item()),
            "cosine_min": float(cos_flat.min().item()),
            "cosine_max": float(cos_flat.max().item()),
            "fraction_redundant": float((cos_flat > 0.95).float().mean().item()),
            "fraction_anticorrelated": float((cos_flat < -0.5).float().mean().item()),
            "n_head_pairs": int(triu_indices.shape[1] * B),
            "n_matrices": int(B * H),
        }
