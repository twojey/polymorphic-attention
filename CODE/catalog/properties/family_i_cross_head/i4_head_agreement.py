"""
i4_head_agreement.py — Property I4 : accord cross-head sur le top-1 voisin.

Pour chaque query t, on note j*_h = argmax_j A[h, t, j] (tête h). On mesure
la fraction de paires (h1, h2) qui s'accordent : j*_{h1} = j*_{h2}.

Agreement = 1 ⇒ toutes têtes convergent ⇒ multi-head ≈ single-head dégénéré.
Agreement = 0 ⇒ têtes orthogonales ⇒ multi-head utile.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class I4HeadAgreement(Property):
    """I4 — fraction de paires de têtes qui s'accordent sur argmax row-wise."""

    name = "I4_head_agreement"
    family = "I"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if H < 2:
            return {
                "skip_reason": "H<2",
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        argmax = A_work.argmax(dim=-1)  # (B, H, N)
        # Pairwise agreement
        triu = torch.triu_indices(H, H, offset=1)  # (2, n_pairs)
        a1 = argmax[:, triu[0]]  # (B, n_pairs, N)
        a2 = argmax[:, triu[1]]
        agree = (a1 == a2).float()  # (B, n_pairs, N)
        # Per pair mean
        per_pair = agree.mean(dim=-1)  # (B, n_pairs)
        per_pair_flat = per_pair.float().flatten()

        return {
            "head_agreement_median": float(per_pair_flat.median().item()),
            "head_agreement_mean": float(per_pair_flat.mean().item()),
            "head_agreement_max": float(per_pair_flat.max().item()),
            "fraction_pairs_strong_agreement": float(
                (per_pair_flat > 0.50).float().mean().item()
            ),
            "n_pairs_per_example": int(triu.shape[1]),
            "n_matrices": int(B * H),
        }
