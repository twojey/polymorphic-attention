"""
i1_head_diversity.py — Property I1 : diversité inter-heads.

Spec : DOC/CATALOGUE §I1 "var_h(A[ℓ,h,:,:])".

Pour chaque batch elem et position (t, t'), calcule la variance sur la dim
h des coefficients A[b, h, t, t']. Forte variance → têtes très différentes
(diversification). Faible variance → têtes redondantes.

Output : moyenne et médiane de var_h(A) sur la grille (B, t, t').
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class I1HeadDiversity(Property):
    """I1 — variance inter-heads des coefficients d'attention."""

    name = "I1_head_diversity"
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
                "var_h_median": 0.0,
                "var_h_mean": 0.0,
                "diversity_score_mean": 0.0,
                "n_matrices": int(B * H),
                "n_heads": H,
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Variance sur dim h (head)
        var_h = A_work.var(dim=1, unbiased=False)  # (B, N, N2)
        var_flat = var_h.float().flatten()

        # Score "diversité" : moyenne de var_h / (mean²+ε) — coefficient de variation²
        mean_h = A_work.mean(dim=1)  # (B, N, N2)
        cv2 = var_h / (mean_h ** 2 + 1e-12)
        cv2_flat = cv2.float().flatten()

        return {
            "var_h_median": float(var_flat.median().item()),
            "var_h_mean": float(var_flat.mean().item()),
            "var_h_p90": float(var_flat.quantile(0.90).item()),
            "diversity_score_mean": float(cv2_flat.mean().item()),
            "diversity_score_median": float(cv2_flat.median().item()),
            "n_matrices": int(B * H),
            "n_heads": H,
            "n_entries_aggregated": int(var_flat.numel()),
        }
