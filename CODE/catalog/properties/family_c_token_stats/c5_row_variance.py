"""
c5_row_variance.py — Property C5 : variance des coefficients par ligne.

Spec : DOC/00b §C5 "var(A[t,h,:])".

Pour chaque ligne d'attention A[t,:] (distribution sur les keys), calcule
la variance des coefficients :

    var(A[t,:]) = E[A_ij²] − (E[A_ij])² = (1/N) Σ A_ij² − (1/N²)

(le terme moyen est 1/N car les lignes sont stochastiques). Plus la
distribution est uniforme, plus la variance est faible (0 si exactement
uniforme). Plus elle est piquée, plus la variance est grande.

Reliée à la `purity` Σ p² = collision entropy via:
    var = Σ p²/N − 1/N²
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C5RowVariance(Property):
    """C5 — variance des coefficients par ligne (vue distributionnelle)."""

    name = "C5_row_variance"
    family = "C"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # Variance par ligne sur dim key (N2)
        mean = A_work.mean(dim=-1, keepdim=True)  # (B, H, N, 1)
        var = ((A_work - mean) ** 2).mean(dim=-1)  # (B, H, N)
        # Purity = Σ p² ; variance = purity/N − (1/N)² pour distribution stochastique
        purity = (A_work ** 2).sum(dim=-1)  # (B, H, N)

        var_flat = var.float().flatten()
        pur_flat = purity.float().flatten()

        return {
            "row_variance_median": float(var_flat.median().item()),
            "row_variance_mean": float(var_flat.mean().item()),
            "row_variance_p10": float(var_flat.quantile(0.10).item()),
            "row_variance_p90": float(var_flat.quantile(0.90).item()),
            "purity_median": float(pur_flat.median().item()),
            "purity_mean": float(pur_flat.mean().item()),
            # Purity = 1 → one-hot ; purity = 1/N → uniforme
            "fraction_purity_above_0p5": float(
                (pur_flat > 0.5).float().mean().item()
            ),
            "fraction_purity_below_2_over_N": float(
                (pur_flat < (2.0 / N2)).float().mean().item()
            ),
            "n_rows": int(var_flat.numel()),
        }
