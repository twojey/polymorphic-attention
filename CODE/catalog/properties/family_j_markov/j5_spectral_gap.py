"""
j5_spectral_gap.py — Property J5 : trou spectral Markov.

Pour une chaîne de Markov stochastique P (lignes sommant à 1), le trou
spectral γ = 1 − |λ_2| où λ_2 est la deuxième valeur propre (en module).

γ proche de 1 : chaîne mixante rapidement (t_mix ≈ log(1/ε)/γ).
γ proche de 0 : chaîne lente, possiblement quasi-réductible.

Lié à J3 mixing time : 1/γ borne supérieure de t_mix. Mesurer les deux
permet de vérifier l'inégalité (sanity check).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class J5SpectralGap(Property):
    """J5 — trou spectral γ = 1 − |λ_2| de la chaîne de Markov."""

    name = "J5_spectral_gap"
    family = "J"
    cost_class = 3  # eigvals batched
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Re-normalize lignes au cas où
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum
        # eigvals non symétrique
        try:
            eig = torch.linalg.eigvals(P)  # complex (B, H, N)
        except Exception:
            return {"skip_reason": "eigvals failed", "n_matrices": int(B * H)}
        mags = eig.abs()  # (B, H, N)
        # Tri descendant
        mags_sorted, _ = mags.sort(dim=-1, descending=True)
        lam1 = mags_sorted[..., 0]
        lam2 = mags_sorted[..., 1] if N >= 2 else mags_sorted[..., 0]
        gap = (1.0 - lam2).clamp_min(0.0)
        gap_flat = gap.float().flatten()
        lam2_flat = lam2.float().flatten()
        lam1_flat = lam1.float().flatten()

        return {
            "spectral_gap_median": float(gap_flat.median().item()),
            "spectral_gap_mean": float(gap_flat.mean().item()),
            "spectral_gap_min": float(gap_flat.min().item()),
            "lambda2_abs_median": float(lam2_flat.median().item()),
            "lambda1_abs_median": float(lam1_flat.median().item()),
            "fraction_slow_mixing_gap_below_0p10": float(
                (gap_flat < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
