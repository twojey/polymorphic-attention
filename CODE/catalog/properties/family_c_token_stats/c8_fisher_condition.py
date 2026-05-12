"""
c8_fisher_condition.py — Property C8 : conditionnement de la matrice de Fisher.

Spec : DOC/CATALOGUE §C8.

Pour F = diag(p) − p pᵀ (Fisher d'une distrib softmax), les eigenvalues
non-nulles sont {p_i} (en bonne approximation). Le conditionnement est :

    κ_F = λ_max(F) / λ_min⁺(F) ≈ max p / min p (parmi p > 0)

Reliée à la "facilité de l'inférence" : κ_F grand → ligne softmax peu
informative dans certaines directions (e.g. distrib quasi-one-hot).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C8FisherCondition(Property):
    """C8 — nombre de conditionnement κ(F) par ligne softmax (proxy max p / min p)."""

    name = "C8_fisher_condition"
    family = "C"
    cost_class = 1
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

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum
        P_safe = P.clamp_min(self.eps_floor)

        max_p = P_safe.max(dim=-1).values  # (B, H, N)
        # min p, mais en excluant les "vraies" zéros - on prend min de P_safe
        min_p = P_safe.min(dim=-1).values  # (B, H, N)
        kappa = max_p / min_p  # (B, H, N)
        log_kappa = kappa.log10()

        log_k_flat = log_kappa.float().flatten()
        return {
            "log10_kappa_fisher_median": float(log_k_flat.median().item()),
            "log10_kappa_fisher_mean": float(log_k_flat.mean().item()),
            "log10_kappa_fisher_p90": float(log_k_flat.quantile(0.90).item()),
            "fraction_well_conditioned_kappa_below_10": float(
                (kappa.flatten() < 10).float().mean().item()
            ),
            "n_rows": int(log_k_flat.numel()),
        }
