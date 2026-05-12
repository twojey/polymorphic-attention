"""
c1_kl_baseline.py — Property C1 : KL vs baseline empirique.

Spec : DOC/CATALOGUE §C1 "KL local vs baseline empirique" — Sprint 1.5.

Baseline empirique : moyenne A[t, :] sur les positions query (= distribution
moyenne d'attention "globale"). KL(A[t,:] ‖ baseline_emp) mesure à quel
point la ligne t s'écarte de la moyenne.

À distinguer de C2 (KL vs uniform théorique) : C1 s'adapte aux fréquences
observées dans la séquence.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C1KLBaseline(Property):
    """C1 — KL(A[t,:] ‖ baseline empirique) par ligne."""

    name = "C1_kl_baseline"
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

        # Baseline empirique : moyenne sur dim query (N) → (B, H, N2)
        baseline = P.mean(dim=-2, keepdim=True).clamp_min(self.eps_floor)
        P_safe = P.clamp_min(self.eps_floor)

        # KL par ligne : Σ_j P[t, j] log(P[t, j] / baseline[j])
        log_ratio = P_safe.log() - baseline.log()
        kl = (P * log_ratio).sum(dim=-1)  # (B, H, N), nats

        kl_flat = kl.float().flatten()
        kl_flat = kl_flat[torch.isfinite(kl_flat)]

        return {
            "kl_baseline_median": float(kl_flat.median().item()) if kl_flat.numel() else float("nan"),
            "kl_baseline_mean": float(kl_flat.mean().item()) if kl_flat.numel() else float("nan"),
            "kl_baseline_p10": float(kl_flat.quantile(0.10).item()) if kl_flat.numel() else float("nan"),
            "kl_baseline_p90": float(kl_flat.quantile(0.90).item()) if kl_flat.numel() else float("nan"),
            "fraction_high_kl_above_1": float(
                (kl_flat > 1.0).float().mean().item()
            ),
            "n_rows": int(kl_flat.numel()),
        }
