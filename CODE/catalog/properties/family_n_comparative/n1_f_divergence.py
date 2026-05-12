"""
n1_f_divergence.py — Property N1 : F-divergence Oracle vs student.

Spec : DOC/CATALOGUE §N1 "D_KL(A_oracle ‖ A_student) + D_JS, par ligne
(distribution)".

Nécessite ctx.metadata["student_attn"] : Tensor (B, H, N, N) du student.
Si absent → skip cleanly.

Sortie : KL divergence ligne-par-ligne (médiane), JS divergence, χ².
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class N1FDivergence(Property):
    """N1 — KL/JS/χ² Oracle vs student par ligne d'attention."""

    name = "N1_f_divergence"
    family = "N"
    cost_class = 2
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-12) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        student = ctx.metadata.get("student_attn", None) if ctx.metadata else None
        if student is None:
            return {
                "n_matrices": int(B * H),
                "skip_reason": "student_attn missing — requires distilled model phase 3",
                "student_available": False,
            }
        S = student.to(device=ctx.device, dtype=ctx.dtype)
        # Aligner shapes par layer si nécessaire ; on suppose même shape pour V1
        if S.shape != A.shape:
            return {
                "n_matrices": int(B * H),
                "skip_reason": f"student shape {S.shape} != oracle {A.shape}",
                "student_available": False,
            }

        P = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        Q = S.clamp_min(self.eps_floor)
        # KL ligne-par-ligne
        kl = (P * (P / Q).log()).sum(dim=-1)  # (B, H, N)
        # JS : 0.5 KL(P||M) + 0.5 KL(Q||M) avec M = (P+Q)/2
        M = (P + Q) / 2
        kl_pm = (P * (P / M).log()).sum(dim=-1)
        kl_qm = (Q * (Q / M).log()).sum(dim=-1)
        js = 0.5 * kl_pm + 0.5 * kl_qm
        # χ² : Σ (P - Q)² / Q
        chi2 = ((P - Q) ** 2 / Q).sum(dim=-1)

        return {
            "kl_oracle_student_median": float(kl.float().median().item()),
            "kl_oracle_student_p90": float(kl.float().quantile(0.90).item()),
            "js_oracle_student_median": float(js.float().median().item()),
            "chi2_oracle_student_median": float(chi2.float().median().item()),
            "fraction_kl_below_0p01": float((kl < 0.01).float().mean().item()),
            "student_available": True,
            "n_matrices": int(B * H),
        }
