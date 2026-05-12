"""
n4_prediction_agreement.py — Property N4 : accord top-1 prediction Oracle/student.

Pour chaque ligne A[b, h, t, :], comparer argmax (= "à qui le token t
attend le plus") entre Oracle et student.

Métrique : fraction de lignes où argmax_oracle == argmax_student.

Skip cleanly si ctx.metadata['student_attn'] absent.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class N4PredictionAgreement(Property):
    """N4 — accord top-1 row-wise argmax(Oracle) vs argmax(student)."""

    name = "N4_prediction_agreement"
    family = "N"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        student = ctx.metadata.get("student_attn", None) if ctx.metadata else None
        if student is None:
            return {
                "skip_reason": "student_attn missing",
                "student_available": False,
                "n_matrices": int(B * H),
            }
        S = student.to(device=ctx.device, dtype=ctx.dtype)
        if S.shape != A.shape:
            return {
                "skip_reason": f"shape mismatch {S.shape} vs {A.shape}",
                "student_available": False,
                "n_matrices": int(B * H),
            }
        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        argmax_o = A_work.argmax(dim=-1)  # (B, H, N)
        argmax_s = S.argmax(dim=-1)
        agree = (argmax_o == argmax_s).float()  # (B, H, N)

        # Top-k agreement
        top_o = A_work.topk(k=min(3, N), dim=-1).indices  # (B, H, N, k)
        top_s = S.topk(k=min(3, N), dim=-1).indices
        # set intersection size per (b, h, t)
        match_top3 = torch.zeros(B, H, N, device=A_work.device, dtype=A_work.dtype)
        for i in range(top_o.shape[-1]):
            for j in range(top_s.shape[-1]):
                match_top3 += (top_o[..., i] == top_s[..., j]).float()
        intersection_size = match_top3 / top_o.shape[-1]

        return {
            "top1_agreement_mean": float(agree.float().mean().item()),
            "top1_agreement_per_head_median": float(
                agree.mean(dim=-1).float().median().item()
            ),
            "top3_intersection_mean": float(intersection_size.float().mean().item()),
            "student_available": True,
            "n_matrices": int(B * H),
        }
