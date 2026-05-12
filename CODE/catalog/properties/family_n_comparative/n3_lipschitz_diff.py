"""
n3_lipschitz_diff.py — Property N3 : Lipschitz différentielle Oracle vs student.

Spec : DOC/CATALOGUE §N3 "estimation de la borne Lipschitz du
différentiel A_oracle − A_student. Si bornée, la distillation
préserve la régularité (continuité Hölder bornée)".

Nécessite ctx.metadata["student_attn"]. Si absent → skip cleanly.

V1 : pour chaque paire (i, j) intra-batch, calcule
    L_O = ‖A_O[i] − A_O[j]‖_F / ‖input_i − input_j‖
    L_S = ‖A_S[i] − A_S[j]‖_F / ‖input_i − input_j‖
    diff = |L_O - L_S|
Sortie : moyenne, max de diff (Lipschitz constant du différentiel).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class N3LipschitzDiff(Property):
    """N3 — Lipschitz constant du différentiel Oracle − student."""

    name = "N3_lipschitz_diff"
    family = "N"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, max_pairs: int = 64, eps_floor: float = 1e-6) -> None:
        self.max_pairs = max_pairs
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
        if S.shape != A.shape:
            return {
                "n_matrices": int(B * H),
                "skip_reason": f"student shape {S.shape} != oracle {A.shape}",
                "student_available": False,
            }
        if B < 2:
            return {"n_matrices": int(B * H), "skip_reason": "batch<2",
                    "student_available": False}

        O = A.to(device=ctx.device, dtype=ctx.dtype).mean(dim=1)  # (B, N, N)
        SS = S.mean(dim=1)

        # Distance d'entrée : tokens si dispo, sinon row-sums
        tokens = ctx.metadata.get("tokens", None) if ctx.metadata else None
        if tokens is not None and tokens.shape[0] == B:
            tok = tokens.to(device=ctx.device)
            d_in = (tok.unsqueeze(1) != tok.unsqueeze(0)).float().mean(dim=-1)
        else:
            rs = O.sum(dim=-1)  # use Oracle as reference
            d_in = (rs.unsqueeze(1) - rs.unsqueeze(0)).norm(dim=-1)

        # Pairwise attention distance
        d_O = (O.unsqueeze(1) - O.unsqueeze(0)).flatten(start_dim=2).norm(dim=-1)
        d_S = (SS.unsqueeze(1) - SS.unsqueeze(0)).flatten(start_dim=2).norm(dim=-1)

        iu = torch.triu_indices(B, B, offset=1)
        d_in_p = d_in[iu[0], iu[1]].float()
        d_O_p = d_O[iu[0], iu[1]].float()
        d_S_p = d_S[iu[0], iu[1]].float()

        if d_in_p.numel() > self.max_pairs:
            top = torch.topk(d_O_p, k=self.max_pairs).indices
            d_in_p = d_in_p[top]
            d_O_p = d_O_p[top]
            d_S_p = d_S_p[top]

        mask = d_in_p > self.eps_floor
        if not mask.any():
            return {
                "n_matrices": int(B * H),
                "skip_reason": "no valid input distances",
                "student_available": True,
            }
        L_O = d_O_p[mask] / d_in_p[mask]
        L_S = d_S_p[mask] / d_in_p[mask]
        diff = (L_O - L_S).abs()

        return {
            "lipschitz_diff_max": float(diff.max().item()),
            "lipschitz_diff_median": float(diff.median().item()),
            "lipschitz_diff_p90": float(diff.quantile(0.90).item()),
            "lipschitz_oracle_max": float(L_O.max().item()),
            "lipschitz_student_max": float(L_S.max().item()),
            "rel_diff_median": float(
                (diff / L_O.clamp_min(self.eps_floor)).median().item()
            ),
            "student_available": True,
            "n_pairs_valid": int(mask.sum().item()),
            "n_matrices": int(B * H),
        }
