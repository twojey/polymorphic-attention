"""
n2_preservation.py — Property N2 : préservation propriétés post-distillation.

Spec : DOC/CATALOGUE §N2 "fraction d'invariants (rang effectif, entropie,
sparsité) qui sont préservés à ε près entre Oracle et student".

Nécessite ctx.metadata["student_attn"]. Si absent → skip cleanly.

Calcule plusieurs summary stats sur Oracle puis sur student, et la
distance relative pour chacune. Retourne le pourcentage d'invariants
préservés à 10 %.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _r_eff(A: torch.Tensor, theta: float = 0.99, eps_floor: float = 1e-30) -> torch.Tensor:
    s = torch.linalg.svdvals(A)
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = cumsum[..., -1:].clamp_min(eps_floor)
    ratio = cumsum / total
    return (ratio >= theta).float().argmax(dim=-1) + 1


def _spectral_entropy(A: torch.Tensor, eps_floor: float = 1e-30) -> torch.Tensor:
    s = torch.linalg.svdvals(A)
    s2 = s.pow(2)
    total = s2.sum(dim=-1, keepdim=True).clamp_min(eps_floor)
    p = (s2 / total).clamp_min(eps_floor)
    return -(p * p.log()).sum(dim=-1)


def _sparse_frac(A: torch.Tensor, eps_floor: float = 1e-30) -> torch.Tensor:
    A_abs = A.abs()
    flat = A_abs.flatten(start_dim=-2)
    max_per = flat.amax(dim=-1, keepdim=True).clamp_min(eps_floor)
    return (flat < 0.05 * max_per).float().mean(dim=-1)


@register_property
class N2Preservation(Property):
    """N2 — fraction d'invariants préservés Oracle → student."""

    name = "N2_preservation"
    family = "N"
    cost_class = 2
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, tolerance: float = 0.10, eps_floor: float = 1e-30) -> None:
        self.tolerance = tolerance
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

        O = A.to(device=ctx.device, dtype=ctx.dtype)

        # 3 invariants
        r_O = _r_eff(O).float()
        r_S = _r_eff(S).float()
        H_O = _spectral_entropy(O).float()
        H_S = _spectral_entropy(S).float()
        sp_O = _sparse_frac(O).float()
        sp_S = _sparse_frac(S).float()

        def _rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a - b).abs() / a.abs().clamp_min(self.eps_floor)

        rel_r = _rel(r_O, r_S).flatten()
        rel_H = _rel(H_O, H_S).flatten()
        rel_sp = _rel(sp_O, sp_S).flatten()

        pres_r = (rel_r < self.tolerance).float().mean()
        pres_H = (rel_H < self.tolerance).float().mean()
        pres_sp = (rel_sp < self.tolerance).float().mean()
        overall = (pres_r + pres_H + pres_sp) / 3

        return {
            "preservation_r_eff": float(pres_r.item()),
            "preservation_spectral_entropy": float(pres_H.item()),
            "preservation_sparse_frac": float(pres_sp.item()),
            "preservation_overall": float(overall.item()),
            "rel_err_r_eff_median": float(rel_r.median().item()),
            "rel_err_spectral_entropy_median": float(rel_H.median().item()),
            "rel_err_sparse_frac_median": float(rel_sp.median().item()),
            "student_available": True,
            "tolerance": self.tolerance,
            "n_matrices": int(B * H),
        }
