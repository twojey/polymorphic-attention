"""
f2_temporal_stability.py — Property F2 : stabilité temporelle intra-matrice.

Spec : DOC/CATALOGUE §F2 "‖A_t − A_{t+1}‖".

Pour chaque matrice A (B, H, N, N), regarde la variation entre lignes
adjacentes A[t, :] et A[t+1, :] :

    δ_t = ‖A[t+1, :] − A[t, :]‖_2

Mesure la "régularité" de l'attention en fonction de la position query.
Une attention quasi-toeplitz (causalité régulière) aura δ_t ≈ 0 partout.
Une attention "switch" abrupt aura δ_t pique à certaines positions.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class F2TemporalStability(Property):
    """F2 — variation entre lignes adjacentes (vue 'jerk' de l'attention)."""

    name = "F2_temporal_stability"
    family = "F"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < 2:
            return {"n_matrices": int(B * H), "delta_median": 0.0}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        diff = A_work[..., 1:, :] - A_work[..., :-1, :]  # (B, H, N-1, N2)
        delta = diff.norm(dim=-1)  # (B, H, N-1), L2 par row diff
        # Normalisation par ‖A_t‖_2 (relative variation)
        ref_norm = A_work[..., :-1, :].norm(dim=-1).clamp_min(1e-30)
        rel_delta = delta / ref_norm  # (B, H, N-1)

        d_flat = delta.float().flatten()
        rel_flat = rel_delta.float().flatten()

        # Métriques per-matrix : max et mean δ_t
        max_delta_per_mat = delta.amax(dim=-1).float().flatten()  # (B*H,)
        mean_delta_per_mat = delta.mean(dim=-1).float().flatten()

        return {
            "delta_median": float(d_flat.median().item()),
            "delta_mean": float(d_flat.mean().item()),
            "delta_p90": float(d_flat.quantile(0.90).item()),
            "rel_delta_median": float(rel_flat.median().item()),
            "rel_delta_mean": float(rel_flat.mean().item()),
            "max_delta_per_mat_median": float(max_delta_per_mat.median().item()),
            "mean_delta_per_mat_median": float(mean_delta_per_mat.median().item()),
            "fraction_smooth_rel_delta_below_0p10": float(
                (rel_flat < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
            "n_transitions": int(d_flat.numel()),
        }
