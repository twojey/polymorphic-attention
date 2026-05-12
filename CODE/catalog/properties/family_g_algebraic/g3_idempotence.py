"""
g3_idempotence.py — Property G3 : idempotence.

Spec : DOC/CATALOGUE §G3.

ε_idem(A) = ‖A² − A‖_F / ‖A‖_F

Une matrice idempotente (A² = A) est un projecteur, propriété rare et
spécifique. Pour attention dense softmax, A² est loin de A en général.

Métrique alternative : `convergence_rate` = ‖A^k − A^(k+1)‖_F pour k=2-4
mesure la stabilité dynamique (cf. Markov mixing).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G3Idempotence(Property):
    """G3 — distance à un projecteur (A² = A) + rate de convergence A^k."""

    name = "G3_idempotence"
    family = "G"
    cost_class = 2  # k matmuls O(N³)
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, max_power: int = 4) -> None:
        if max_power < 2:
            raise ValueError("max_power doit être ≥ 2")
        self.max_power = max_power

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)

        # A^2
        A2 = A_work @ A_work
        residual = A2 - A_work
        eps_idem = (
            residual.flatten(start_dim=-2).norm(dim=-1) / A_norm
        ).float().flatten()

        results: dict[str, float | int | str | bool] = {
            "epsilon_idempotence_median": float(eps_idem.median().item()),
            "epsilon_idempotence_mean": float(eps_idem.mean().item()),
            "epsilon_idempotence_min": float(eps_idem.min().item()),
            "epsilon_idempotence_max": float(eps_idem.max().item()),
            "fraction_quasi_idempotent_0p10": float(
                (eps_idem < 0.10).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }

        # ‖A^k − A^(k+1)‖_F / ‖A^k‖_F pour k=2, 3, ... — convergence
        Ak = A2
        for k in range(2, self.max_power):
            Ak_next = Ak @ A_work
            diff = (Ak_next - Ak).flatten(start_dim=-2).norm(dim=-1)
            Ak_norm = Ak.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
            rel = (diff / Ak_norm).float().flatten()
            results[f"power_diff_k{k}_median"] = float(rel.median().item())
            results[f"power_diff_k{k}_mean"] = float(rel.mean().item())
            Ak = Ak_next

        return results
