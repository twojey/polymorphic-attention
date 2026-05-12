"""
t3_cyclic_equivariance.py — Property T3 : équivariance au shift cyclique.

Pour A : Tn ↦ Tn invariant au shift cyclique τ (i.e. circulant), on a
A = Sₖ A Sₖᵀ pour tout k, où S est l'opérateur de shift.

Mesure : ε_τ = ‖A − Sₖ A Sₖᵀ‖_F / ‖A‖_F pour plusieurs k ∈ {1, 2, 4, N/4}.
A est circulant ssi ε_τ = 0 pour tout k.

Bornes attendues :
- Attention causale dense : ε_τ > 0 (la diagonale tronquée brise la cyclicité)
- Attention bidir RoPE pur sans data : potentiellement quasi-équivariant
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class T3CyclicEquivariance(Property):
    """T3 — distance à l'équivariance par shift cyclique."""

    name = "T3_cyclic_equivariance"
    family = "T"
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
        if N != N2 or N < 4:
            return {"skip_reason": "non-square or N<4", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)

        results: dict[str, float | int | str | bool] = {}
        # Shifts à tester
        shifts = [1, 2, max(1, N // 4)]
        per_mat_means = []
        for k in shifts:
            # roll lignes et colonnes par k
            A_shift = torch.roll(A_work, shifts=(k, k), dims=(-2, -1))
            diff = (A_work - A_shift).flatten(start_dim=-2).norm(dim=-1)
            eps = (diff / A_norm).float().flatten()
            tag = f"k{k}"
            results[f"epsilon_cyclic_{tag}_median"] = float(eps.median().item())
            per_mat_means.append(eps)

        mean_eps = torch.stack(per_mat_means).mean(dim=0)
        results["epsilon_cyclic_mean_across_shifts_median"] = float(
            mean_eps.median().item()
        )
        results["fraction_approx_circulant"] = float(
            (mean_eps < 0.10).float().mean().item()
        )
        results["shifts_tested"] = str(shifts)
        results["n_matrices"] = int(B * H)
        return results
