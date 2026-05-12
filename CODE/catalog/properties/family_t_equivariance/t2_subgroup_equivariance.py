"""
t2_subgroup_equivariance.py — Property T2 : G-équivariance sous-groupes spécifiques.

Spec : DOC/CATALOGUE §T2.

Plus discriminant que T1 (test stochastique de permutation aléatoire),
T2 teste l'équivariance sous des sous-groupes structurés :
- Reflection : π = reverse (i → N-1-i)
- Cyclic shift : π = (i → (i+k) mod N) pour k ∈ {1, N/2}
- Block-permutation (échange de 2 blocs égaux)

Pour chaque sous-groupe, ε_G(A) = ‖A − π·A·π^T‖_F / ‖A‖_F.

V1 simple : test 3 sous-groupes standards et rapporte ε par groupe + min global.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class T2SubgroupEquivariance(Property):
    """T2 — équivariance sous sous-groupes structurés (reverse, cyclic shift)."""

    name = "T2_subgroup_equivariance"
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

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)

        results: dict[str, float | int | str | bool] = {}
        all_eps_per_mat: list[torch.Tensor] = []

        # 1. Reverse symmetry
        perm = torch.arange(N - 1, -1, -1, device=A_work.device)
        A_rev = A_work[..., perm, :][..., :, perm]
        eps_rev = (A_work - A_rev).flatten(start_dim=-2).norm(dim=-1) / A_norm
        results["eps_reverse_median"] = float(eps_rev.float().median().item())
        all_eps_per_mat.append(eps_rev.float())

        # 2. Cyclic shift k=1
        perm = torch.cat([torch.arange(1, N, device=A_work.device), torch.tensor([0], device=A_work.device)])
        A_shift1 = A_work[..., perm, :][..., :, perm]
        eps_shift1 = (A_work - A_shift1).flatten(start_dim=-2).norm(dim=-1) / A_norm
        results["eps_cyclic_shift1_median"] = float(eps_shift1.float().median().item())
        all_eps_per_mat.append(eps_shift1.float())

        # 3. Cyclic shift k=N/2
        if N >= 2:
            k_half = N // 2
            perm = torch.cat([
                torch.arange(k_half, N, device=A_work.device),
                torch.arange(0, k_half, device=A_work.device),
            ])
            A_shift_half = A_work[..., perm, :][..., :, perm]
            eps_shift_half = (A_work - A_shift_half).flatten(start_dim=-2).norm(dim=-1) / A_norm
            results["eps_cyclic_shift_half_median"] = float(eps_shift_half.float().median().item())
            all_eps_per_mat.append(eps_shift_half.float())

        # Min ε sur les sous-groupes
        min_eps = torch.stack(all_eps_per_mat).min(dim=0).values  # (B, H)
        results["eps_best_subgroup_median"] = float(min_eps.flatten().median().item())
        results["fraction_close_to_some_subgroup_below_0p10"] = float(
            (min_eps.flatten() < 0.10).float().mean().item()
        )
        results["n_subgroups_tested"] = len(all_eps_per_mat)
        results["n_matrices"] = int(B * H)
        return results
