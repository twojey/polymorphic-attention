"""
v4_commutator_norm.py — Property V4 : norme du commutateur cross-head.

Pour deux têtes h1, h2 : commutateur [A_{h1}, A_{h2}] = A_{h1} A_{h2} − A_{h2} A_{h1}.

Si A est un opérateur compact "scalaire" (proportionnel à I), commutateur = 0.
Sinon, ‖[A_{h1}, A_{h2}]‖_F mesure la non-commutativité, signature
d'irréductibilité de l'attention multi-tête.

Pour ASP : si commutateurs petits ⇒ têtes "presque diagonales dans une base
commune" ⇒ multi-head est dégénéré.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class V4CommutatorNorm(Property):
    """V4 — ‖[A_{h1}, A_{h2}]‖_F / (‖A_{h1}‖ · ‖A_{h2}‖) cross-head."""

    name = "V4_commutator_norm"
    family = "V"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30, max_pairs: int = 12) -> None:
        self.eps_floor = eps_floor
        self.max_pairs = max_pairs

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if H < 2 or N != N2:
            return {"skip_reason": "H<2 or non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Norme par tête
        norms = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)  # (B, H)

        triu = torch.triu_indices(H, H, offset=1)
        n_pairs_total = triu.shape[1]
        if n_pairs_total > self.max_pairs:
            triu = triu[:, : self.max_pairs]
        n_pairs = triu.shape[1]

        # Pour chaque paire, calcul commutator
        ratios = []
        for p in range(n_pairs):
            h1 = int(triu[0, p].item())
            h2 = int(triu[1, p].item())
            A1 = A_work[:, h1]
            A2 = A_work[:, h2]
            comm = A1 @ A2 - A2 @ A1  # (B, N, N)
            c_norm = comm.flatten(start_dim=-2).norm(dim=-1)
            denom = (norms[:, h1] * norms[:, h2]).clamp_min(self.eps_floor)
            ratios.append((c_norm / denom).float())

        if not ratios:
            return {"skip_reason": "no pairs", "n_matrices": int(B * H)}
        all_r = torch.cat(ratios)
        return {
            "commutator_norm_median": float(all_r.median().item()),
            "commutator_norm_mean": float(all_r.mean().item()),
            "commutator_norm_max": float(all_r.max().item()),
            "fraction_commuting_below_0p05": float(
                (all_r < 0.05).float().mean().item()
            ),
            "n_pairs_evaluated": int(n_pairs),
            "n_matrices": int(B * H),
        }
