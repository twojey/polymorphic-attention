"""
j1_markov_convergence.py — Property J1 : convergence A^k vers rang-1.

Spec : DOC/CATALOGUE §J1.

Si A est row-stochastique irréductible apériodique, alors A^k → 1 · πᵀ
quand k → ∞ (Perron-Frobenius). On mesure :

    `markov_convergence_kth(k) = ‖A^k − rank1(A^k)‖_F / ‖A^k‖_F`

où rank1(M) = u·σ·vᵀ est l'approximation rank-1 via SVD top-1.

À mesure que k croît, ce résidu décroît exponentiellement avec un taux
gouverné par λ_2 (2ème valeur propre en module). C'est le test "matrice
de transition Markov pure".
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class J1MarkovConvergence(Property):
    """J1 — convergence A^k vers rang-1 (test Markov-ness)."""

    name = "J1_markov_convergence"
    family = "J"
    cost_class = 3  # matmuls + SVD top-1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self, powers: tuple[int, ...] = (2, 4, 8, 16), eps_floor: float = 1e-30
    ) -> None:
        for p in powers:
            if p < 1:
                raise ValueError(f"power {p} doit être ≥ 1")
        self.powers = tuple(sorted(set(powers)))
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Renormalise rows pour assurer Markov
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum

        results: dict[str, float | int | str | bool] = {}
        max_k = max(self.powers)
        Pk = P.clone()
        target_powers = set(self.powers)
        # Stocke aussi rank1 residuals
        for k in range(1, max_k + 1):
            if k > 1:
                Pk = Pk @ P
            if k in target_powers:
                # SVD top-1 par batch elem
                # full_matrices=False : faster
                U, S, Vh = torch.linalg.svd(Pk, full_matrices=False)
                # Rank-1 approx
                top_outer = (U[..., :, :1] * S[..., :1].unsqueeze(-2)) @ Vh[..., :1, :]
                resid = Pk - top_outer
                rel = (
                    resid.flatten(start_dim=-2).norm(dim=-1)
                    / Pk.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
                ).float().flatten()
                # Spectral gap : sigma_2 / sigma_1
                gap = (S[..., 1] / S[..., 0].clamp_min(self.eps_floor)).float().flatten()
                results[f"rank1_residual_k{k}_median"] = float(rel.median().item())
                results[f"rank1_residual_k{k}_mean"] = float(rel.mean().item())
                results[f"sigma2_over_sigma1_k{k}_median"] = float(gap.median().item())

        # Test si convergence rate respecte un decay log-linéaire :
        # rate = (1/k) log(residual_k) — proxy mixing time spectral
        if len(self.powers) >= 2:
            largest_k = max(self.powers)
            smallest_k = min(self.powers)
            r_max = results.get(f"rank1_residual_k{largest_k}_median", 0.0)
            r_min = results.get(f"rank1_residual_k{smallest_k}_median", 1.0)
            if r_min > self.eps_floor and r_max > self.eps_floor:
                rate = (
                    (-torch.tensor(r_max).log() + torch.tensor(r_min).log())
                    / (largest_k - smallest_k)
                )
                results["mixing_rate_log_decay"] = float(rate.item())

        results["n_matrices"] = int(B * H)
        results["seq_len"] = int(N)
        results["max_power"] = max_k
        return results
