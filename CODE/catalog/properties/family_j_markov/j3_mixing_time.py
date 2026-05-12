"""
j3_mixing_time.py — Property J3 : temps de mélange Markov.

Spec : DOC/CATALOGUE §J3 "t_mix(ε) = min t : ‖A^t − π‖_TV < ε".

Pour une chaîne de Markov stationnaire π, le mixing time t_mix(ε) est
le plus petit t tel que la distance de variation totale (TV) entre
chaque ligne de A^t et π soit ≤ ε.

Algorithme :
1. Estimer π via power iteration (cf. K3)
2. Calculer A^t itérativement et mesurer max_i ‖A^t[i,:] − π‖_TV
3. Retourner le premier t où max distance < ε pour chaque ε de la grille

Si pas de convergence avant max_iter, retourne max_iter (saturation).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class J3MixingTime(Property):
    """J3 — temps de mélange t_mix(ε) à plusieurs seuils ε."""

    name = "J3_mixing_time"
    family = "J"
    cost_class = 4  # matmuls iter + estimate pi
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        epsilons: tuple[float, ...] = (0.5, 0.25, 0.10, 0.01),
        max_iter: int = 64,
        eps_floor: float = 1e-30,
    ) -> None:
        for e in epsilons:
            if not 0.0 < e < 1.0:
                raise ValueError(f"epsilon {e} doit être ∈ (0, 1)")
        self.epsilons = tuple(sorted(epsilons, reverse=True))  # 0.5, 0.25, ...
        self.max_iter = max_iter
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
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum  # (B, H, N, N)

        # Estimer π via power iter rapide
        pi = torch.ones(B, H, N, device=A_work.device, dtype=A_work.dtype) / N
        for _ in range(min(50, self.max_iter)):
            new_pi = torch.einsum("bhi,bhij->bhj", pi, P)
            new_pi = new_pi / new_pi.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
            if (new_pi - pi).abs().sum(dim=-1).max() < 1e-7:
                pi = new_pi
                break
            pi = new_pi

        # t_mix par batch elem, par ε. Init = max_iter (= "saturé").
        t_mix = {
            e: torch.full(
                (B, H), float(self.max_iter), device=A_work.device, dtype=A_work.dtype
            )
            for e in self.epsilons
        }
        unreached = {e: torch.ones(B, H, dtype=torch.bool, device=A_work.device) for e in self.epsilons}

        Pk = P.clone()
        for t in range(1, self.max_iter + 1):
            if t > 1:
                Pk = Pk @ P
            # TV(Pk[i,:], π) = 0.5 · Σ |Pk[i,j] − π[j]|, max sur i
            diff = (Pk - pi.unsqueeze(-2)).abs().sum(dim=-1) * 0.5  # (B, H, N)
            max_tv = diff.max(dim=-1).values  # (B, H)

            for e in self.epsilons:
                hit = (max_tv < e) & unreached[e]
                t_mix[e] = torch.where(hit, torch.tensor(float(t)), t_mix[e])
                unreached[e] = unreached[e] & ~hit
            if all(not u.any().item() for u in unreached.values()):
                break

        results: dict[str, float | int | str | bool] = {}
        for e in self.epsilons:
            tv = t_mix[e].float().flatten()
            tag = f"{e:.2f}".replace(".", "p")
            results[f"tmix_eps_{tag}_median"] = float(tv.median().item())
            results[f"tmix_eps_{tag}_mean"] = float(tv.mean().item())
            results[f"tmix_eps_{tag}_max"] = float(tv.max().item())
            results[f"tmix_eps_{tag}_saturated_fraction"] = float(
                (tv >= self.max_iter).float().mean().item()
            )

        results["max_iter"] = self.max_iter
        results["n_matrices"] = int(B * H)
        results["seq_len"] = int(N)
        return results
