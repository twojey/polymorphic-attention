"""
s4_cp_rank_proxy.py — Property S4 : proxy de CP-rank par ALS limité.

Le CP-rank (Canonical Polyadic) d'un tenseur T est le nombre minimum de
termes rank-1 nécessaire pour décomposer T :
    T = Σ_{r=1..R} λ_r · a_r ⊗ b_r ⊗ c_r ⊗ d_r

Calculer le CP-rank exact est NP-hard. Proxy : on lance ALS (Alternating
Least Squares) avec R = 1, 2, 4, 8 et on rapporte la fraction d'énergie
résiduelle ‖T − T̂_R‖_F / ‖T‖_F.

Diagnostic d'approximabilité CP. Faible résidu pour petit R = T bien
décomposable en somme outer products.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _als_step(
    T: torch.Tensor, factors: list[torch.Tensor], mode: int
) -> torch.Tensor:
    """Update factor[mode] via ALS least-squares."""
    n_dims = T.ndim
    # Khatri-Rao of all factors except mode
    others = [f for i, f in enumerate(factors) if i != mode]
    # Sequential KR product
    KR = others[0]
    for f in others[1:]:
        KR = torch.einsum("ir,jr->ijr", KR, f).reshape(-1, KR.shape[-1])
    # Mode-`mode` unfolding
    perm = [mode] + [i for i in range(n_dims) if i != mode]
    T_unf = T.permute(*perm).reshape(T.shape[mode], -1)  # (n_mode, prod_others)
    # Solve via least squares : factor[mode] = T_unf @ KR @ (KR.T @ KR)^-1
    gram = KR.transpose(-1, -2) @ KR
    new_factor = T_unf @ KR @ torch.linalg.pinv(gram)
    return new_factor


@register_property
class S4CpRankProxy(Property):
    """S4 — résidus ALS pour R ∈ {1, 2, 4, 8} = proxy CP-rank."""

    name = "S4_cp_rank_proxy"
    family = "S"
    cost_class = 4
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        ranks: tuple[int, ...] = (1, 2, 4, 8),
        n_iter: int = 20,
        eps_floor: float = 1e-30,
        seed_offset: int = 0,
    ) -> None:
        self.ranks = tuple(sorted(ranks))
        self.n_iter = n_iter
        self.eps_floor = eps_floor
        self.seed_offset = seed_offset

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        norm_T = A_work.flatten().norm().clamp_min(self.eps_floor)
        seed = int(ctx.metadata.get("seed", 0)) + self.seed_offset
        g = torch.Generator(device="cpu").manual_seed(seed)
        dims = [B, H, N, N2]

        residuals = {}
        for R in self.ranks:
            # Init factors random
            factors = [
                torch.randn(d, R, generator=g, dtype=A_work.dtype).to(A_work.device)
                for d in dims
            ]
            # ALS iterations
            for _ in range(self.n_iter):
                for mode in range(4):
                    factors[mode] = _als_step(A_work, factors, mode)
            # Reconstruct T_hat = Σ_r f0[:, r] ⊗ f1[:, r] ⊗ f2[:, r] ⊗ f3[:, r]
            T_hat = torch.einsum(
                "ir,jr,kr,lr->ijkl",
                factors[0], factors[1], factors[2], factors[3]
            )
            resid = (A_work - T_hat).flatten().norm() / norm_T
            residuals[R] = float(resid.float().item())

        out: dict[str, float | int | str | bool] = {}
        for R, v in residuals.items():
            out[f"cp_residual_R{R}"] = v
        out["cp_residual_min"] = float(min(residuals.values()))
        out["cp_first_rank_below_0p20"] = int(
            min((r for r, v in residuals.items() if v < 0.20), default=-1)
        )
        out["n_matrices"] = int(B * H)
        return out
