"""
p6_gramians.py — Property P6 : observabilité / contrôlabilité gramians.

Spec : DOC/CATALOGUE §P6 "gramians de contrôlabilité W_c et
observabilité W_o ; les HSV sont les racines carrées des valeurs
propres de W_c · W_o (forme équilibrée)".

V1 simplifié : pour A vue comme transition d'état (LTI sans B/C
explicites), on utilise les gramians "auto-induits" :
    W_c = Σ_k=0..K A^k · (A^k)ᵀ
    W_o = Σ_k=0..K (Aᵀ)^k · A^k
Les valeurs propres de W_c sont des proxies de la "richesse" de
l'espace contrôlable. Les valeurs propres de W_o sont des proxies de
l'observabilité.

Sortie : eigvals W_c, W_o, leur produit (HSV²), et leur condition
number — discriminants de la stabilité du système.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P6Gramians(Property):
    """P6 — gramians observabilité/contrôlabilité (proxy)."""

    name = "P6_gramians"
    family = "P"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, K_powers: int = 4, eps_floor: float = 1e-30) -> None:
        self.K = K_powers
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        I_N = torch.eye(N, dtype=A_work.dtype, device=A_work.device)
        Ak = I_N.expand_as(A_work).clone()
        W_c = torch.zeros_like(A_work)
        W_o = torch.zeros_like(A_work)
        for _ in range(self.K + 1):
            W_c = W_c + Ak @ Ak.transpose(-2, -1)
            W_o = W_o + Ak.transpose(-2, -1) @ Ak
            Ak = Ak @ A_work

        eig_c = torch.linalg.eigvalsh((W_c + W_c.transpose(-2, -1)) / 2)  # (B, H, N) tri ascendant
        eig_o = torch.linalg.eigvalsh((W_o + W_o.transpose(-2, -1)) / 2)
        eig_c = eig_c.flip(dims=(-1,))  # decr
        eig_o = eig_o.flip(dims=(-1,))

        # HSV² ≈ eigvalues de W_c · W_o (forme équilibrée approximative)
        prod = W_c @ W_o
        eig_prod = torch.linalg.eigvals(prod).real  # (B, H, N)
        eig_prod_abs = eig_prod.abs().sort(dim=-1, descending=True).values
        hsv_proxy = eig_prod_abs.clamp_min(0.0).sqrt()

        # Condition numbers
        eig_c_pos = eig_c.clamp_min(self.eps_floor)
        eig_o_pos = eig_o.clamp_min(self.eps_floor)
        kappa_c = (eig_c_pos[..., 0] / eig_c_pos[..., -1]).clamp_min(1.0).log10()
        kappa_o = (eig_o_pos[..., 0] / eig_o_pos[..., -1]).clamp_min(1.0).log10()

        return {
            "log10_kappa_controllability_median": float(kappa_c.float().median().item()),
            "log10_kappa_observability_median": float(kappa_o.float().median().item()),
            "eig_c_top_median": float(eig_c[..., 0].float().median().item()),
            "eig_o_top_median": float(eig_o[..., 0].float().median().item()),
            "hsv_proxy_top_median": float(hsv_proxy[..., 0].float().median().item()),
            "hsv_proxy_sigma2_over_sigma1_median": float(
                (hsv_proxy[..., 1] / hsv_proxy[..., 0].clamp_min(self.eps_floor)).float().median().item()
            ) if hsv_proxy.shape[-1] > 1 else 0.0,
            "K_powers": int(self.K),
            "n_matrices": int(B * H),
        }
