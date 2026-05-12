"""
c2_kl_uniform.py — Property C2 : KL divergence vs uniforme causale.

Spec : DOC/00b §C2 "KL(A[t,:] ‖ uniform_t)".

Pour chaque ligne A[t,:] (distribution sur N positions key), compare à la
distribution uniforme. Variante causale : uniform sur les t+1 premières
positions (autoregressive) si demandée.

KL(p ‖ u) = Σ p_i log(p_i / u_i) = log(N) − H(p) si u_i = 1/N constant.
Donc KL vs uniform = log(N) − H(p) (équivalent à entropie shifted).

Variante causale : u_i = 1/(t+1) pour i ≤ t, 0 sinon. KL fini si A[t, i]=0
pour i > t (i.e., A est strictement causale).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C2KLUniform(Property):
    """C2 — KL(A[t,:] ‖ uniform). Mode causal: u_t = 1/(t+1) sur les t+1
    premières positions ; mode flat: u = 1/N uniforme."""

    name = "C2_kl_uniform"
    family = "C"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, causal: bool = False, eps_floor: float = 1e-30) -> None:
        self.causal = causal
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        # Normalisation : les lignes doivent sommer à 1 (sinon on renormalise)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum  # (B, H, N, N2)

        if self.causal:
            # u_t[i] = 1/(t+1) pour i ≤ t, 0 sinon → uniform sur préfix causal
            t_idx = torch.arange(N, device=A_work.device).view(-1, 1)  # (N, 1)
            i_idx = torch.arange(N2, device=A_work.device).view(1, -1)  # (1, N2)
            mask = (i_idx <= t_idx).to(A_work.dtype)  # (N, N2)
            t_plus_one = (t_idx + 1).to(A_work.dtype).clamp_min(1.0)
            log_uniform = -t_plus_one.log()  # log(1/(t+1)) per query position
            # KL = Σ p log p − Σ p log u, sur le support causal
            # On nullifie P hors support causal (devrait déjà être ≈ 0)
            log_P = P.clamp_min(self.eps_floor).log()
            # Σ p log p sur le support causal seulement
            entropy_neg = (P * log_P * mask).sum(dim=-1)  # (B, H, N)
            cross_entropy_neg = (P * log_uniform * mask).sum(dim=-1)  # (B, H, N)
            kl = entropy_neg - cross_entropy_neg  # (B, H, N)
        else:
            # u uniforme sur N : KL = log N − H(P)
            log_P = P.log()
            entropy = -(P * log_P).sum(dim=-1)  # (B, H, N)
            kl = math.log(N2) - entropy

        kl_flat = kl.float().flatten()
        kl_flat = kl_flat[torch.isfinite(kl_flat)]

        return {
            "kl_uniform_median": float(kl_flat.median().item()) if kl_flat.numel() else float("nan"),
            "kl_uniform_mean": float(kl_flat.mean().item()) if kl_flat.numel() else float("nan"),
            "kl_uniform_p10": float(kl_flat.quantile(0.10).item()) if kl_flat.numel() else float("nan"),
            "kl_uniform_p90": float(kl_flat.quantile(0.90).item()) if kl_flat.numel() else float("nan"),
            "causal_mode": str(self.causal),
            "n_rows": int(kl_flat.numel()),
        }
