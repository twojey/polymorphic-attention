"""
asp_layer.py — Assemblage final ASPLayer.

Spec : DOC/03 §6.

y = Backbone(x) + ΔAttn(x ; m_t)

avec :
- Backbone : opérateur structuré causal (interface abstraite, dérivé phase 2)
- ΔAttn(x, m) = Σ_i m_{t,i} · u_i v_i^T x   (correction Matriochka pondérée par soft-mask)
- LayerNorm post-addition
- α_t ∈ [0, 1] fourni par le Spectromètre (phase 4) ; en phase 3 on utilise
  α_t = 1 (rang plein) pour les sanity checks de saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from phase3_kernel_asp.backbone import Backbone, IdentityBackbone
from phase3_kernel_asp.matriochka import MatriochkaBases, MatriochkaInitConfig
from phase3_kernel_asp.soft_mask import soft_mask


@dataclass
class ASPLayerConfig:
    d_model: int
    R_max: int
    soft_mask_beta: float = 4.0
    layernorm: bool = True
    init_strategy: str = "xavier"


class ASPLayer(nn.Module):
    """ASPLayer = Backbone + ΔAttn Matriochka modulée par soft-mask."""

    def __init__(
        self,
        cfg: ASPLayerConfig,
        backbone: Backbone | None = None,
        matriochka_init: MatriochkaInitConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone or IdentityBackbone()
        init = matriochka_init or MatriochkaInitConfig(strategy=cfg.init_strategy)
        self.matriochka = MatriochkaBases(cfg.d_model, cfg.R_max, init=init)
        self.matriochka.freeze_smart_columns()
        self.norm = nn.LayerNorm(cfg.d_model) if cfg.layernorm else nn.Identity()

    def forward_with_alpha(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Forward avec α_t (B, N) ∈ [0, 1]. m calculé via soft_mask."""
        backbone_out = self.backbone(x)
        # m : (B, N, R_max)
        m = soft_mask(alpha=alpha, R_max=self.cfg.R_max, beta=self.cfg.soft_mask_beta)
        # ΔAttn = Σ_i m_{t,i} · u_i v_i^T x  → (B, N, d)
        # implémentation : weight = (U * m_i) @ V^T → mais m varie par token
        delta = self._matriochka_correction_per_token(x, m)
        return self.norm(backbone_out + delta)

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward avec mask explicite (B, N, R_max). Plus général que alpha."""
        backbone_out = self.backbone(x)
        delta = self._matriochka_correction_per_token(x, mask)
        return self.norm(backbone_out + delta)

    def forward_with_rank(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Forward avec masque [1×r, 0×(R_max-r)] uniforme sur tous les tokens.

        Utilisé par L_matriochka (sampling de r) et les sanity checks (r=R_max
        pour saturation, r=0 pour effondrement).
        """
        backbone_out = self.backbone(x)
        if r == 0:
            return self.norm(backbone_out)
        delta = self.matriochka.correction(x, r)
        return self.norm(backbone_out + delta)

    def _matriochka_correction_per_token(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        """ΔAttn par token avec masque continu m : (B, N, R_max).

        Calcul : (B, N, d) → on projette via Σ_i m_i · u_i ⊗ v_i.
        """
        # weight_per_token[b, t] = U @ diag(m[b, t]) @ V^T = (U * m[b,t]) @ V^T
        # On fait un einsum efficace.
        # x : (B, N, d), U : (d, R), V : (d, R), m : (B, N, R)
        U = self.matriochka.U_max  # (d, R_max)
        V = self.matriochka.V_max  # (d, R_max)
        # V^T x : (B, N, R)
        Vx = torch.einsum("bnd,dr->bnr", x, V)
        # appliquer m : (B, N, R) * (B, N, R) → (B, N, R)
        weighted = Vx * m
        # U @ result : (B, N, d)
        out = torch.einsum("bnr,dr->bnd", weighted, U)
        return out
