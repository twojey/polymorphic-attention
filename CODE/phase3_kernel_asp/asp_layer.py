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
    # Mode V2 (2026-05-12) : choix entre projection linéaire (V1 spec, sans
    # interaction token-token) et attention low-rank (Q = x·U, K = x·V,
    # softmax(QK^T)·x). V1 ne pouvait pas apprendre SMNIST faute de mixing.
    delta_attn_mode: str = "linear"  # "linear" (V1 spec) | "attention" (V2)


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
        if self.cfg.delta_attn_mode == "attention":
            delta = self._matriochka_attention(x, r)
        else:
            delta = self.matriochka.correction(x, r)
        return self.norm(backbone_out + delta)

    def _matriochka_correction_per_token(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        """ΔAttn par token avec masque continu m : (B, N, R_max).

        Mode V1 ("linear"): ΔAttn = Σ_i m_{t,i} · u_i v_iᵀ x (projection
            linéaire conditionnelle au token, PAS d'interaction token-token).
        Mode V2 ("attention"): ΔAttn = softmax((x·U·m) (x·V·m)ᵀ / √r) · x
            (vraie attention low-rank avec mask Matriochka).
        """
        if self.cfg.delta_attn_mode == "attention":
            return self._matriochka_attention_with_mask(x, m)
        # V1 path (linear)
        U = self.matriochka.U_max  # (d, R_max)
        V = self.matriochka.V_max  # (d, R_max)
        Vx = torch.einsum("bnd,dr->bnr", x, V)
        weighted = Vx * m
        out = torch.einsum("bnr,dr->bnd", weighted, U)
        return out

    def _matriochka_attention(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Attention low-rank au rang exact r (sanity + L_matriochka).

        U_max[:, :r] et V_max[:, :r] servent de projections key/query :
            Q = x @ U[:, :r]   (B, N, r)
            K = x @ V[:, :r]   (B, N, r)
            scores = Q Kᵀ / √r → (B, N, N)
            A = softmax(scores)
            out = A @ x → (B, N, d)
        """
        if r == 0:
            return torch.zeros_like(x)
        U_r = self.matriochka.U_max[:, :r]  # (d, r)
        V_r = self.matriochka.V_max[:, :r]
        Q = x @ U_r              # (B, N, r)
        K = x @ V_r              # (B, N, r)
        scores = Q @ K.transpose(-2, -1) / max(r, 1) ** 0.5  # (B, N, N)
        A = torch.softmax(scores, dim=-1)
        out = A @ x              # (B, N, d)
        return out

    def _matriochka_attention_with_mask(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        """Attention low-rank avec mask continu m : (B, N, R_max).

        Mask appliqué sur les colonnes de U, V après projection :
            Q_full = x @ U_max  (B, N, R_max)
            Q_masked = Q_full * m_t (mask par token)
        Comme m varie par token, on calcule Q et K avec mask broadcasté.
        """
        U = self.matriochka.U_max  # (d, R_max)
        V = self.matriochka.V_max
        R = self.cfg.R_max
        Q_full = x @ U             # (B, N, R)
        K_full = x @ V             # (B, N, R)
        # Masque par token : m est (B, N, R) → appliquer composante par composante
        Q = Q_full * m
        K = K_full * m
        # Softmax low-rank attention. Le scaling par √R_eff est approximatif :
        # on prend la somme du mask comme proxy du rang effectif.
        r_eff = m.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, N, 1)
        scores = Q @ K.transpose(-2, -1) / r_eff.sqrt()  # (B, N, N)
        A = torch.softmax(scores, dim=-1)
        out = A @ x
        return out
