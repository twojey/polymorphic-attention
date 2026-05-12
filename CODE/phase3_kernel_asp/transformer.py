"""
transformer.py — ASPTransformer : Transformer multi-couches utilisant ASPLayer.

Spec : DOC/03. Architecture symétrique à OracleTransformer (phase 1) pour
comparabilité directe : mêmes embeddings, mêmes positions, même head, mais
chaque block attention remplacé par un ASPLayer.

L'évaluation phase 3 compare cette architecture à l'OracleTransformer sur
le split `init_phase3` (3 sets disjoints règle DOC/01 §8.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from phase3_kernel_asp.asp_layer import ASPLayer, ASPLayerConfig
from phase3_kernel_asp.backbone import Backbone, IdentityBackbone
from phase3_kernel_asp.backbone_concrete import build_backbone_from_class
from phase3_kernel_asp.matriochka import MatriochkaInitConfig


@dataclass
class ASPTransformerConfig:
    """Reproduit OracleConfig pour comparabilité, plus champs phase 3."""
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 8192
    dropout: float = 0.0
    n_classes: int = 10
    pad_id: int = 0
    # --- Phase 3 spécifique ---
    R_max: int = 32
    soft_mask_beta: float = 4.0
    init_strategy: str = "xavier"  # xavier | orthogonal | smart
    backbone_class: str = "identity"  # à raffiner post-phase-2
    backbone_params: dict = field(default_factory=dict)


def _sinusoidal_positions(max_len: int, d: int) -> torch.Tensor:
    """Identique à phase 1 (positions sinusoïdales absolues)."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d, dtype=torch.float32).unsqueeze(0)
    angle = pos / (10000.0 ** (2 * (i // 2) / d))
    pe = torch.zeros(max_len, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class FeedForward(nn.Module):
    """FFN identique à OracleTransformer (pour comparabilité)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(torch.nn.functional.gelu(self.fc1(x))))


class ASPBlock(nn.Module):
    """Block transformer phase 3 : ASPLayer remplace DenseAttention.

    Architecture identique au TransformerBlock phase 1 :
        x = x + ASPLayer(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Le forward de l'ASPLayer en phase 3 (sans Spectromètre actif) utilise
    α_t=1 par token → m_t=1 partout → rang plein. Le Spectromètre (phase 4)
    branchera un α_t prédit token par token.
    """

    def __init__(
        self,
        cfg: ASPTransformerConfig,
        backbone: Backbone,
        matriochka_init: MatriochkaInitConfig | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.asp = ASPLayer(
            ASPLayerConfig(
                d_model=cfg.d_model,
                R_max=cfg.R_max,
                soft_mask_beta=cfg.soft_mask_beta,
                layernorm=False,  # LN externe à la résiduelle
                init_strategy=cfg.init_strategy,
            ),
            backbone=backbone,
            matriochka_init=matriochka_init,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward_rank(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Forward avec rank fixe (smoke checks et eval rank-truncated)."""
        x = x + self.asp.forward_with_rank(self.ln1(x), r)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
        """Forward avec α (B, N) ∈ [0, 1]. Si None → α=1 partout (rang plein)."""
        if alpha is None:
            # ASPLayer.forward_with_rank(R_max) court-circuite le soft_mask
            x = x + self.asp.forward_with_rank(self.ln1(x), self.asp.cfg.R_max)
        else:
            x = x + self.asp.forward_with_alpha(self.ln1(x), alpha)
        x = x + self.ffn(self.ln2(x))
        return x


class ASPTransformer(nn.Module):
    """Transformer phase 3 utilisant ASPLayer à chaque block.

    Interface identique à OracleTransformer (pour eval comparable).
    """

    def __init__(
        self,
        cfg: ASPTransformerConfig,
        matriochka_init_per_layer: list[MatriochkaInitConfig] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.register_buffer(
            "pos_embed", _sinusoidal_positions(cfg.max_seq_len, cfg.d_model), persistent=False
        )
        # Un Backbone par couche (peut être identique ou différencié)
        self.blocks = nn.ModuleList()
        for ell in range(cfg.n_layers):
            backbone = build_backbone_from_class(
                cfg.backbone_class, d_model=cfg.d_model, params=cfg.backbone_params,
            )
            init = (
                matriochka_init_per_layer[ell]
                if matriochka_init_per_layer is not None
                else None
            )
            self.blocks.append(ASPBlock(cfg, backbone=backbone, matriochka_init=init))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Identique à OracleTransformer.forward sauf qu'on n'a pas attn_mask
        (l'ASPLayer ne lit pas une matrice d'attention dense, le PAD est
        géré par token embedding zeros via padding_idx).
        """
        B, N = tokens.shape
        x = self.tok_embed(tokens) + self.pos_embed[:N].unsqueeze(0)
        for block in self.blocks:
            x = block(x, alpha=alpha)
        x = self.ln_f(x)
        idx = query_pos[:, None, None].expand(-1, 1, x.size(-1))
        h = x.gather(dim=1, index=idx).squeeze(1)
        return self.head(h)

    def forward_at_rank(
        self, tokens: torch.Tensor, query_pos: torch.Tensor, r: int,
    ) -> torch.Tensor:
        """Forward complet avec rang Matriochka fixe r (pour sanity + L_matriochka)."""
        B, N = tokens.shape
        x = self.tok_embed(tokens) + self.pos_embed[:N].unsqueeze(0)
        for block in self.blocks:
            x = block.forward_rank(x, r)
        x = self.ln_f(x)
        idx = query_pos[:, None, None].expand(-1, 1, x.size(-1))
        h = x.gather(dim=1, index=idx).squeeze(1)
        return self.head(h)
