"""
transformer.py — Oracle Transformer dense pour phase 1.

Conforme à DOC/01 §8 :
- Attention dense pleine, pas de GQA/MQA/SWA.
- Chaque tête a ses propres K, V.
- Pas de top-k routing, pas de sparsité injectée.
- Flash Attention OK à l'entraînement (équivalent mathématique), désactivé
  à l'extraction de A.
- Position encoding : sinusoïdal absolu (pas de RoPE pour V1, à documenter).

Le modèle est volontairement simple. Toute optimisation (Flash, fused QKV)
est appliquée au moment de l'entraînement uniquement, pas de l'extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OracleConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 4096
    dropout: float = 0.0
    n_classes: int = 10  # phase 1 : prédiction sur QUERY token, ∈ [0, 10)
    pad_id: int = 0      # important pour le mask, override depuis le SSG


def _sinusoidal_positions(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d, dtype=torch.float32).unsqueeze(0)
    angle = pos / (10000.0 ** (2 * (i // 2) / d))
    pe = torch.zeros(max_len, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class DenseAttention(nn.Module):
    """Attention dense par tête, exposant les poids `attn_weights`.

    À l'entraînement : utilise scaled_dot_product_attention si possible
    (Flash equivalent). À l'extraction : `force_explicit=True` pour
    matérialiser A et la cacher dans `self.last_attn`.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout
        self.last_attn: torch.Tensor | None = None  # (B, H, N, N) si capturée

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        capture_attn: bool = False,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # chacun (B, N, H, d_head)
        q = q.transpose(1, 2)        # (B, H, N, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if capture_attn:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = torch.softmax(scores, dim=-1)
            self.last_attn = attn.detach()
            out = attn @ v
        else:
            self.last_attn = None
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: OracleConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = DenseAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        capture_attn: bool,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask, capture_attn=capture_attn)
        x = x + self.ffn(self.ln2(x))
        return x


class OracleTransformer(nn.Module):
    """Transformer dense pour Oracle phase 1.

    Le head de classification lit l'embedding au token QUERY (premier ou
    dernier non-PAD selon la convention SSG ; ici on prend le dernier
    token non-PAD avant le PAD final).
    """

    def __init__(self, cfg: OracleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.register_buffer(
            "pos_embed", _sinusoidal_positions(cfg.max_seq_len, cfg.d_model), persistent=False
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        capture_attn: bool = False,
    ) -> torch.Tensor:
        """tokens : (B, N) int64 ; query_pos : (B,) int64. Retourne logits (B, n_classes)."""
        B, N = tokens.shape
        x = self.tok_embed(tokens) + self.pos_embed[:N].unsqueeze(0)

        # Mask additif : -inf sur PAD (pour bloquer attention vers PAD)
        pad_mask = tokens == self.cfg.pad_id  # (B, N)
        # broadcast (B, 1, 1, N) ; appliqué aux scores (B, H, N, N)
        attn_mask = torch.zeros(B, 1, 1, N, device=tokens.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, capture_attn=capture_attn)
        x = self.ln_f(x)

        # extraire l'embedding au QUERY token
        idx = query_pos[:, None, None].expand(-1, 1, x.size(-1))
        h = x.gather(dim=1, index=idx).squeeze(1)
        return self.head(h)
