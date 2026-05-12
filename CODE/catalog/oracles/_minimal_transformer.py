"""
_minimal_transformer.py — Minimal Transformer pour Oracles LL/Vision/Code.

Spec : DOC/CATALOGUE §3.3.

Architecture GPT-style minimaliste utilisée comme fallback quand un
checkpoint HuggingFace n'est pas disponible. Vrai usage Sprint S5-S7 :
remplacer par DINOv2 / Llama / StarCoder via le backend HF.

Caractéristiques :
- Causal mask configurable (LL + Code) ou bidirectionnel (Vision)
- output_attentions=True intégré
- Init Xavier/normal, pas d'entraînement nécessaire pour smoke tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MinimalTransformerSpec:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    causal: bool = True  # True pour LM/Code, False pour Vision (bidir)
    dropout: float = 0.0
    pad_id: int = 0


class _MultiHeadAttention(nn.Module):
    def __init__(self, spec: MinimalTransformerSpec) -> None:
        super().__init__()
        if spec.d_model % spec.n_heads != 0:
            raise ValueError(f"d_model={spec.d_model} pas divisible par n_heads={spec.n_heads}")
        self.spec = spec
        self.d_head = spec.d_model // spec.n_heads
        self.qkv = nn.Linear(spec.d_model, 3 * spec.d_model, bias=False)
        self.out = nn.Linear(spec.d_model, spec.d_model, bias=False)

    def forward(self, x: torch.Tensor, return_attn: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, self.spec.n_heads, self.d_head).transpose(1, 2)
                   for t in qkv)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.spec.causal:
            mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, N, d_head)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out(out)
        return out, attn if return_attn else None


class _TransformerLayer(nn.Module):
    def __init__(self, spec: MinimalTransformerSpec) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.d_model)
        self.attn = _MultiHeadAttention(spec)
        self.norm2 = nn.LayerNorm(spec.d_model)
        self.ff = nn.Sequential(
            nn.Linear(spec.d_model, spec.d_ff),
            nn.GELU(),
            nn.Linear(spec.d_ff, spec.d_model),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        h, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, attn


class MinimalTransformer(nn.Module):
    """Mini-GPT pour tests Oracle (LM + ViT + Code).

    Si Vision : passer spec.causal=False, l'embedding accepte tokens patches.
    """

    def __init__(self, spec: MinimalTransformerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.tok_emb = nn.Embedding(spec.vocab_size, spec.d_model, padding_idx=spec.pad_id)
        self.pos_emb = nn.Embedding(spec.max_seq_len, spec.d_model)
        self.layers = nn.ModuleList([_TransformerLayer(spec) for _ in range(spec.n_layers)])
        self.ln_f = nn.LayerNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, output_attentions: bool = True) -> dict:
        B, N = input_ids.shape
        if N > self.spec.max_seq_len:
            raise ValueError(f"seq_len={N} > max_seq_len={self.spec.max_seq_len}")
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        attns: list[torch.Tensor] = []
        for layer in self.layers:
            x, a = layer(x, return_attn=output_attentions)
            if output_attentions and a is not None:
                attns.append(a)
        x = self.ln_f(x)
        logits = self.head(x)
        return {
            "logits": logits,
            "attentions": attns if output_attentions else None,
        }

    def load_or_init(self, checkpoint_path: str | None = None) -> None:
        """Charge un state_dict si valide, sinon garde init aléatoire (warn)."""
        if checkpoint_path is not None:
            from pathlib import Path
            import sys
            p = Path(checkpoint_path)
            if p.is_file():
                try:
                    state = torch.load(p, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(
                        f"[MinimalTransformer.load_or_init] checkpoint {p} "
                        f"invalide ({type(e).__name__}) — random init conservé.",
                        file=sys.stderr,
                    )
                    return
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                try:
                    self.load_state_dict(state, strict=False)
                except Exception as e:
                    print(
                        f"[MinimalTransformer.load_or_init] state_dict shape "
                        f"mismatch ({type(e).__name__}) — random init conservé.",
                        file=sys.stderr,
                    )
