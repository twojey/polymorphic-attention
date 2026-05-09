"""
extract.py — extraction des matrices d'attention de l'Oracle.

Conforme à DOC/01 §3.1, §8.4, §8.7 :
- Pas d'agrégation pré-extraction (par tête, par couche, par exemple)
- Cast FP64 au moment de l'extraction (BF16 à l'entraînement)
- L'Oracle n'est pas modifié entre entraînement et extraction
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phase1_metrologie.oracle.transformer import OracleTransformer


@dataclass
class AttentionDump:
    """Résultat de l'extraction sur un batch.

    `attn[ℓ]` : tensor (B, H, N, N) FP64 contenant la matrice softmax(QK^T/√d)
    pour la couche ℓ.
    """

    attn: list[torch.Tensor]            # liste de (B, H, N, N) FP64
    tokens: torch.Tensor                # (B, N) int64
    targets: torch.Tensor               # (B,) int64
    omegas: torch.Tensor                # (B,) int
    deltas: torch.Tensor                # (B,) int
    entropies: torch.Tensor             # (B,) float

    def n_layers(self) -> int:
        return len(self.attn)

    def n_heads(self) -> int:
        return self.attn[0].size(1)

    def seq_len(self) -> int:
        return self.attn[0].size(2)


class AttentionExtractor:
    """Extrait les matrices A par couche en FP64 sur un batch donné."""

    def __init__(self, model: OracleTransformer) -> None:
        self.model = model

    @torch.no_grad()
    def extract(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        targets: torch.Tensor,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        entropies: torch.Tensor,
    ) -> AttentionDump:
        was_training = self.model.training
        self.model.eval()
        try:
            _ = self.model(tokens, query_pos, capture_attn=True)
            attn_per_layer: list[torch.Tensor] = []
            for block in self.model.blocks:
                a = block.attn.last_attn
                assert a is not None, "DenseAttention.last_attn manquant — capture_attn ignoré ?"
                attn_per_layer.append(a.to(torch.float64).contiguous())
                # libère le buffer pour ne pas garder en mémoire au-delà du dump
                block.attn.last_attn = None
        finally:
            if was_training:
                self.model.train()
        return AttentionDump(
            attn=attn_per_layer,
            tokens=tokens.detach(),
            targets=targets.detach(),
            omegas=omegas.detach(),
            deltas=deltas.detach(),
            entropies=entropies.detach(),
        )
