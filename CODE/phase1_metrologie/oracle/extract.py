"""
extract.py — extraction des matrices d'attention de l'Oracle.

Conforme à DOC/01 §3.1, §8.4, §8.7 :
- Pas d'agrégation pré-extraction (par tête, par couche, par exemple)
- Cast FP64 au moment de l'extraction (BF16 à l'entraînement)
- L'Oracle n'est pas modifié entre entraînement et extraction

Refactor 2026-05-11 (carnet entrée du jour, suite crash Run 3) :
- Ajout API `extract_per_layer` (générateur) → libération mémoire progressive
- Ajout API `extract_streamed` (callback) → consommation immédiate sans accumulation
- Ajout API `extract_windowed_per_layer` → fenêtres K×K diagonales pour phase 2
- ExtractorConfig : validation numérique opt-in, empty_cache, fp64, max_layers
- Backward compat : `extract()` retourne toujours un AttentionDump complet
- Tests : voir tests/test_extract_streaming.py
- DOC/00b : nouvelles APIs référencées dans §II.7

Limites connues :
- Le forward Oracle reste monolithique (toutes attentions matérialisées dans
  les buffers .last_attn pendant le forward). Le streaming réduit la
  mémoire FP64 + activations downstream, PAS le pic du forward lui-même.
- Pour seq_len > ~8192 sur GPU 24GB, il faudra un refactor transformer.py
  avec hook per-layer (TODO phase 2, cf. DOC/02 §extraction).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from phase1_metrologie.oracle.transformer import OracleTransformer


# -----------------------------------------------------------------------------
# Dataclasses : dumps
# -----------------------------------------------------------------------------


@dataclass
class AttentionDump:
    """Résultat de l'extraction sur un batch (toutes couches matérialisées).

    Backward compat : conserve la sémantique d'origine (extract() retourne
    dump complet). Pour grandes seq_len, préférer extract_per_layer /
    extract_streamed afin d'éviter d'accumuler tous les FP64 (B,H,N,N) en RAM.

    `attn[ℓ]` : tensor (B, H, N, N) FP64 contenant la matrice softmax(QK^T/√d)
    pour la couche ℓ.
    """

    attn: list[torch.Tensor]
    tokens: torch.Tensor
    targets: torch.Tensor
    omegas: torch.Tensor
    deltas: torch.Tensor
    entropies: torch.Tensor

    def n_layers(self) -> int:
        return len(self.attn)

    def n_heads(self) -> int:
        return self.attn[0].size(1)

    def seq_len(self) -> int:
        return self.attn[0].size(2)


@dataclass
class LayerDump:
    """Dump streamé pour une seule couche (mode extract_per_layer / streamed).

    Si `is_windowed=True`, `attn` est une fenêtre carrée K×K extraite le long
    de la diagonale (cf. extract_windowed_per_layer). Le champ `window_offset`
    indique alors la position de départ de la fenêtre dans la matrice complète.
    """

    layer: int
    attn: torch.Tensor
    tokens: torch.Tensor
    targets: torch.Tensor
    omegas: torch.Tensor
    deltas: torch.Tensor
    entropies: torch.Tensor
    is_windowed: bool = False
    window_size: int | None = None
    window_offset: int | None = None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ExtractorConfig:
    """Pilote les compromis mémoire/robustesse pour l'extraction."""

    fp64: bool = True
    validate_numerics: bool = False
    empty_cache_per_layer: bool = True
    max_layers: int | None = None
    stream_to_disk: Path | None = None


# -----------------------------------------------------------------------------
# Extracteur
# -----------------------------------------------------------------------------


class AttentionExtractor:
    """Extrait les matrices A par couche.

    Trois APIs disponibles :
    - `extract(...)` — backward compat, dump complet en RAM.
    - `extract_per_layer(...)` — générateur yield LayerDump, libère par couche.
    - `extract_streamed(..., callback)` — callback API, libère après callback.
    - `extract_windowed_per_layer(...)` — fenêtres K×K diagonales (phase 2).
    """

    def __init__(
        self,
        model: OracleTransformer,
        config: ExtractorConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or ExtractorConfig()
        self._device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self.model.parameters()).device
        return self._device

    # -------------------------------------------------------------------------
    # Helpers internes
    # -------------------------------------------------------------------------

    def _maybe_validate(self, a: torch.Tensor, layer: int) -> None:
        """Vérifie l'absence de NaN/Inf et range [0,1] (post-softmax)."""
        if not self.config.validate_numerics:
            return
        if torch.isnan(a).any():
            raise ValueError(f"NaN détecté dans attention couche {layer}")
        if torch.isinf(a).any():
            raise ValueError(f"Inf détecté dans attention couche {layer}")
        amin = a.min().item()
        amax = a.max().item()
        if amin < -1e-6 or amax > 1.0 + 1e-6:
            raise ValueError(
                f"Attention couche {layer} hors [0,1] : min={amin:.3e} max={amax:.3e}"
            )

    def _release_buffers(self) -> None:
        for block in self.model.blocks:
            block.attn.last_attn = None
        if self.config.empty_cache_per_layer and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _convert(self, a: torch.Tensor, layer: int) -> torch.Tensor:
        out = a.to(torch.float64).contiguous() if self.config.fp64 else a.contiguous()
        self._maybe_validate(out, layer)
        return out

    def _maybe_dump_to_disk(self, layer: int, dump: LayerDump) -> None:
        if self.config.stream_to_disk is None:
            return
        self.config.stream_to_disk.mkdir(parents=True, exist_ok=True)
        path = self.config.stream_to_disk / f"layer_{layer:03d}.pt"
        torch.save(
            {
                "layer": dump.layer,
                "attn": dump.attn.cpu(),
                "tokens": dump.tokens.cpu(),
                "targets": dump.targets.cpu(),
                "omegas": dump.omegas.cpu(),
                "deltas": dump.deltas.cpu(),
                "entropies": dump.entropies.cpu(),
                "is_windowed": dump.is_windowed,
                "window_size": dump.window_size,
                "window_offset": dump.window_offset,
            },
            path,
        )

    @torch.no_grad()
    def _forward_with_capture(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            _ = self.model(tokens, query_pos, capture_attn=True)
        finally:
            if was_training:
                self.model.train()

    def _resolve_n_layers(self) -> int:
        n_layers = len(self.model.blocks)
        if self.config.max_layers is not None:
            n_layers = min(n_layers, self.config.max_layers)
        return n_layers

    # -------------------------------------------------------------------------
    # API : extract (backward compat)
    # -------------------------------------------------------------------------

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
        """API historique : dump complet de toutes les couches.

        Mémoire : conserve toutes les couches en VRAM/RAM simultanément.
        Pour seq_len > 4096, préférer extract_per_layer ou extract_streamed.
        """
        tokens = tokens.to(self.device)
        query_pos = query_pos.to(self.device)

        self._forward_with_capture(tokens, query_pos)
        try:
            attn_per_layer: list[torch.Tensor] = []
            n_layers = self._resolve_n_layers()
            for ell in range(n_layers):
                block = self.model.blocks[ell]
                a = block.attn.last_attn
                if a is None:
                    raise RuntimeError(
                        f"last_attn manquant couche {ell} — capture_attn=False ?"
                    )
                attn_per_layer.append(self._convert(a, ell))
                block.attn.last_attn = None
        finally:
            self._release_buffers()

        return AttentionDump(
            attn=attn_per_layer,
            tokens=tokens.detach(),
            targets=targets.detach(),
            omegas=omegas.detach(),
            deltas=deltas.detach(),
            entropies=entropies.detach(),
        )

    # -------------------------------------------------------------------------
    # API : extract_per_layer (générateur)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_per_layer(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        targets: torch.Tensor,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        entropies: torch.Tensor,
    ) -> Iterator[LayerDump]:
        """Générateur : yield LayerDump par couche, libère après chaque yield.

        Permet au caller de traiter une couche puis la laisser libérer avant
        de demander la suivante. Pic mémoire FP64 ≈ 1 couche au lieu de L.
        """
        tokens = tokens.to(self.device)
        query_pos = query_pos.to(self.device)
        tokens_d = tokens.detach()
        targets_d = targets.detach()
        omegas_d = omegas.detach()
        deltas_d = deltas.detach()
        entropies_d = entropies.detach()

        self._forward_with_capture(tokens, query_pos)
        try:
            n_layers = self._resolve_n_layers()
            for ell in range(n_layers):
                block = self.model.blocks[ell]
                a = block.attn.last_attn
                if a is None:
                    raise RuntimeError(
                        f"last_attn manquant couche {ell} — capture_attn=False ?"
                    )
                a_out = self._convert(a, ell)
                block.attn.last_attn = None

                dump = LayerDump(
                    layer=ell,
                    attn=a_out,
                    tokens=tokens_d,
                    targets=targets_d,
                    omegas=omegas_d,
                    deltas=deltas_d,
                    entropies=entropies_d,
                )
                self._maybe_dump_to_disk(ell, dump)
                yield dump

                del a_out
                if self.config.empty_cache_per_layer and self.device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            self._release_buffers()

    # -------------------------------------------------------------------------
    # API : extract_streamed (callback)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_streamed(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        targets: torch.Tensor,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        entropies: torch.Tensor,
        per_layer_callback: Callable[[LayerDump], None],
    ) -> None:
        """Variante callback de extract_per_layer.

        Appelle per_layer_callback(layer_dump) pour chaque couche. Le caller
        peut compute / sauvegarder / accumuler puis return. Le buffer est
        libéré après le callback.
        """
        for layer_dump in self.extract_per_layer(
            tokens, query_pos, targets, omegas, deltas, entropies
        ):
            per_layer_callback(layer_dump)

    # -------------------------------------------------------------------------
    # API : extract_windowed_per_layer (phase 2)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_windowed_per_layer(
        self,
        tokens: torch.Tensor,
        query_pos: torch.Tensor,
        targets: torch.Tensor,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        entropies: torch.Tensor,
        *,
        K: int,
        stride: int | None = None,
    ) -> Iterator[LayerDump]:
        """Fenêtres K×K diagonales par couche (usage phase 2 audit spectral).

        Pour seq_len N et fenêtre K avec stride=K (défaut), itère ⌈(N-K+1)/stride⌉
        fenêtres. Chaque LayerDump contient une fenêtre (B, H, K, K) FP64.

        Args:
            K: taille de la fenêtre carrée
            stride: pas entre deux fenêtres (défaut = K, donc non-recouvrant)

        Yields:
            LayerDump avec is_windowed=True, window_size=K, window_offset=start.
        """
        if K <= 0:
            raise ValueError(f"K doit être > 0, reçu K={K}")
        stride_eff = stride if stride is not None else K
        if stride_eff <= 0:
            raise ValueError(f"stride doit être > 0, reçu stride={stride_eff}")

        for layer_dump in self.extract_per_layer(
            tokens, query_pos, targets, omegas, deltas, entropies
        ):
            a = layer_dump.attn
            B, H, N, _ = a.shape
            if K > N:
                raise ValueError(f"K={K} > seq_len={N} pour la couche {layer_dump.layer}")
            for start in range(0, N - K + 1, stride_eff):
                end = start + K
                window = a[:, :, start:end, start:end].contiguous()
                yield LayerDump(
                    layer=layer_dump.layer,
                    attn=window,
                    tokens=layer_dump.tokens,
                    targets=layer_dump.targets,
                    omegas=layer_dump.omegas,
                    deltas=layer_dump.deltas,
                    entropies=layer_dump.entropies,
                    is_windowed=True,
                    window_size=K,
                    window_offset=start,
                )
