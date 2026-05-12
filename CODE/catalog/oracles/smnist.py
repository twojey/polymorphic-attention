"""
smnist.py — SMNISTOracle : adapter Structure-MNIST sur le pipeline catalog.

Spec : DOC/00_FONDATIONS.md §Oracle.

Wrappe l'Oracle phase 1 existant (`phase1_metrologie/oracle/transformer.py`)
en exposant l'interface AbstractOracle. Le `regime_grid()` reproduit le
sweep monovariate du SSG (ω, Δ, ℋ).

Pattern :
- À l'init : charge le checkpoint Oracle + reconstruit le vocab + construit
  l'extracteur d'attention.
- `extract_regime(regime, n_examples)` : génère un mini-dataset SSG pour
  ce point de stress, fait le forward et capture l'attention par couche.

Limitations V1 :
- Pas de batching adaptatif inter-régimes (chaque appel = un régime, un
  forward).
- Le checkpoint Oracle doit être disponible localement.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from phase1_metrologie.oracle.extract import AttentionExtractor, ExtractorConfig
from phase1_metrologie.oracle.train import collate, find_query_positions
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import (
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
)


class SMNISTOracle(AbstractOracle):
    """Oracle Structure-MNIST entraîné phase 1, exposé à la Battery.

    Args
    ----
    checkpoint_path : path du .ckpt produit par phase 1
    n_ops, n_noise : vocab SSG (cf. oracle_smnist.yaml)
    model_kwargs : override des hyperparams du transformer (d_model, n_layers, etc.)
    device : "cpu" ou "cuda"
    fp64_extract : True = cast attention en FP64 à l'extraction (spec DOC/01 §8.4)
    """

    domain = "smnist"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        n_ops: int = 4,
        n_noise: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 8192,
        n_classes: int = 10,
        device: str = "cpu",
        fp64_extract: bool = True,
        oracle_id: str | None = None,
        seed_base: int = 1000,
    ) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint Oracle introuvable : {ckpt}")
        self.checkpoint_path = ckpt
        self.oracle_id = oracle_id or f"smnist_{ckpt.stem}"
        self.device = device
        self.seed_base = seed_base

        self.vocab = Vocab(n_ops=n_ops, n_noise=n_noise)
        self.model_cfg = OracleConfig(
            vocab_size=self.vocab.size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            max_seq_len=max_seq_len, dropout=0.0, n_classes=n_classes,
            pad_id=self.vocab.PAD,
        )
        self._model = self._load_model()
        self._extractor = AttentionExtractor(
            self._model,
            config=ExtractorConfig(
                fp64=fp64_extract,
                validate_numerics=False,
                empty_cache_per_layer=True,
            ),
        )
        # Cache d'infos
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model

    def _load_model(self) -> OracleTransformer:
        state = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model = OracleTransformer(self.model_cfg)
        cleaned = {
            k.removeprefix("_forward_module.").removeprefix("_orig_mod."): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Extrait n_examples matrices d'attention pour ce point de stress SSG.

        Génère un mini-dataset SSG en local, forward sur l'Oracle, capture
        l'attention par couche en FP64 (ou FP32 selon ExtractorConfig).
        """
        import logging
        logger = logging.getLogger("catalog.oracles.smnist")

        if regime.omega is None or regime.delta is None:
            raise ValueError(
                f"SMNISTOracle requiert regime.omega et regime.delta non-null, "
                f"reçu {regime}"
            )

        logger.info(
            "[smnist] extract_regime ω=%d Δ=%d n_ex=%d",
            regime.omega, regime.delta, n_examples,
        )

        cfg = StructureMNISTConfig(
            omega=int(regime.omega),
            delta=int(regime.delta),
            entropy=float(regime.entropy or 0.0),
            n_examples=n_examples,
            n_ops=self.vocab.n_ops,
            n_noise=self.vocab.n_noise,
            seed=self.seed_base + hash(regime.key) % (2**31 - 1),
        )
        dataset = StructureMNISTDataset(cfg)
        loader = DataLoader(
            dataset, batch_size=n_examples, shuffle=False,
            collate_fn=partial(collate, pad_id=self.vocab.PAD),
            num_workers=0, pin_memory=False,
        )

        per_layer_chunks: list[list[torch.Tensor]] = [[] for _ in range(self.n_layers)]
        omegas_chunks: list[torch.Tensor] = []
        deltas_chunks: list[torch.Tensor] = []
        entropies_chunks: list[torch.Tensor] = []
        tokens_chunks: list[torch.Tensor] = []
        query_chunks: list[torch.Tensor] = []

        for batch_idx, batch in enumerate(loader):
            logger.info(
                "[smnist] batch %d ω=%d Δ=%d : forward + extract per-layer",
                batch_idx, regime.omega, regime.delta,
            )
            tokens = batch["tokens"]
            qpos = find_query_positions(tokens, self.vocab.QUERY)
            for layer_dump in self._extractor.extract_per_layer(
                tokens, qpos, batch["targets"],
                batch["omegas"], batch["deltas"], batch["entropies"],
            ):
                per_layer_chunks[layer_dump.layer].append(layer_dump.attn.cpu())
            omegas_chunks.append(batch["omegas"].cpu())
            deltas_chunks.append(batch["deltas"].cpu())
            entropies_chunks.append(batch["entropies"].cpu())
            tokens_chunks.append(tokens.cpu())
            query_chunks.append(qpos.cpu())

        attn_stacked = [torch.cat(chunks, dim=0) for chunks in per_layer_chunks]
        logger.info(
            "[smnist] extract_regime ω=%d Δ=%d done : %d layers stacked",
            regime.omega, regime.delta, len(attn_stacked),
        )
        return AttentionDump(
            attn=attn_stacked,
            omegas=torch.cat(omegas_chunks, dim=0),
            deltas=torch.cat(deltas_chunks, dim=0),
            entropies=torch.cat(entropies_chunks, dim=0),
            tokens=torch.cat(tokens_chunks, dim=0),
            query_pos=torch.cat(query_chunks, dim=0),
            metadata={
                "oracle_id": self.oracle_id,
                "regime": regime.key,
                "n_examples": n_examples,
                "seed": cfg.seed,
            },
        )

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille standard SMNIST : sweep monovariate ω puis Δ.

        Reproduit oracle_smnist.yaml par défaut : ω ∈ [0,1,2,4,6,8] à Δ=16,
        Δ ∈ [0,16,64,256] à ω=2 (Δ=1024 omis par défaut — seq=5127 trop gros).
        """
        omegas_sweep = [0, 1, 2, 4, 6, 8]
        deltas_sweep = [0, 16, 64, 256]
        out: list[RegimeSpec] = []
        for omega in omegas_sweep:
            out.append(RegimeSpec(omega=omega, delta=16, entropy=0.0))
        for delta in deltas_sweep:
            if delta == 16:
                continue  # déjà inclus dans le sweep ω à ω=2
            out.append(RegimeSpec(omega=2, delta=delta, entropy=0.0))
        return out
