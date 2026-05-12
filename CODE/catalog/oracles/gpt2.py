"""
gpt2.py — GPT2Oracle : adapter GPT-2 pré-entraîné (OpenAI, HuggingFace Hub).

Aucune clé HuggingFace requise (modèle public).
Téléchargement automatique au premier appel (~350 MB pour small), puis cache
local dans ~/.cache/huggingface/.

Différence avec LLOracle : GPT-2 est pré-entraîné sur données inconnues
(internet). Le régime ne contrôle pas la structure des données — seule la
longueur de séquence (delta) et la profondeur du prompt (omega) varient.

Rôle dans le projet : tester H_universal (les propriétés catalogue sont-elles
invariantes au pretraining ?) en comparant avec SMNISTOracle et LLOracle
from-scratch. cf. DOC/carnet_de_bord.md §"Double Oracle LL".

Attention causale : GPT-2 = masque triangulaire, triangle supérieur ≈ 0.
Les Properties du catalogue fonctionnent sur matrices causales, la comparaison
cross-Oracle doit en tenir compte.
"""

from __future__ import annotations

from typing import Callable

import torch

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from catalog.oracles.language import HFLanguageBackend, nested_parentheses_template

# Variants GPT-2 disponibles sur HuggingFace Hub (aucune clé requise)
GPT2_VARIANTS: dict[str, str] = {
    "small":  "gpt2",         # 117M params, 12 layers, 12 heads, d=768
    "medium": "gpt2-medium",  # 345M params, 24 layers, 16 heads, d=1024
    "large":  "gpt2-large",   # 774M params, 36 layers, 20 heads, d=1280
    "xl":     "gpt2-xl",      # 1.5B params, 48 layers, 25 heads, d=1600
}

GPT2_MAX_SEQ_LEN = 1024  # limite hard GPT-2 (position embeddings absolus)


class GPT2Oracle(AbstractOracle):
    """Oracle GPT-2 pré-entraîné (OpenAI) exposé à la Battery.

    Régimes : sweep sur seq_len (delta). omega = profondeur du prompt
    nested_parentheses (0 = phrase plate, 2+ = clauses imbriquées) — permet
    de varier la structure long-range sans contrôle du training data.

    Args
    ----
    variant : "small" | "medium" | "large" | "xl"
    device : "cpu" ou "cuda"
    prompt_fn : callable (depth, seq_len, seed) → str.
        Défaut = nested_parentheses_template (identique à LLOracle).
    n_examples_max : cap pour éviter OOM sur CPU
    oracle_id : identifiant MLflow ; défaut = "gpt2_<variant>"
    """

    domain = "pretrained_ll"

    def __init__(
        self,
        variant: str = "small",
        *,
        device: str = "cpu",
        prompt_fn: Callable[[int, int, int], str] = nested_parentheses_template,
        n_examples_max: int = 32,
        oracle_id: str | None = None,
    ) -> None:
        if variant not in GPT2_VARIANTS:
            raise ValueError(
                f"variant GPT-2 inconnu : {variant!r}. "
                f"Valides : {sorted(GPT2_VARIANTS)}"
            )
        model_name = GPT2_VARIANTS[variant]
        self.variant = variant
        self.oracle_id = oracle_id or f"gpt2_{variant}"
        self.device = device
        self.prompt_fn = prompt_fn
        self.n_examples_max = n_examples_max

        self._backend = HFLanguageBackend(
            model_name, device=device, torch_dtype=torch.float32
        )
        self.n_layers = self._backend.n_layers
        self.n_heads = self._backend.n_heads

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Extrait n_examples matrices d'attention pour ce régime.

        delta = seq_len cible, cappé à GPT2_MAX_SEQ_LEN=1024.
        omega = profondeur du prompt nested_parentheses.
        """
        seq_len = min(
            int(regime.delta) if regime.delta is not None else 64,
            GPT2_MAX_SEQ_LEN,
        )
        depth = int(regime.omega) if regime.omega is not None else 0
        n_examples = min(n_examples, self.n_examples_max)

        texts = [
            self.prompt_fn(depth, seq_len, seed=i)
            for i in range(n_examples)
        ]
        input_ids = self._backend.tokenize(texts, max_len=seq_len)
        attns = self._backend.forward_with_attn(input_ids)
        attns_fp64 = [a.to(torch.float64).cpu() for a in attns]

        return AttentionDump(
            attn=attns_fp64,
            omegas=torch.zeros(n_examples, dtype=torch.float32),
            deltas=torch.full((n_examples,), float(seq_len)),
            entropies=torch.zeros(n_examples),
            tokens=input_ids.cpu(),
            query_pos=torch.zeros(n_examples, dtype=torch.long),
            metadata={
                "oracle_id": self.oracle_id,
                "variant": self.variant,
                "seq_len": seq_len,
                "depth": depth,
                "domain": self.domain,
                "causal": True,
            },
        )

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille GPT-2 : sweep seq_len × 2 profondeurs de prompt.

        delta ∈ [64, 128, 256, 512] (512 raisonnable sur CPU, max 1024).
        omega ∈ [0, 2] (phrases plates vs clauses imbriquées).
        """
        seq_lens = [64, 128, 256, 512]
        depths = [0, 2]
        return [
            RegimeSpec(omega=d, delta=s, entropy=0.0)
            for d in depths
            for s in seq_lens
        ]
