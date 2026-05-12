"""
language.py — LLOracle : adapter pour Oracle de modélisation langagière (LL).

Spec : DOC/CATALOGUE §3.3 "TinyStories pour LL Sprint S7".

Wrappe un Transformer de type GPT entraîné sur TinyStories ou similaire,
expose l'interface AbstractOracle. Le "régime" pour LL n'est pas (ω, Δ, ℋ)
mais des proxies de stress structurel :

- `complexity_level` : depth structurelle (clauses nested, parenthèses)
- `seq_len` : longueur fenêtre contexte
- `vocab_perplexity` : entropie tokens hors-domaine (proxy ℋ pour LL)

V1 sans training : le checkpoint doit déjà exister. La méthode
`extract_regime` génère un dataset à la volée à partir d'un prompt template
et de paramètres de stress, fait un forward pass et capture l'attention.

Code structuré pour permettre future Sprint S7 (training + checkpoint
spécifique LL).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class LLModelSpec:
    """Spec du modèle LL attendu pour adapter."""
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    pad_id: int = 0
    eos_id: int = 1


class LLOracle(AbstractOracle):
    """Oracle de modélisation langagière (TinyStories, BabyLM, etc.).

    Args
    ----
    checkpoint_path : path du checkpoint .pt (state_dict d'un GPT-like)
    model_spec : LLModelSpec — hyperparams du modèle
    device : "cpu" ou "cuda"
    tokenizer : callable str → list[int] (ou objet exposant .encode/.decode)
    prompt_template : fonction (complexity_level, seq_len) → str (génère
                      un prompt avec stress structurel paramétrable)

    V1 : implémentation squelette. Les méthodes _load_model, _generate_dataset
    et extract_regime doivent être complétées avec un vrai checkpoint LL et
    un tokenizer/template (Sprint S7).
    """

    domain = "ll"

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_spec: LLModelSpec,
        *,
        tokenizer: Any = None,
        prompt_template: Any = None,
        device: str = "cpu",
        oracle_id: str | None = None,
    ) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint LL introuvable : {ckpt}")
        self.checkpoint_path = ckpt
        self.model_spec = model_spec
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.device = device
        self.oracle_id = oracle_id or f"ll_{ckpt.stem}"
        self.n_layers = model_spec.n_layers
        self.n_heads = model_spec.n_heads

        self._model: Any = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # TODO Sprint S7 : charger l'architecture exacte (GPT/Mistral/Llama
        # selon checkpoint) et initialiser self._model.
        raise NotImplementedError(
            "LLOracle._ensure_loaded : adapter spécifique au checkpoint LL "
            "non encore implémenté (Sprint S7). Pour l'instant LLOracle expose "
            "l'interface mais le pipeline d'extraction est à compléter."
        )

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Extrait n_examples attentions pour ce régime LL.

        Régime pour LL : utilise extra={'complexity_level': int, 'seq_len': int}
        ou les champs natifs omega/delta réinterprétés (omega = complexity_level,
        delta = seq_len fenêtre).
        """
        if self.prompt_template is None or self.tokenizer is None:
            raise NotImplementedError(
                "LLOracle.extract_regime : prompt_template + tokenizer requis "
                "(passer en __init__). V2 Sprint S7."
            )
        self._ensure_loaded()
        # Code structuré, à compléter Sprint S7
        raise NotImplementedError("Pipeline forward LL à implémenter Sprint S7.")

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille standard LL : sweep complexity_level × seq_len.

        Convention :
        - omega ← complexity_level ∈ {0, 1, 2, 4, 8} (depth structurelle)
        - delta ← seq_len ∈ {64, 256, 1024} (fenêtre contexte)
        """
        complexity_levels = [0, 1, 2, 4, 8]
        seq_lens = [64, 256, 1024]
        out: list[RegimeSpec] = []
        for c in complexity_levels:
            for s in seq_lens:
                out.append(RegimeSpec(omega=c, delta=s, entropy=0.0))
        return out


def make_tinystories_template() -> Any:
    """Factory placeholder pour template TinyStories (Sprint S7).

    Sprint S7 : implémenter avec exemples nested clauses ('The dog that the
    cat that the mouse saw chased ran away.') et niveaux de complexité.
    """
    raise NotImplementedError(
        "make_tinystories_template : Sprint S7 — implémenter avec dataset "
        "TinyStories réel + paramétrisation depth structurelle."
    )
