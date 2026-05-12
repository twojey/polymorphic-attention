"""
code.py — CodeOracle : adapter pour modèle de code (Dyck-k étendu, mini-DSL).

Spec : DOC/CATALOGUE §3.3 "Code".

V1 sans training : adapter pour Transformer entraîné sur séquences de
code synthétique (Dyck-k = langage de parenthèses bien parenthésées avec
k types).

Régime pour Code = (depth, nesting_factor, vocab_size) :
- omega ← depth max de nesting
- delta ← seq_len
- entropy ← vocab_size_used (proxy ℋ)

V1 squelette : Sprint S6+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class CodeModelSpec:
    vocab_size: int
    max_depth: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int


class CodeOracle(AbstractOracle):
    """Oracle Code Transformer (Dyck-k étendu, mini-DSL)."""

    domain = "code"

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_spec: CodeModelSpec | None = None,
        *,
        device: str = "cpu",
        oracle_id: str | None = None,
    ) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint Code introuvable : {ckpt}")
        self.checkpoint_path = ckpt
        self.model_spec = model_spec
        self.device = device
        self.oracle_id = oracle_id or f"code_{ckpt.stem}"
        self.n_layers = model_spec.n_layers if model_spec else 4
        self.n_heads = model_spec.n_heads if model_spec else 4

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Sprint S6 prep — V1 squelette. Compléter avec :
        - parseur Dyck-k extended pour génération
        - tokenizer paren/bracket/keyword
        - forward + extraction attention
        """
        raise NotImplementedError(
            "CodeOracle.extract_regime : Sprint S6 prep — V1 squelette."
        )

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille Code : depth × seq_len × vocab."""
        depths = [1, 2, 4, 8]
        seq_lens = [64, 256, 1024]
        out: list[RegimeSpec] = []
        for d in depths:
            for s in seq_lens:
                out.append(RegimeSpec(omega=d, delta=s, entropy=0.0))
        return out
