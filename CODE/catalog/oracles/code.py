"""
code.py — CodeOracle complet : adapter Dyck-k extended pour catalog.

Spec : DOC/CATALOGUE §3.3 "Code".

V2 complet : génération de chaînes Dyck-k bien parenthésées avec k types
de brackets + tokenizer custom + MinimalTransformer causal.

Régime Code : omega = depth max d'imbrication, delta = seq_len, entropy = k (vocab).

Dyck-k : langage des chaînes équilibrées avec k types de parenthèses
distinctes. Ex k=2 : '([])' valide, '([)]' invalide.

Sans checkpoint : attentions Xavier random.
Sprint S6 : remplacer par StarCoder-2 ou Code-Llama via HFLanguageBackend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from catalog.oracles._minimal_transformer import (
    MinimalTransformer,
    MinimalTransformerSpec,
)
from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class CodeModelSpec:
    vocab_size: int = 32  # 2 × k_max brackets + spéciaux
    max_depth: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 1024
    pad_id: int = 0
    eos_id: int = 1


# ----------------------------------------------------------------------------
# Dyck-k generator
# ----------------------------------------------------------------------------

def generate_dyck_k(depth: int, seq_len: int, k: int, seed: int = 0) -> list[int]:
    """Génère une chaîne Dyck-k bien parenthésée de longueur ≤ seq_len.

    Token convention :
    - 0 : PAD
    - 1 : EOS
    - 2*i + 2 : open bracket type i (i ∈ [0, k))
    - 2*i + 3 : close bracket type i

    Algorithm : random walk avec stack, ne ferme qu'avec le bracket courant.
    """
    rng = torch.Generator().manual_seed(seed)
    out: list[int] = []
    stack: list[int] = []
    target_open_prob = 0.55
    # Réserver place pour fermer tous les brackets ouverts + 1 EOS
    # invariant : on s'arrête d'ouvrir si len(out) + len(stack) >= seq_len - 1
    while len(out) < seq_len - 1:
        slots_remaining = (seq_len - 1) - len(out)
        if not stack:
            if slots_remaining < 2:
                break  # plus de place pour une paire (...)
            t = int(torch.randint(0, k, (1,), generator=rng).item())
            out.append(2 * t + 2)
            stack.append(t)
        elif len(stack) >= depth or slots_remaining <= len(stack):
            # Profondeur max OU plus de place pour fermer → close
            t = stack.pop()
            out.append(2 * t + 3)
        else:
            r = float(torch.rand(1, generator=rng).item())
            if r < target_open_prob:
                t = int(torch.randint(0, k, (1,), generator=rng).item())
                out.append(2 * t + 2)
                stack.append(t)
            else:
                t = stack.pop()
                out.append(2 * t + 3)
    # Fermer tout ce qui reste (devrait toujours rentrer)
    while stack:
        t = stack.pop()
        out.append(2 * t + 3)
    out.append(1)  # EOS
    while len(out) < seq_len:
        out.append(0)
    return out[:seq_len]


def validate_dyck_k(tokens: list[int]) -> bool:
    """Vérifie qu'une suite de tokens forme une chaîne Dyck-k valide."""
    stack: list[int] = []
    for t in tokens:
        if t == 0 or t == 1:
            continue  # PAD/EOS
        if t % 2 == 0:  # open
            stack.append((t - 2) // 2)
        else:  # close
            expect_t = (t - 3) // 2
            if not stack or stack[-1] != expect_t:
                return False
            stack.pop()
    return len(stack) == 0


# ----------------------------------------------------------------------------
# CodeOracle backend
# ----------------------------------------------------------------------------

class _CodeBackend:
    n_layers: int
    n_heads: int

    def forward_with_attn(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class MinimalCodeBackend(_CodeBackend):
    """Backend Code minimal : MinimalTransformer causal + Dyck-k tokens."""

    def __init__(self, spec: CodeModelSpec,
                 checkpoint_path: str | Path | None = None,
                 device: str = "cpu") -> None:
        m_spec = MinimalTransformerSpec(
            vocab_size=spec.vocab_size, d_model=spec.d_model,
            n_heads=spec.n_heads, n_layers=spec.n_layers,
            d_ff=spec.d_ff, max_seq_len=spec.max_seq_len,
            causal=True, pad_id=spec.pad_id,
        )
        self.model = MinimalTransformer(m_spec)
        if checkpoint_path is not None:
            self.model.load_or_init(str(checkpoint_path))
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.n_layers = spec.n_layers
        self.n_heads = spec.n_heads
        self.spec = spec

    @torch.no_grad()
    def forward_with_attn(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids, output_attentions=True)
        return out["attentions"]


# ----------------------------------------------------------------------------
# CodeOracle
# ----------------------------------------------------------------------------

class CodeOracle(AbstractOracle):
    """Oracle Code Transformer (Dyck-k étendu, StarCoder, Code-Llama)."""

    domain = "code"

    def __init__(
        self,
        *,
        backend: _CodeBackend | None = None,
        checkpoint_path: str | Path | None = None,
        model_spec: CodeModelSpec | None = None,
        n_bracket_types: int = 3,
        device: str = "cpu",
        oracle_id: str | None = None,
        n_examples_max: int = 64,
    ) -> None:
        if backend is None:
            if checkpoint_path is not None:
                ckpt = Path(checkpoint_path)
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Checkpoint Code introuvable : {ckpt}")
            spec = model_spec or CodeModelSpec()
            backend = MinimalCodeBackend(spec, checkpoint_path=checkpoint_path, device=device)
            stem = Path(checkpoint_path).stem if checkpoint_path else "random_init"
            self.oracle_id = oracle_id or f"code_minimal_{stem}"
        else:
            self.oracle_id = oracle_id or "code_external"
        self.backend = backend
        self.k = n_bracket_types
        self.device = device
        self.n_layers = backend.n_layers
        self.n_heads = backend.n_heads
        self.n_examples_max = n_examples_max

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        depth = int(regime.omega) if regime.omega is not None else 4
        seq_len = int(regime.delta) if regime.delta is not None else 64
        max_supported = getattr(getattr(self.backend, "spec", None), "max_seq_len", 1024)
        seq_len = min(seq_len, max_supported, 1024)
        n_examples = min(n_examples, self.n_examples_max)

        sequences = [generate_dyck_k(depth, seq_len, self.k, seed=i)
                     for i in range(n_examples)]
        input_ids = torch.tensor(sequences, dtype=torch.long)
        attns = self.backend.forward_with_attn(input_ids)
        attns_fp64 = [a.to(torch.float64).cpu() for a in attns]

        # Validation Dyck-k (sanity check)
        n_valid = sum(1 for s in sequences if validate_dyck_k(s))

        return AttentionDump(
            attn=attns_fp64,
            omegas=torch.full((n_examples,), float(depth)),
            deltas=torch.full((n_examples,), float(seq_len)),
            entropies=torch.full((n_examples,), float(self.k)),
            tokens=input_ids,
            query_pos=torch.zeros(n_examples, dtype=torch.long),
            metadata={
                "oracle_id": self.oracle_id,
                "depth": depth,
                "seq_len": seq_len,
                "n_bracket_types": self.k,
                "n_valid_dyck_k": n_valid,
                "domain": self.domain,
            },
        )

    def regime_grid(self) -> list[RegimeSpec]:
        depths = [1, 2, 4, 8]
        seq_lens = [64, 256, 1024]
        out: list[RegimeSpec] = []
        for d in depths:
            for s in seq_lens:
                out.append(RegimeSpec(omega=d, delta=s, entropy=0.0))
        return out
