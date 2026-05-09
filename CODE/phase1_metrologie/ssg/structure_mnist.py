"""
structure_mnist.py — Structural Stress Generator (SSG) sur Structure-MNIST.

Implémentation V1 token-level. Voir DOC/01_phase_metrologie.md sections 2 et 4.

Trois axes orthogonaux :

- ω (récursion)  : profondeur de composition fonctionnelle. Chaque exemple
  contient ω opérations modulaires successives appliquées à un digit MNIST
  initial. Le modèle doit prédire le résultat final mod 10.
- Δ (distance)   : nombre maximal de tokens distracteurs entre deux tokens
  informatifs successifs. Distance de dépendance.
- ℋ (entropie)   : ratio signal/distracteurs. Plus ℋ est grand, plus les
  distracteurs ressemblent aux tokens informatifs (vocabulaire élargi,
  bruit non-trivial à filtrer).

V1 = token-level (digits = tokens d'identité 0-9, pas de pixels). C'est une
simplification documentée par rapport à la spec qui mentionne des pixels MNIST.
La structure (ω, Δ, ℋ) qu'on étudie est préservée. V2 (post-Sprint 1)
introduira le mode patch-pixels si nécessaire.

Format du dataset
-----------------

Chaque exemple est une séquence d'IDs de tokens, terminée par un token QUERY
qui marque le point de décision. La cible est un entier ∈ [0, 10).

```
[BOS] [DIGIT_d0] [PAD/NOISE × Δ] [OP_o1] [PAD/NOISE × Δ] [DIGIT_d1] ... [QUERY]
```

Le modèle voit la séquence entière et prédit la classe en position QUERY.

Vocabulaire
-----------

- 0..9          : DIGIT_0 .. DIGIT_9
- 10..10+K-1    : OP_0 .. OP_{K-1}      (K opérations modulaires)
- 10+K          : PAD                    (token padding inerte)
- 11+K..11+K+M-1: NOISE_0 .. NOISE_{M-1} (M tokens distracteurs structurés)
- ...           : BOS, QUERY

ℋ effectif : pour ℋ ∈ [0, 1], le ratio de tokens NOISE_* (vs PAD inerte)
parmi les distracteurs est ℋ. ℋ=0 : que des PAD. ℋ=1 : que du NOISE_*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------
# Vocabulaire
# ----------------------------------------------------------------


class Vocab:
    """Vocabulaire discret du SSG token-level."""

    def __init__(self, n_ops: int = 4, n_noise: int = 8) -> None:
        self.n_digits = 10
        self.n_ops = n_ops
        self.n_noise = n_noise

        self.DIGIT_OFFSET = 0
        self.OP_OFFSET = self.n_digits
        self.PAD = self.OP_OFFSET + self.n_ops
        self.NOISE_OFFSET = self.PAD + 1
        self.BOS = self.NOISE_OFFSET + self.n_noise
        self.QUERY = self.BOS + 1
        self.size = self.QUERY + 1

    def digit(self, d: int) -> int:
        assert 0 <= d < 10
        return self.DIGIT_OFFSET + d

    def op(self, o: int) -> int:
        assert 0 <= o < self.n_ops
        return self.OP_OFFSET + o

    def noise(self, idx: int) -> int:
        assert 0 <= idx < self.n_noise
        return self.NOISE_OFFSET + idx


# ----------------------------------------------------------------
# Opérations modulaires
# ----------------------------------------------------------------


def _op_table(n_ops: int) -> list[tuple[int, int]]:
    """Table fixe (a, b) -> op(x, d) = (a * x + b * d) mod 10. Reproductible."""
    table = [
        (1, 1),  # +
        (1, -1),  # -
        (1, 2),  # +2
        (2, 1),  # *2 + d
        (3, 0),  # *3
        (1, 3),  # +3
        (1, -2),  # -2
        (4, 1),  # *4 + d
    ]
    return table[:n_ops]


def apply_program(initial: int, ops: list[int], digits: list[int], op_table: list[tuple[int, int]]) -> int:
    x = initial
    for op_id, d in zip(ops, digits, strict=True):
        a, b = op_table[op_id]
        x = (a * x + b * d) % 10
    return x


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------


@dataclass
class StructureMNISTConfig:
    """Configuration d'un sweep ou d'un set Structure-MNIST.

    Au moins un parmi (omega, delta, entropy) doit être fixe. Les autres
    peuvent être un range pour balayage.
    """

    omega: int                  # profondeur de récursion (nombre d'ops)
    delta: int                  # nombre de distracteurs entre deux tokens informatifs
    entropy: float              # ∈ [0, 1] : ratio NOISE_*/PAD parmi distracteurs
    n_examples: int             # nombre d'exemples à générer
    n_ops: int = 4
    n_noise: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.omega >= 0
        assert self.delta >= 0
        assert 0.0 <= self.entropy <= 1.0
        assert self.n_examples > 0

    def expected_seq_len(self) -> int:
        # [BOS] + (digit + Δ + op + Δ) × ω + digit + Δ + [QUERY]
        # Pour ω=0 : [BOS] + digit + [QUERY] (pas d'opération)
        if self.omega == 0:
            return 3 + self.delta
        return 1 + (1 + self.delta + 1 + self.delta) * self.omega + 1 + self.delta + 1


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------


@dataclass
class StructureMNISTSample:
    tokens: torch.Tensor         # (L,) int64
    target: int                  # ∈ [0, 10)
    omega: int
    delta: int
    entropy: float
    info_positions: list[int]    # positions des tokens informatifs (digits + ops + query)


class StructureMNISTDataset(Dataset[StructureMNISTSample]):
    """Dataset Structure-MNIST V1 token-level.

    Chaque exemple est généré déterministiquement à partir de (seed, idx).
    """

    def __init__(self, config: StructureMNISTConfig) -> None:
        self.cfg = config
        self.vocab = Vocab(n_ops=config.n_ops, n_noise=config.n_noise)
        self.op_table = _op_table(config.n_ops)
        self.seq_len = config.expected_seq_len()

    def __len__(self) -> int:
        return self.cfg.n_examples

    def __getitem__(self, idx: int) -> StructureMNISTSample:
        rng = np.random.default_rng(seed=self.cfg.seed * 1_000_003 + idx)

        omega = self.cfg.omega
        delta = self.cfg.delta
        entropy = self.cfg.entropy

        digits: list[int] = [int(rng.integers(0, 10)) for _ in range(omega + 1)]
        ops: list[int] = [int(rng.integers(0, self.cfg.n_ops)) for _ in range(omega)]
        target = apply_program(digits[0], ops, digits[1:], self.op_table)

        tokens: list[int] = [self.vocab.BOS]
        info_positions: list[int] = [0]

        # premier digit
        tokens.append(self.vocab.digit(digits[0]))
        info_positions.append(len(tokens) - 1)

        for i in range(omega):
            self._fill_distractors(tokens, delta, entropy, rng)
            tokens.append(self.vocab.op(ops[i]))
            info_positions.append(len(tokens) - 1)
            self._fill_distractors(tokens, delta, entropy, rng)
            tokens.append(self.vocab.digit(digits[i + 1]))
            info_positions.append(len(tokens) - 1)

        # distracteurs avant query
        self._fill_distractors(tokens, delta, entropy, rng)
        tokens.append(self.vocab.QUERY)
        info_positions.append(len(tokens) - 1)

        # padding final pour aligner sur seq_len attendue
        while len(tokens) < self.seq_len:
            tokens.append(self.vocab.PAD)

        token_tensor = torch.tensor(tokens[: self.seq_len], dtype=torch.int64)
        return StructureMNISTSample(
            tokens=token_tensor,
            target=int(target),
            omega=omega,
            delta=delta,
            entropy=entropy,
            info_positions=info_positions,
        )

    def _fill_distractors(
        self, tokens: list[int], delta: int, entropy: float, rng: np.random.Generator
    ) -> None:
        for _ in range(delta):
            if rng.random() < entropy:
                tokens.append(self.vocab.noise(int(rng.integers(0, self.cfg.n_noise))))
            else:
                tokens.append(self.vocab.PAD)


# ----------------------------------------------------------------
# Tri-partition (règle des trois sets, DOC/01 §8.6)
# ----------------------------------------------------------------


@dataclass
class SplitConfig:
    """Tri-partition train_oracle / audit_svd / init_phase3 (70/20/10 par défaut)."""

    train_oracle: float = 0.70
    audit_svd: float = 0.20
    init_phase3: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_oracle + self.audit_svd + self.init_phase3
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"split must sum to 1, got {total}"


def split_indices(n_examples: int, split: SplitConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=split.seed)
    idx = rng.permutation(n_examples)
    n_train = int(split.train_oracle * n_examples)
    n_audit = int(split.audit_svd * n_examples)
    return {
        "train_oracle": idx[:n_train],
        "audit_svd": idx[n_train : n_train + n_audit],
        "init_phase3": idx[n_train + n_audit :],
    }


def sweep_monovariate(
    *,
    axis: str,
    values: list,
    base: StructureMNISTConfig,
) -> Iterator[StructureMNISTConfig]:
    """Génère une suite de configs balayant un axe avec les autres figés."""
    for v in values:
        kwargs = base.__dict__.copy()
        kwargs[axis] = v
        yield StructureMNISTConfig(**kwargs)
