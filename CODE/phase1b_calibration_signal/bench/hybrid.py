"""
hybrid.py — banc hybride 50% SSG variant + 50% bruit pur.

Spec : DOC/01b §2.1.

50% SSG :
- ω ∈ {2, 4, 6, 8, 10, 12}
- Δ ∈ {64, 256, 1024, 4096}
- ℋ basse (peu de bruit)

50% structure absente :
- (ω, Δ) = (0, 0)
- ℋ étalée sur toute sa plage

Mélange par séquence (pas par batch) — chaque token est évalué dans son
propre contexte.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from phase1_metrologie.ssg.structure_mnist import StructureMNISTConfig, StructureMNISTDataset, StructureMNISTSample


@dataclass
class HybridBenchConfig:
    n_examples: int = 2000
    seed: int = 0
    seq_len: int | None = None  # si fourni, force un seq_len constant en padding/troncage

    structured_omegas: tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    structured_deltas: tuple[int, ...] = (64, 256, 1024, 4096)
    structured_entropy_max: float = 0.2  # ℋ "basse"

    noise_entropies: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


class HybridBenchDataset(Dataset[StructureMNISTSample]):
    """Banc hybride. La moitié des exemples ont ω, Δ > 0, ℋ basse ; l'autre
    moitié ont ω=Δ=0 et ℋ étalée.
    """

    def __init__(self, cfg: HybridBenchConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n_struct = cfg.n_examples // 2
        n_noise = cfg.n_examples - n_struct

        self._items: list[StructureMNISTConfig] = []
        for _ in range(n_struct):
            omega = int(rng.choice(cfg.structured_omegas))
            delta = int(rng.choice(cfg.structured_deltas))
            entropy = float(rng.uniform(0.0, cfg.structured_entropy_max))
            self._items.append(
                StructureMNISTConfig(omega=omega, delta=delta, entropy=entropy, n_examples=1, seed=int(rng.integers(0, 10**9)))
            )
        for _ in range(n_noise):
            entropy = float(rng.choice(cfg.noise_entropies))
            self._items.append(
                StructureMNISTConfig(omega=0, delta=0, entropy=entropy, n_examples=1, seed=int(rng.integers(0, 10**9)))
            )

        # mélange par séquence
        random.Random(cfg.seed + 1).shuffle(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> StructureMNISTSample:
        cfg = self._items[idx]
        sample = StructureMNISTDataset(cfg)[0]
        if self.cfg.seq_len is not None:
            sample = self._fix_length(sample, self.cfg.seq_len)
        return sample

    def _fix_length(self, sample: StructureMNISTSample, target_len: int) -> StructureMNISTSample:
        from phase1_metrologie.ssg.structure_mnist import Vocab

        vocab = Vocab()
        L = sample.tokens.numel()
        if L == target_len:
            return sample
        if L > target_len:
            # tronque depuis la fin (en gardant BOS au début, on coupe avant QUERY ?)
            # Mieux : on tronque la queue PAD si possible, sinon on tronque les distracteurs.
            tokens = sample.tokens[:target_len]
            return StructureMNISTSample(
                tokens=tokens, target=sample.target, omega=sample.omega,
                delta=sample.delta, entropy=sample.entropy, info_positions=sample.info_positions,
            )
        # padding à droite avec PAD
        pad = torch.full((target_len - L,), vocab.PAD, dtype=sample.tokens.dtype)
        tokens = torch.cat([sample.tokens, pad], dim=0)
        return StructureMNISTSample(
            tokens=tokens, target=sample.target, omega=sample.omega,
            delta=sample.delta, entropy=sample.entropy, info_positions=sample.info_positions,
        )


def build_hybrid_bench(cfg: HybridBenchConfig) -> HybridBenchDataset:
    return HybridBenchDataset(cfg)
