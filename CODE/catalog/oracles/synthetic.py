"""
synthetic.py — Oracle synthétique pour tests (génère attention softmax aléatoire).

Permet de tester l'interface Battery + Properties sans dépendre d'un vrai
Oracle entraîné. Génère des matrices d'attention contrôlées :
- structure paramétrable (toeplitz_ish, hankel_ish, random)
- rang contrôlé (low_rank avec r imposé)
- causalité optionnelle
"""

from __future__ import annotations

import torch

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


class SyntheticOracle(AbstractOracle):
    """Oracle qui génère des matrices d'attention synthétiques contrôlées.

    Utilisé pour tests d'intégration de la Battery sans Oracle entraîné.
    Le paramètre `structure` injecte une signature mathématique connue
    pour valider que les Properties la détectent.
    """

    domain = "synthetic"

    def __init__(
        self,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        d_model: int = 32,  # informatif, pas utilisé directement (matrices N×N)
        seq_len: int = 16,
        structure: str = "random",  # "random" | "low_rank" | "toeplitz" | "hankel"
        target_rank: int = 4,
        seed: int = 0,
        oracle_id: str | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.structure = structure
        self.target_rank = target_rank
        self.seed = seed
        self.oracle_id = oracle_id or f"synthetic_{structure}_r{target_rank}_seed{seed}"

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Génère n_examples matrices d'attention pour ce régime.

        Le régime n'a pas d'effet sur la structure générée (Synthetic ne
        simule pas un vrai sweep — juste des matrices contrôlées par
        `self.structure`). Les axes omegas/deltas/entropies du dump sont
        remplis avec les valeurs du régime pour traçabilité.
        """
        g = torch.Generator().manual_seed(self.seed + hash(regime.key) % (2**31))
        N = self.seq_len
        attn_per_layer: list[torch.Tensor] = []

        for ell in range(self.n_layers):
            A = self._generate_attention(n_examples, N, generator=g, layer=ell)
            attn_per_layer.append(A)

        omegas = torch.full(
            (n_examples,), regime.omega if regime.omega is not None else 0,
            dtype=torch.int64,
        )
        deltas = torch.full(
            (n_examples,), regime.delta if regime.delta is not None else 0,
            dtype=torch.int64,
        )
        entropies = torch.full(
            (n_examples,), regime.entropy if regime.entropy is not None else 0.0,
            dtype=torch.float32,
        )
        return AttentionDump(
            attn=attn_per_layer,
            omegas=omegas, deltas=deltas, entropies=entropies,
            metadata={"oracle_id": self.oracle_id, "structure": self.structure,
                      "target_rank": self.target_rank, "regime": regime.key},
        )

    def _generate_attention(
        self, B: int, N: int, *, generator: torch.Generator, layer: int
    ) -> torch.Tensor:
        """Génère (B, H, N, N) selon la structure demandée."""
        if self.structure == "random":
            raw = torch.randn(B, self.n_heads, N, N, generator=generator, dtype=torch.float64)
            return torch.softmax(raw, dim=-1)
        if self.structure == "low_rank":
            # rank-target_rank : softmax(U V^T) avec U, V de rang target_rank
            r = self.target_rank
            U = torch.randn(B, self.n_heads, N, r, generator=generator, dtype=torch.float64)
            V = torch.randn(B, self.n_heads, N, r, generator=generator, dtype=torch.float64)
            scores = U @ V.transpose(-2, -1)
            return torch.softmax(scores, dim=-1)
        if self.structure == "toeplitz":
            # Toeplitz approximative : c[i-j] avec c généré aléatoirement
            c = torch.randn(B, self.n_heads, 2 * N - 1, generator=generator, dtype=torch.float64)
            scores = torch.zeros(B, self.n_heads, N, N, dtype=torch.float64)
            for i in range(N):
                for j in range(N):
                    scores[..., i, j] = c[..., (i - j) + (N - 1)]
            return torch.softmax(scores, dim=-1)
        if self.structure == "hankel":
            # Hankel : h[i+j]
            h = torch.randn(B, self.n_heads, 2 * N - 1, generator=generator, dtype=torch.float64)
            scores = torch.zeros(B, self.n_heads, N, N, dtype=torch.float64)
            for i in range(N):
                for j in range(N):
                    scores[..., i, j] = h[..., i + j]
            return torch.softmax(scores, dim=-1)
        raise ValueError(f"structure inconnue : {self.structure!r}")

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille minimale pour tests : 3 régimes monovariate sur omega."""
        return [
            RegimeSpec(omega=0, delta=16, entropy=0.0),
            RegimeSpec(omega=2, delta=16, entropy=0.0),
            RegimeSpec(omega=4, delta=16, entropy=0.0),
        ]
