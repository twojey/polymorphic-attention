"""
sprint_b_re_extract.py — Sprint B : re-extraction phase 1 V2 dumps SMNIST.

Spec : DOC/ROADMAP §3.9 + DOC/01_phase_metrologie.

Objectif : générer les 9 dumps multi-bucket (ω × Δ × ℋ) sur SMNIST Oracle
V2 conforme phase 1 V2. Pré-requis Sprint C (catalog level_research).

Pipeline :
1. Charger checkpoint SMNIST Oracle (issu phase 1 entraîné)
2. Sélectionner sweep régimes 3 × 3 (omega ∈ {0, 2, 4}, delta ∈ {16, 64, 256})
3. Pour chaque (ω, Δ) : extract_regime(n_examples=512) → AttentionDump FP64
4. Sauvegarder en NPZ pour reuse Sprint C

Compute : ~30 min CPU pod, ~$0.30 (RunPod CPU).
Wall-clock : 30 min - 1 h selon disponibilité pod.
"""

from __future__ import annotations

from pathlib import Path

import torch

from shared.retry import retry_call
from sprints.base import SprintBase


class SprintBReExtract(SprintBase):
    """Sprint B — re-extraction dumps phase 1 V2."""

    sprint_id = "B_re_extract"
    expected_duration_hint = "30 min - 1 h sur pod CPU"
    expected_compute_cost = "$0.30 (RunPod CPU)"
    requires_pod = True  # pod recommandé (CPU suffit)

    def __init__(
        self,
        *,
        oracle_checkpoint: str | Path,
        omegas: tuple[int, ...] = (0, 2, 4),
        deltas: tuple[int, ...] = (16, 64, 256),
        n_examples_per_regime: int = 512,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.omegas = omegas
        self.deltas = deltas
        self.n_examples = n_examples_per_regime
        self.device = device

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "oracle_checkpoint_exists",
            self.oracle_checkpoint.is_file(),
            skip_if_failed=True,
        )

        # Lazy import phase 1
        from catalog.oracles import SMNISTOracle
        from catalog.oracles.base import RegimeSpec

        oracle = SMNISTOracle(checkpoint_path=str(self.oracle_checkpoint),
                              device=self.device)
        self._log_metric("oracle_id", oracle.oracle_id)
        self._log_metric("n_layers", oracle.n_layers)
        self._log_metric("n_heads", oracle.n_heads)

        # Boucle régimes avec checkpoint/resume + retry
        dumps_dir = self.output_dir / "dumps"
        dumps_dir.mkdir(exist_ok=True)
        n_done = 0
        n_skipped = 0
        for omega in self.omegas:
            for delta in self.deltas:
                key = f"dump_omega{omega}_delta{delta}"
                if self._checkpoint_has(key):
                    self.logger.info("[%s] %s : déjà extrait (resume)", self.sprint_id, key)
                    n_done += 1
                    continue
                regime = RegimeSpec(omega=omega, delta=delta, entropy=0.0)
                self.logger.info("[%s] extracting régime (ω=%d, Δ=%d)…",
                                 self.sprint_id, omega, delta)
                try:
                    # Retry 3× : extract_regime peut échouer transitoirement
                    # (OOM ponctuelle, race condition Oracle, etc.)
                    dump = retry_call(
                        oracle.extract_regime,
                        args=(regime, self.n_examples),
                        max_attempts=3, base_delay=1.0, jitter=0.5,
                        logger=self.logger,
                    )
                    dump.validate()
                except Exception as e:
                    self.logger.error(
                        "[%s] skip régime (ω=%d, Δ=%d) après 3 retries : %s",
                        self.sprint_id, omega, delta, e,
                    )
                    n_skipped += 1
                    continue
                # Save dump
                dump_path = dumps_dir / f"{key}.pt"
                torch.save({
                    "attn": dump.attn,
                    "omegas": dump.omegas,
                    "deltas": dump.deltas,
                    "entropies": dump.entropies,
                    "tokens": dump.tokens,
                    "query_pos": dump.query_pos,
                    "metadata": dump.metadata,
                }, dump_path)
                self._checkpoint_save(key, str(dump_path))
                self._add_artifact(dump_path, f"dump (ω={omega}, Δ={delta})")
                n_done += 1

        self._log_metric("n_dumps_produced", n_done)
        self._log_metric("n_regimes_target", len(self.omegas) * len(self.deltas))
        self._log_metric("n_regimes_skipped", n_skipped)

        self._check_go_nogo(
            "min_dumps_produced",
            n_done >= len(self.omegas) * len(self.deltas) - 1,
            skip_if_failed=False,
        )
