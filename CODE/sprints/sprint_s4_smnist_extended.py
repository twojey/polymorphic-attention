"""
sprint_s4_smnist_extended.py — Sprint S4 : SMNIST seq_len étendu.

Spec : DOC/CATALOGUE §3.3.

Objectif : étendre le sweep SMNIST aux Δ ∈ {1024, 2048, 4096} pour valider
la dépendance long-range. Re-extract phase 1 V2 avec configs plus longues
sur SMNIST Oracle déjà entraîné.

Pipeline (calque sur Sprint B, mais avec Δ longs) :
1. Charger checkpoint SMNIST Oracle
2. Sélectionner sweep ω × Δ (Δ étendu, ω court pour limiter compute)
3. extract_regime par régime (avec retry transients)
4. Sauver dumps pour Sprint C battery

Compute : ~1-2 jours pod GPU (Δ=4096 → ~16k seq_len ~ saturation).
"""

from __future__ import annotations

from pathlib import Path

import torch

from shared.retry import retry_call
from sprints.base import SprintBase


class SprintS4SMNISTExtended(SprintBase):
    """Sprint S4 — SMNIST Δ étendu (long-context)."""

    sprint_id = "S4_smnist_extended"
    expected_duration_hint = "1-2 jours pod GPU léger"
    expected_compute_cost = "$5-10"
    requires_pod = True

    def __init__(
        self,
        *,
        oracle_checkpoint: str | Path,
        deltas_extended: tuple[int, ...] = (1024, 2048, 4096),
        omegas: tuple[int, ...] = (0, 2),
        n_examples_per_regime: int = 64,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.deltas_extended = deltas_extended
        self.omegas = omegas
        self.n_examples = n_examples_per_regime
        self.device = device

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "oracle_checkpoint_exists",
            self.oracle_checkpoint.is_file(),
            skip_if_failed=True,
        )
        self._log_metric("max_delta", max(self.deltas_extended))
        self._log_metric("n_regimes_target",
                         len(self.omegas) * len(self.deltas_extended))

        from catalog.oracles import SMNISTOracle
        from catalog.oracles.base import RegimeSpec

        oracle = SMNISTOracle(
            checkpoint_path=str(self.oracle_checkpoint),
            device=self.device,
        )
        self._log_metric("oracle_id", oracle.oracle_id)

        dumps_dir = self.output_dir / "dumps"
        dumps_dir.mkdir(exist_ok=True)
        n_done = 0
        n_skipped = 0
        for omega in self.omegas:
            for delta in self.deltas_extended:
                key = f"dump_omega{omega}_delta{delta}"
                if self._checkpoint_has(key):
                    self.logger.info("[%s] %s : déjà extrait (resume)",
                                     self.sprint_id, key)
                    n_done += 1
                    continue
                regime = RegimeSpec(omega=omega, delta=delta, entropy=0.0)
                self.logger.info(
                    "[%s] extracting régime long (ω=%d, Δ=%d, n=%d)…",
                    self.sprint_id, omega, delta, self.n_examples,
                )
                try:
                    dump = retry_call(
                        oracle.extract_regime,
                        args=(regime, self.n_examples),
                        max_attempts=3,
                        base_delay=2.0,
                        jitter=0.5,
                        logger=self.logger,
                    )
                    dump.validate()
                except Exception as e:  # noqa: BLE001
                    self.logger.error(
                        "[%s] skip régime (ω=%d, Δ=%d) après retries : %s",
                        self.sprint_id, omega, delta, e,
                    )
                    n_skipped += 1
                    continue

                dump_path = dumps_dir / f"{key}.pt"
                torch.save(
                    {
                        "attn": dump.attn,
                        "omegas": dump.omegas,
                        "deltas": dump.deltas,
                        "entropies": dump.entropies,
                        "tokens": dump.tokens,
                        "query_pos": dump.query_pos,
                        "metadata": dump.metadata,
                    },
                    dump_path,
                )
                self._checkpoint_save(key, str(dump_path))
                self._add_artifact(
                    dump_path, f"dump long (ω={omega}, Δ={delta})"
                )
                n_done += 1

        self._log_metric("n_dumps_produced", n_done)
        self._log_metric("n_regimes_skipped", n_skipped)

        # Critère go : au moins ⌊n_target/2⌋ extraits (Δ=4096 peut OOM)
        n_target = len(self.omegas) * len(self.deltas_extended)
        self._check_go_nogo(
            "min_dumps_produced",
            n_done >= max(1, n_target // 2),
            skip_if_failed=False,
        )
