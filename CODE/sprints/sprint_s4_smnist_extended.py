"""
sprint_s4_smnist_extended.py — Sprint S4 : SMNIST seq_len étendu.

Spec : DOC/CATALOGUE §3.3.

Objectif : étendre le sweep SMNIST aux seq_len = 1024, 2048, 4096 pour
valider la dépendance long-range. Re-extract phase 1 avec configs
plus longues.

Pipeline :
1. Phase 1 SMNIST avec configs étendues (delta_max=4096)
2. Battery research × régimes étendus
3. Comparer signatures r_eff(delta) pour scaling delta

Compute : ~2 jours pod CPU/GPU léger.
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintS4SMNISTExtended(SprintBase):
    """Sprint S4 — SMNIST seq_len étendu jusque 4096."""

    sprint_id = "S4_smnist_extended"
    expected_duration_hint = "2 jours pod"
    expected_compute_cost = "$5"
    requires_pod = True

    def __init__(
        self,
        *,
        oracle_checkpoint: str | Path,
        deltas_extended: tuple[int, ...] = (1024, 2048, 4096),
        omegas: tuple[int, ...] = (0, 2, 4),
        n_examples_per_regime: int = 128,
        device: str = "cpu",
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
        # TODO Sprint S4 : re-extract phase 1 avec configs delta étendues
        # + Battery research sur dumps long-range.
        self._add_artifact(self.output_dir / "TODO_s4.md",
                          "Stub Sprint S4 — à compléter avec pod")
