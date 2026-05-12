"""
sprint_f_phase4_autonomous.py — Sprint F : phase 4b autonomous routing.

Spec : DOC/04_phase_routage_budget §4b.

Objectif : faire fonctionner l'ASPLayer SANS ground truth ω/Δ/ℋ.
Le Spectromètre infère r à partir du contexte uniquement, puis route
l'attention via R_max(r_pred). C'est le test crucial Partie 2.

Pipeline :
1. Repartir du checkpoint Sprint E (warm-up)
2. Switch mode='autonomous' : signaux ω/Δ/ℋ retirés du input
3. Re-train + évaluer val_acc + corrélation r_pred ground truth
4. Critère go : val_acc ≥ 0.90 × Oracle ET routing latency reasonable

Compute : ~2 jours pod GPU.
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintFPhase4Autonomous(SprintBase):
    """Sprint F — phase 4b autonomous routing."""

    sprint_id = "F_phase4_autonomous"
    expected_duration_hint = "2 jours pod GPU"
    expected_compute_cost = "$10-20"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_e_checkpoint: str | Path,
        n_epochs: int = 50,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_e_checkpoint = Path(sprint_e_checkpoint)
        self.n_epochs = n_epochs
        self.device = device

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_e_checkpoint_exists",
            self.sprint_e_checkpoint.is_file() or self.sprint_e_checkpoint.is_dir(),
            skip_if_failed=False,
        )

        try:
            import phase4_routage_budget.run as phase4_run  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"phase4_routage_budget : {e}")

        # TODO Sprint F : training autonomous via phase4_routage_budget.run
        # avec mode='autonomous', signaux ground truth masqués.
        self._log_metric("status", "phase4b_autonomous_training_pending_pod")
        self._add_artifact(self.output_dir / "TODO_phase4b.md",
                          "Stub Sprint F — à compléter avec pod GPU")
