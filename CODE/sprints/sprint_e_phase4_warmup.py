"""
sprint_e_phase4_warmup.py — Sprint E : phase 4a warm-up Spectromètre.

Spec : DOC/04_phase_routage_budget §4a.

Objectif : entraîner le Spectromètre conjointement avec l'ASPLayer dans
un setting "warm-up" : signaux ω, Δ, ℋ fournis directement comme features
input (ground truth, pas d'inférence).

Pipeline :
1. Initialiser ASPLayer Sprint D + Spectromètre (CODE/phase4_routage_budget/spectrometer.py)
2. Training conjoint avec curriculum (DOC/04 §curriculum)
3. Loss : task_loss + λ_spec · L_R (Spectromètre prédit r ground truth)
4. Critère go : val_acc ≥ 0.95 × Oracle ET corrélation Spearman(r_pred, r_target) > 0.7

Compute : ~1 jour pod GPU.
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintEPhase4Warmup(SprintBase):
    """Sprint E — phase 4a warm-up Spectromètre."""

    sprint_id = "E_phase4_warmup"
    expected_duration_hint = "1 jour pod GPU"
    expected_compute_cost = "$5-10"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_d_checkpoint: str | Path,
        n_epochs: int = 30,
        lambda_spec: float = 1.0,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_d_checkpoint = Path(sprint_d_checkpoint)
        self.n_epochs = n_epochs
        self.lambda_spec = lambda_spec
        self.device = device

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_d_checkpoint_exists",
            self.sprint_d_checkpoint.is_file() or self.sprint_d_checkpoint.is_dir(),
            skip_if_failed=False,
        )

        # Lazy import phase 4
        try:
            import phase4_routage_budget.run as phase4_run  # noqa: F401
            from phase4_routage_budget.spectrometer import Spectrometer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"phase4_routage_budget : {e}")

        # TODO Sprint E : lancer training warm-up via phase4_routage_budget.run
        # avec mode='warmup' et λ_spec.
        self._log_metric("status", "phase4a_warmup_training_pending_pod")
        self._add_artifact(self.output_dir / "TODO_phase4a.md",
                          "Stub Sprint E — à compléter avec pod GPU")
