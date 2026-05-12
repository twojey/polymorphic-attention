"""
sprint_g_phase5_validation.py — Sprint G : phase 5 validation 5a-6c.

Spec : DOC/05_phase_pareto.

Objectif : exécuter la suite de validation Partie 2 ASP — tests
identifiabilité (5a), élasticité (5b), self-emergent half-truth (5c),
anti-fraud (5d), OOD robustness (5e), R_max/2 (6c).

Pipeline :
1. Charger checkpoint Sprint F (modèle ASP autonome)
2. Pour chaque test : appeler CODE/phase5_pareto/test_*.py
3. Agréger résultats → verdict final Partie 2 (GO/NO-GO)

Compute : ~3 jours pod GPU.
"""

from __future__ import annotations

from pathlib import Path

from sprints.base import SprintBase


class SprintGPhase5Validation(SprintBase):
    """Sprint G — phase 5 validation finale ASP."""

    sprint_id = "G_phase5_validation"
    expected_duration_hint = "3 jours pod GPU"
    expected_compute_cost = "$15-25"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_f_checkpoint: str | Path,
        tests_to_run: tuple[str, ...] = ("5a", "5b", "5c", "5d", "5e", "6c"),
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_f_checkpoint = Path(sprint_f_checkpoint)
        self.tests_to_run = tests_to_run
        self.device = device

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_f_checkpoint_exists",
            self.sprint_f_checkpoint.is_file() or self.sprint_f_checkpoint.is_dir(),
            skip_if_failed=False,
        )

        # Lazy import phase 5
        try:
            from phase5_pareto import run as phase5_run  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"phase5_pareto : {e}")

        # Pour chaque test : appel orchestré
        results: dict[str, dict] = {}
        for test_id in self.tests_to_run:
            if self._checkpoint_has(f"test_{test_id}"):
                results[test_id] = self._checkpoint_load(f"test_{test_id}")
                continue
            # TODO Sprint G : appeler phase5_pareto.run avec --test test_id
            results[test_id] = {"status": "pending_pod"}
            self._checkpoint_save(f"test_{test_id}", results[test_id])
        self._log_metric("tests_completed", sum(1 for r in results.values()
                                                if r.get("status") != "pending_pod"))
        self._log_metric("tests_pending", sum(1 for r in results.values()
                                              if r.get("status") == "pending_pod"))

        # Verdict final
        passed = [t for t, r in results.items() if r.get("passed") is True]
        failed = [t for t, r in results.items() if r.get("passed") is False]
        pending = [t for t, r in results.items() if r.get("status") == "pending_pod"]
        self._log_metric("verdict_passed", len(passed))
        self._log_metric("verdict_failed", len(failed))
        self._log_metric("verdict_pending", len(pending))

        self._check_go_nogo(
            "ASP_validation_complete",
            len(passed) == len(self.tests_to_run),
            skip_if_failed=False,
        )

        self._add_artifact(self.output_dir / "TODO_phase5.md",
                          "Stub Sprint G — à compléter avec pod GPU")
