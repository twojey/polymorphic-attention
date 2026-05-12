"""
sprint_g_phase5_validation.py — Sprint G : phase 5 validation 5a-6c.

Spec : DOC/05_phase_pareto.

Objectif : exécuter la suite de validation Partie 2 ASP — tests
identifiabilité (5a), élasticité (5b), SE/HT (5c), OOD (5e), R_max/2 (6c).

Pipeline :
1. Charger checkpoint Sprint F (modèle ASP autonome)
2. Appeler phase5_pareto.run --tests 5a,5b,5c,5e,6c → results.json
3. Parser verdict par test
4. Verdict final ASP : mandatory 5a + 5c + 6c (autres bonus)

Compute : ~3 jours pod GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.retry import retry_call
from sprints.base import SprintBase


# Tests mandatoires pour verdict ASP positif
MANDATORY_TESTS = ("5a", "5c", "6c")


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
        oracle_checkpoint: str | Path,
        tests_to_run: tuple[str, ...] = ("5a", "5b", "5c", "5e", "6c"),
        seq_len: int = 32,
        d_model: int = 32,
        R_max: int = 8,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_f_checkpoint = Path(sprint_f_checkpoint)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.tests_to_run = tests_to_run
        self.seq_len = seq_len
        self.d_model = d_model
        self.R_max = R_max
        self.device = device

    def _run_phase5(self) -> dict[str, Any]:
        phase5_out = self.output_dir / "phase5"
        cmd = [
            sys.executable, "-m", "phase5_pareto.run",
            "--asp-checkpoint", str(self.sprint_f_checkpoint),
            "--oracle-checkpoint", str(self.oracle_checkpoint),
            "--output", str(phase5_out),
            "--tests", ",".join(self.tests_to_run),
            "--device", self.device,
            "--seq-len", str(self.seq_len),
            "--d-model", str(self.d_model),
            "--R-max", str(self.R_max),
            "--no-mlflow",
        ]
        self.logger.info("[%s] cmd: %s", self.sprint_id, " ".join(cmd))

        env = {"PYTHONPATH": "CODE"}
        env.update(os.environ)

        def _exec():
            return subprocess.run(
                cmd, env=env, check=True, capture_output=True, text=True,
                timeout=24 * 3600,
            )

        try:
            proc = retry_call(
                _exec, args=(), max_attempts=2, base_delay=10.0,
                jitter=1.0, logger=self.logger,
            )
            (self.output_dir / "phase5_stdout.log").write_text(proc.stdout)
            (self.output_dir / "phase5_stderr.log").write_text(proc.stderr)
            results_path = phase5_out / "results.json"
            if results_path.is_file():
                return {"status": "ok",
                        "results": json.loads(results_path.read_text())}
            return {"status": "ok_no_results"}
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "phase5 returncode=%s stderr=%s",
                e.returncode, e.stderr[-1000:] if e.stderr else "",
            )
            return {"status": "failed", "returncode": e.returncode}

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_f_checkpoint_exists",
            self.sprint_f_checkpoint.is_file() or self.sprint_f_checkpoint.is_dir(),
            skip_if_failed=False,
        )

        result = self._run_phase5()
        self._log_metric("phase5_status", result["status"])

        results_per_test: dict[str, dict] = result.get("results", {})
        passed = [t for t, r in results_per_test.items()
                  if r.get("passed") is True]
        failed = [t for t, r in results_per_test.items()
                  if r.get("passed") is False]

        self._log_metric("n_tests_passed", len(passed))
        self._log_metric("n_tests_failed", len(failed))
        self._log_metric("tests_passed_list", ",".join(passed))
        self._log_metric("tests_failed_list", ",".join(failed))

        # Verdict mandatoires (5a + 5c + 6c)
        mandatory_passed = [
            t for t in MANDATORY_TESTS
            if results_per_test.get(t, {}).get("passed") is True
        ]
        verdict_asp_positive = len(mandatory_passed) == len(MANDATORY_TESTS)
        self._log_metric("mandatory_passed",
                         ",".join(mandatory_passed))
        self._log_metric("verdict_asp_positive", verdict_asp_positive)

        # Bonus : 5b + 5e (non bloquants mais bonus)
        bonus = [t for t in ("5b", "5e") if t in passed]
        self._log_metric("bonus_passed", ",".join(bonus))

        self._check_go_nogo(
            "mandatory_tests_5a_5c_6c_passed",
            verdict_asp_positive,
            skip_if_failed=False,
        )

        self._add_artifact(
            self.output_dir / "phase5",
            "Phase 5 validation finale (tests 5a-6c)",
        )
