"""
sprint_f_phase4_autonomous.py — Sprint F : phase 4b autonome Spectromètre.

Spec : DOC/04_phase_routage_budget §4b.

Objectif : passer en routage **autonome** sans signaux ground-truth.
Le Spectromètre infère R_target uniquement à partir des signaux observables
phase 1.5 retenus (S_Spectral, etc.).

Critère go crucial : val_acc ≥ 0.90 × Oracle SANS ground-truth.

Pipeline :
1. Charger checkpoint Sprint E (4a warm-up terminé)
2. Lancer phase4_routage_budget.run --max-epochs-4a 0 --max-epochs-4b N
3. Construire Diagramme de Phase + Pareto sur sweep λ_budget
4. Critère go : Pareto monotone ET ≥ 1 point avec acc/oracle ≥ 0.90

Compute : ~2 jours pod GPU.
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


class SprintFPhase4Autonomous(SprintBase):
    """Sprint F — phase 4b autonome + sweep λ_budget."""

    sprint_id = "F_phase4_autonomous"
    expected_duration_hint = "2 jours pod GPU"
    expected_compute_cost = "$10-20"
    requires_pod = True

    def __init__(
        self,
        *,
        sprint_e_checkpoint: str | Path,
        config_path: str | Path,
        oracle_checkpoint: str | Path,
        signals: str = "S_Spectral",
        max_epochs_4b: int = 40,
        steps_per_epoch: int = 200,
        lambda_budgets: str = "0.001,0.003,0.01,0.03,0.1,0.3,1.0",
        oracle_baseline_acc: float = 0.645,
        target_acc_ratio: float = 0.90,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sprint_e_checkpoint = Path(sprint_e_checkpoint)
        self.config_path = Path(config_path)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.signals = signals
        self.max_epochs_4b = max_epochs_4b
        self.steps_per_epoch = steps_per_epoch
        self.lambda_budgets = lambda_budgets
        self.oracle_baseline_acc = oracle_baseline_acc
        self.target_acc_ratio = target_acc_ratio
        self.device = device

    def _run_phase4_autonomous(self) -> dict[str, Any]:
        phase4_out = self.output_dir / "phase4_4b"
        cmd = [
            sys.executable, "-m", "phase4_routage_budget.run",
            "--config", str(self.config_path),
            "--oracle-checkpoint", str(self.oracle_checkpoint),
            "--signals", self.signals,
            "--output", str(phase4_out),
            "--max-epochs-4a", "0",
            "--max-epochs-4b", str(self.max_epochs_4b),
            "--steps-per-epoch", str(self.steps_per_epoch),
            "--lambda-budgets", self.lambda_budgets,
            "--device", self.device,
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
            (self.output_dir / "phase4_stdout.log").write_text(proc.stdout)
            (self.output_dir / "phase4_stderr.log").write_text(proc.stderr)
            results_path = phase4_out / "results.json"
            if results_path.is_file():
                return {"status": "ok",
                        "results": json.loads(results_path.read_text())}
            return {"status": "ok_no_results"}
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "phase4 returncode=%s stderr=%s",
                e.returncode, e.stderr[-1000:] if e.stderr else "",
            )
            return {"status": "failed", "returncode": e.returncode}

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "sprint_e_checkpoint_exists",
            self.sprint_e_checkpoint.is_file() or self.sprint_e_checkpoint.is_dir(),
            skip_if_failed=False,
        )
        self._check_go_nogo(
            "oracle_checkpoint_exists",
            self.oracle_checkpoint.is_file(),
            skip_if_failed=True,
        )

        result = self._run_phase4_autonomous()
        self._log_metric("phase4b_status", result["status"])

        if "results" in result:
            diagrams = result["results"].get("diagrams", {})
            monotone = bool(diagrams.get("phase_diagram_monotone", False))
            pareto = diagrams.get("pareto_curve", [])
            n_pareto = len(pareto)
            target = self.oracle_baseline_acc * self.target_acc_ratio

            n_pareto_above_target = sum(
                1 for p in pareto if p.get("quality", 0.0) >= target
            )

            self._log_metric("phase_diagram_monotone", monotone)
            self._log_metric("n_pareto_points", n_pareto)
            self._log_metric("n_pareto_above_target", n_pareto_above_target)
            self._log_metric("target_acc", target)

            self._check_go_nogo(
                "phase_diagram_monotone",
                monotone, skip_if_failed=False,
            )
            self._check_go_nogo(
                f"pareto_above_target_{target:.3f}_present",
                n_pareto_above_target >= 1,
                skip_if_failed=False,
            )

        self._add_artifact(
            self.output_dir / "phase4_4b",
            "Phase 4b autonome — sweep λ_budget + Diagramme + Pareto",
        )
