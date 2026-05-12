"""Tests intégration phase 4 — driver run.py end-to-end smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_phase4_smoke(tmp_path: Path) -> None:
    """phase4.run produit results.json avec phase_4a + phase_4b + diagrams."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("phase4:\n  R_max: 4\n")
    oracle_ckpt = tmp_path / "oracle.ckpt"
    oracle_ckpt.write_text("fake")
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable, "-m", "phase4_routage_budget.run",
        "--config", str(config_path),
        "--oracle-checkpoint", str(oracle_ckpt),
        "--signals", "S_Spectral",
        "--output", str(out_dir),
        "--max-epochs-4a", "1",
        "--max-epochs-4b", "1",
        "--steps-per-epoch", "3",
        "--batch-size", "4",
        "--seq-len", "12",
        "--d-model", "8",
        "--R-max", "4",
        "--lambda-budgets", "0.01,0.1",
        "--device", "cpu",
        "--no-mlflow",
    ]
    env = {"PYTHONPATH": "CODE"}
    env.update(os.environ)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    results_path = out_dir / "results.json"
    assert results_path.is_file()
    data = json.loads(results_path.read_text())
    assert "phase_4a" in data
    assert "phase_4b" in data
    assert "diagrams" in data
    assert "loss_task" in data["phase_4a"]
    assert "per_lambda" in data["phase_4b"]
    assert len(data["phase_4b"]["per_lambda"]) == 2
    diag = data["diagrams"]
    assert "phase_diagram" in diag
    assert "pareto_curve" in diag
    assert isinstance(diag["phase_diagram_monotone"], bool)


def test_run_phase4_resume(tmp_path: Path) -> None:
    """Run + relance : checkpoint resume détecté."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("phase4:\n  R_max: 4\n")
    oracle_ckpt = tmp_path / "oracle.ckpt"
    oracle_ckpt.write_text("fake")
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable, "-m", "phase4_routage_budget.run",
        "--config", str(config_path),
        "--oracle-checkpoint", str(oracle_ckpt),
        "--signals", "S_Spectral",
        "--output", str(out_dir),
        "--max-epochs-4a", "1",
        "--max-epochs-4b", "1",
        "--steps-per-epoch", "2",
        "--batch-size", "4",
        "--seq-len", "8",
        "--d-model", "8",
        "--R-max", "4",
        "--lambda-budgets", "0.01",
        "--device", "cpu",
        "--no-mlflow",
    ]
    env = {"PYTHONPATH": "CODE"}
    env.update(os.environ)
    # First run
    proc1 = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=120)
    assert proc1.returncode == 0
    # Second run should see SKIP markers via checkpoint
    proc2 = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=120)
    assert proc2.returncode == 0
    assert "SKIP" in proc2.stdout
