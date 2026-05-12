"""Tests intégration phase 5 — driver run.py end-to-end avec wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_evaluable_wrapper_forward(tmp_path: Path) -> None:
    """Wrapper retourne (logits, R_target) avec bonnes shapes, no grad."""
    from phase5_pareto.evaluable_wrapper import ASPLayerEvaluableImpl
    impl = ASPLayerEvaluableImpl(
        d_model=8, R_max=4, seq_len=16, n_classes=10,
        device="cpu", vocab_size=64,
    )
    tokens = torch.randint(0, 64, (4, 16))
    qpos = torch.tensor([15, 15, 15, 15])
    logits, R = impl.forward_eval(tokens, qpos)
    assert logits.shape == (4, 10)
    assert R.shape == (4, 16)
    assert not logits.requires_grad
    assert not R.requires_grad
    assert impl.R_max == 4


def test_evaluable_wrapper_load_or_init_missing(tmp_path: Path) -> None:
    """Si checkpoint absent → init aléatoire propre, pas de crash."""
    from phase5_pareto.evaluable_wrapper import ASPLayerEvaluableImpl
    impl = ASPLayerEvaluableImpl.load_or_init(
        checkpoint_path=tmp_path / "absent.ckpt",
        d_model=8, R_max=4, seq_len=16, n_classes=10, device="cpu",
    )
    assert impl.R_max == 4
    tokens = torch.randint(0, 64, (2, 16))
    qpos = torch.tensor([15, 15])
    logits, R = impl.forward_eval(tokens, qpos)
    assert logits.shape == (2, 10)


def test_run_phase5_smoke(tmp_path: Path) -> None:
    """End-to-end : driver phase5.run produit results.json valide."""
    asp_ckpt = tmp_path / "asp.ckpt"
    oracle_ckpt = tmp_path / "oracle.ckpt"
    asp_ckpt.write_text("fake")
    oracle_ckpt.write_text("fake")
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable, "-m", "phase5_pareto.run",
        "--asp-checkpoint", str(asp_ckpt),
        "--oracle-checkpoint", str(oracle_ckpt),
        "--output", str(out_dir),
        "--tests", "5a,5b,5c,5e,6c",
        "--device", "cpu",
        "--seq-len", "16",
        "--d-model", "8",
        "--R-max", "4",
        "--no-mlflow",
    ]
    env = {"PYTHONPATH": "CODE"}
    import os
    env.update(os.environ)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    results_path = out_dir / "results.json"
    assert results_path.is_file()
    data = json.loads(results_path.read_text())
    assert set(data.keys()) == {"5a", "5b", "5c", "5e", "6c"}
    for test_id, res in data.items():
        assert "test" in res, f"test {test_id} missing test field"
        # 5a a passed_combined (test composite), les autres ont passed
        passed_key = "passed_combined" if test_id == "5a" else "passed"
        assert passed_key in res, f"test {test_id} missing {passed_key}"
