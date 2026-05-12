"""Tests intégration robustesse end-to-end Sprint.

Vérifie qu'un Sprint complet :
- a un manifest (git hash, torch version, etc.)
- écrit un fichier log horodaté
- résiste aux crashs intermittents (retry)
- peut reprendre après un crash via checkpoint
- ne silence aucune erreur
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sprints.base import SprintBase, SprintStatus


class _ManifestSprint(SprintBase):
    sprint_id = "manifest_test"
    expected_duration_hint = "<1s"
    expected_compute_cost = "$0"
    requires_pod = False

    def _run_inner(self) -> None:
        self._log_metric("hello", "world")
        self._check_go_nogo("trivial", True)


class _CrashSprint(SprintBase):
    sprint_id = "crash_test"

    def _run_inner(self) -> None:
        raise RuntimeError("boom intentional")


class _ResumeSprint(SprintBase):
    sprint_id = "resume_test"

    def __init__(self, *, fail_first: bool = False, **kw):
        super().__init__(**kw)
        self.fail_first = fail_first

    def _run_inner(self) -> None:
        if self._checkpoint_has("step1"):
            self._log_metric("resumed", True)
        else:
            self._checkpoint_save("step1", {"value": 42})
            if self.fail_first:
                raise RuntimeError("first run fails")
        self._log_metric("step1_done", True)


def test_manifest_populated(tmp_path: Path) -> None:
    """Le manifest contient git hash, torch version, python version."""
    sprint = _ManifestSprint(output_dir=tmp_path / "out")
    result = sprint.run()
    m = result.manifest
    assert m["sprint_id"] == "manifest_test"
    assert "git_hash" in m
    assert "torch_version" in m
    assert "python_version" in m
    assert "cuda_available" in m
    assert m["requires_pod"] is False


def test_log_file_written(tmp_path: Path) -> None:
    """Un fichier sprint.log est écrit avec timestamps + level."""
    sprint = _ManifestSprint(output_dir=tmp_path / "out")
    sprint.run()
    log_path = tmp_path / "out" / "sprint.log"
    assert log_path.is_file()
    content = log_path.read_text()
    # Format attendu : 2026-...Z [INFO] sprint.manifest_test :: ...
    assert "manifest_test" in content
    assert "START" in content


def test_summary_json_includes_manifest(tmp_path: Path) -> None:
    sprint = _ManifestSprint(output_dir=tmp_path / "out")
    sprint.run()
    import json
    summary = (tmp_path / "out" / "summary.json").read_text()
    data = json.loads(summary)
    assert "manifest" in data
    assert data["manifest"]["sprint_id"] == "manifest_test"


def test_crash_does_not_silence_error(tmp_path: Path) -> None:
    """Un Sprint qui crash logue l'exception complète."""
    sprint = _CrashSprint(output_dir=tmp_path / "out")
    result = sprint.run()
    assert result.status == SprintStatus.FAILED
    assert "boom" in result.error
    # Le log doit contenir la traceback
    log_content = (tmp_path / "out" / "sprint.log").read_text()
    assert "boom intentional" in log_content
    assert "RuntimeError" in log_content
    # Summary doit avoir l'error
    import json
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert "boom" in summary["error"]


def test_checkpoint_resume_after_crash(tmp_path: Path) -> None:
    """Sprint reprend après crash : step1 est skippé au 2e run."""
    # 1er run : fail après save step1
    sprint1 = _ResumeSprint(output_dir=tmp_path / "out", fail_first=True)
    result1 = sprint1.run()
    assert result1.status == SprintStatus.FAILED
    assert sprint1._checkpoint.has("step1")
    # 2e run : pas fail, doit voir step1 déjà fait
    sprint2 = _ResumeSprint(output_dir=tmp_path / "out", fail_first=False)
    result2 = sprint2.run()
    assert result2.status == SprintStatus.SUCCESS
    assert result2.metrics.get("resumed") is True


def test_go_nogo_decisions_logged(tmp_path: Path) -> None:
    """Les décisions go/no-go sont tracées dans le summary + log."""
    class _MultiCriterion(SprintBase):
        sprint_id = "multi_criterion"

        def _run_inner(self) -> None:
            self._check_go_nogo("c1", True)
            self._check_go_nogo("c2", False)  # ne skip pas
            self._check_go_nogo("c3", True)

    sprint = _MultiCriterion(output_dir=tmp_path / "out")
    result = sprint.run()
    assert len(result.go_nogo_decisions) == 3
    log = (tmp_path / "out" / "sprint.log").read_text()
    assert "c1 : PASS" in log
    assert "c2 : FAIL" in log
    assert "c3 : PASS" in log


def test_metric_log_persisted_in_log_file(tmp_path: Path) -> None:
    """_log_metric écrit aussi dans le sprint.log."""
    sprint = _ManifestSprint(output_dir=tmp_path / "out")
    sprint.run()
    log = (tmp_path / "out" / "sprint.log").read_text()
    assert "[ckpt] metric hello=world" in log


def test_checkpoint_fingerprint_mismatch_raises(tmp_path: Path) -> None:
    """Changer seed → fingerprint mismatch → erreur explicite (pas silencieuse)."""
    sprint1 = _ResumeSprint(output_dir=tmp_path / "out", fail_first=False, seed=0)
    sprint1.run()
    # 2e Sprint avec seed différent : le checkpoint refuse de reprendre
    with pytest.raises(RuntimeError, match="Checkpoint state incompatible"):
        _ResumeSprint(output_dir=tmp_path / "out", fail_first=False, seed=99)
