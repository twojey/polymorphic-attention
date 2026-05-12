"""Tests SprintBase + runners squelettes."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprints.base import SprintBase, SprintStatus, _SprintSkipped


class _DummySuccess(SprintBase):
    sprint_id = "dummy_success"
    expected_duration_hint = "1s"
    expected_compute_cost = "$0"
    requires_pod = False

    def _run_inner(self) -> None:
        self._log_metric("ok", 42)
        self._check_go_nogo("trivially_true", True)


class _DummySkip(SprintBase):
    sprint_id = "dummy_skip"

    def _run_inner(self) -> None:
        self._check_go_nogo("trivially_false", False, skip_if_failed=True)


class _DummyFail(SprintBase):
    sprint_id = "dummy_fail"

    def _run_inner(self) -> None:
        raise ValueError("intentional")


def test_sprint_success(tmp_path: Path) -> None:
    sprint = _DummySuccess(output_dir=tmp_path / "out")
    result = sprint.run()
    assert result.status == SprintStatus.SUCCESS
    assert result.metrics["ok"] == 42
    assert len(result.go_nogo_decisions) == 1
    assert result.go_nogo_decisions[0]["passed"] is True


def test_sprint_skipped(tmp_path: Path) -> None:
    sprint = _DummySkip(output_dir=tmp_path / "out")
    result = sprint.run()
    assert result.status == SprintStatus.SKIPPED
    assert "trivially_false" in result.error


def test_sprint_failed(tmp_path: Path) -> None:
    sprint = _DummyFail(output_dir=tmp_path / "out")
    result = sprint.run()
    assert result.status == SprintStatus.FAILED
    assert "intentional" in result.error


def test_sprint_writes_summary(tmp_path: Path) -> None:
    sprint = _DummySuccess(output_dir=tmp_path / "out")
    result = sprint.run()
    summary = tmp_path / "out" / "summary.json"
    assert summary.is_file()
    import json
    with open(summary) as f:
        data = json.load(f)
    assert data["sprint_id"] == "dummy_success"
    assert data["status"] == "success"


def test_sprint_checkpoint_resume(tmp_path: Path) -> None:
    """Le checkpoint persiste entre runs : 2e instance peut reprendre."""
    sprint1 = _DummySuccess(output_dir=tmp_path / "out")
    sprint1._run_inner = lambda: sprint1._checkpoint_save("step1", {"value": 100})  # type: ignore[assignment]
    sprint1.run()
    sprint2 = _DummySuccess(output_dir=tmp_path / "out")
    assert sprint2._checkpoint_has("step1")
    state = sprint2._checkpoint_load("step1")
    assert state["value"] == 100


def test_sprint_registry_imports() -> None:
    """Tous les Sprint runners s'importent sans erreur."""
    from sprints.run import REGISTRY
    expected_ids = {"B", "C", "D", "E", "F", "G", "S4", "S5", "S6", "S7"}
    assert set(REGISTRY.keys()) == expected_ids
    for key, cls in REGISTRY.items():
        assert hasattr(cls, "sprint_id")
        assert hasattr(cls, "_run_inner")
