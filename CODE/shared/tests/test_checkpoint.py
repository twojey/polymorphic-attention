"""Tests pour shared/checkpoint.py générique."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from shared.checkpoint import Checkpoint


def test_create_new_checkpoint(tmp_path: Path) -> None:
    cp, resumed = Checkpoint.create_or_resume(
        tmp_path / "state", fingerprint={"foo": 1, "bar": "x"},
    )
    assert resumed is False
    assert cp.state_dir.is_dir()
    # fingerprint persisté
    fp_path = cp.state_dir / "fingerprint.pkl"
    assert fp_path.exists()
    with open(fp_path, "rb") as f:
        assert pickle.load(f) == {"foo": 1, "bar": "x"}


def test_resume_compatible_fingerprint(tmp_path: Path) -> None:
    fp = {"v": 1}
    Checkpoint.create_or_resume(tmp_path / "s", fingerprint=fp)
    cp2, resumed = Checkpoint.create_or_resume(tmp_path / "s", fingerprint=fp)
    assert resumed is True


def test_incompatible_fingerprint_raises(tmp_path: Path) -> None:
    Checkpoint.create_or_resume(tmp_path / "s", fingerprint={"v": 1})
    with pytest.raises(RuntimeError, match="incompatible"):
        Checkpoint.create_or_resume(tmp_path / "s", fingerprint={"v": 2})


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={})
    obj = {"tensor": torch.tensor([1.0, 2.0, 3.0]), "scalar": 42}
    cp.save("step1", obj)
    assert cp.has("step1")
    loaded = cp.load("step1")
    assert loaded["scalar"] == 42
    assert torch.allclose(loaded["tensor"], torch.tensor([1.0, 2.0, 3.0]))


def test_save_atomic_no_partial_on_overwrite(tmp_path: Path) -> None:
    """Save écrase atomiquement — fichier final toujours valide."""
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={})
    cp.save("k", {"v": 1})
    cp.save("k", {"v": 2})  # overwrite
    assert cp.load("k") == {"v": 2}
    # Pas de fichier .tmp orphelin
    assert not (cp.state_dir / "k.pt.tmp").exists()


def test_load_missing_key_raises(tmp_path: Path) -> None:
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={})
    with pytest.raises(FileNotFoundError):
        cp.load("nonexistent")


def test_clean_removes_state(tmp_path: Path) -> None:
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={"v": 1})
    cp.save("a", {"x": 1})
    cp.save("b", {"y": 2})
    cp.clean()
    assert not cp.has("a")
    assert not cp.has("b")
    assert not (cp.state_dir / "fingerprint.pkl").exists()


def test_keys_listing(tmp_path: Path) -> None:
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={})
    assert cp.keys() == []
    cp.save("alpha", {})
    cp.save("beta", {})
    cp.save("gamma", {})
    assert cp.keys() == ["alpha", "beta", "gamma"]


def test_has_false_before_save(tmp_path: Path) -> None:
    cp, _ = Checkpoint.create_or_resume(tmp_path / "s", fingerprint={})
    assert not cp.has("anything")
