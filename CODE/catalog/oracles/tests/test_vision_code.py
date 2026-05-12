"""Tests squelettes VisionOracle + CodeOracle."""

from __future__ import annotations

import pytest


def test_vision_imports() -> None:
    from catalog.oracles import VisionOracle, VisionModelSpec
    assert VisionOracle.domain == "vision"


def test_vision_rejects_missing_ckpt(tmp_path) -> None:
    from catalog.oracles import VisionOracle
    with pytest.raises(FileNotFoundError):
        VisionOracle(checkpoint_path=tmp_path / "nope.pt")


def test_vision_regime_grid_non_empty(tmp_path) -> None:
    from catalog.oracles import VisionOracle
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    oracle = VisionOracle(checkpoint_path=ckpt)
    grid = oracle.regime_grid()
    assert len(grid) == 9  # 3 patch_sizes × 3 n_classes


def test_vision_extract_raises_v1(tmp_path) -> None:
    from catalog.oracles import VisionOracle
    from catalog.oracles.base import RegimeSpec
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    oracle = VisionOracle(checkpoint_path=ckpt)
    with pytest.raises(NotImplementedError, match="Sprint S5"):
        oracle.extract_regime(RegimeSpec(omega=4, delta=10, entropy=0.0), n_examples=2)


def test_code_imports() -> None:
    from catalog.oracles import CodeOracle, CodeModelSpec
    assert CodeOracle.domain == "code"


def test_code_rejects_missing_ckpt(tmp_path) -> None:
    from catalog.oracles import CodeOracle
    with pytest.raises(FileNotFoundError):
        CodeOracle(checkpoint_path=tmp_path / "nope.pt")


def test_code_regime_grid_non_empty(tmp_path) -> None:
    from catalog.oracles import CodeOracle
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    oracle = CodeOracle(checkpoint_path=ckpt)
    grid = oracle.regime_grid()
    assert len(grid) == 12  # 4 depths × 3 seq_lens


def test_code_extract_raises_v1(tmp_path) -> None:
    from catalog.oracles import CodeOracle
    from catalog.oracles.base import RegimeSpec
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    oracle = CodeOracle(checkpoint_path=ckpt)
    with pytest.raises(NotImplementedError, match="Sprint S6"):
        oracle.extract_regime(RegimeSpec(omega=2, delta=64, entropy=0.0), n_examples=2)
