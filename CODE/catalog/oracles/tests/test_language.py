"""Tests squelette LLOracle (V1 sans training)."""

from __future__ import annotations

import pytest

from catalog.oracles.base import RegimeSpec


def test_ll_oracle_imports() -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    assert LLOracle.domain == "ll"


def test_ll_oracle_rejects_missing_checkpoint(tmp_path) -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    spec = LLModelSpec(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=4,
        d_ff=512, max_seq_len=256,
    )
    with pytest.raises(FileNotFoundError):
        LLOracle(checkpoint_path=tmp_path / "nonexistent.pt", model_spec=spec)


def test_ll_oracle_regime_grid_non_empty(tmp_path) -> None:
    """Régime grid LL : 5 complexity × 3 seq_len = 15 régimes."""
    from catalog.oracles import LLOracle, LLModelSpec
    fake_ckpt = tmp_path / "dummy.pt"
    fake_ckpt.write_bytes(b"placeholder")
    spec = LLModelSpec(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=4,
        d_ff=512, max_seq_len=256,
    )
    oracle = LLOracle(checkpoint_path=fake_ckpt, model_spec=spec)
    grid = oracle.regime_grid()
    assert len(grid) == 15
    assert all(isinstance(r, RegimeSpec) for r in grid)


def test_ll_oracle_extract_regime_raises_v1(tmp_path) -> None:
    """V1 squelette : extract_regime lève NotImplementedError tant que tokenizer/template absents."""
    from catalog.oracles import LLOracle, LLModelSpec
    fake_ckpt = tmp_path / "dummy.pt"
    fake_ckpt.write_bytes(b"placeholder")
    spec = LLModelSpec(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=4,
        d_ff=512, max_seq_len=256,
    )
    oracle = LLOracle(checkpoint_path=fake_ckpt, model_spec=spec)
    regime = RegimeSpec(omega=0, delta=64, entropy=0.0)
    with pytest.raises(NotImplementedError, match="tokenizer"):
        oracle.extract_regime(regime, n_examples=2)
