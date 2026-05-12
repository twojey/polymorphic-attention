"""Tests LLOracle (squelettes V1 + pipeline V2)."""

from __future__ import annotations

import pytest
import torch

from catalog.oracles.base import RegimeSpec


def test_ll_oracle_imports() -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    assert LLOracle.domain == "ll"


def test_ll_oracle_rejects_missing_checkpoint(tmp_path) -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    spec = LLModelSpec(vocab_size=256, d_model=32, n_heads=2, n_layers=2,
                       d_ff=64, max_seq_len=64)
    with pytest.raises(FileNotFoundError):
        LLOracle(checkpoint_path=tmp_path / "nonexistent.pt", model_spec=spec)


def test_ll_oracle_random_init_works() -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    spec = LLModelSpec(vocab_size=256, d_model=32, n_heads=2, n_layers=2,
                       d_ff=64, max_seq_len=64)
    oracle = LLOracle(model_spec=spec)
    grid = oracle.regime_grid()
    assert len(grid) == 15  # 5 depths × 3 seq_lens


def test_ll_oracle_regime_grid_non_empty(tmp_path) -> None:
    from catalog.oracles import LLOracle, LLModelSpec
    fake_ckpt = tmp_path / "dummy.pt"
    fake_ckpt.write_bytes(b"placeholder")
    spec = LLModelSpec(vocab_size=256, d_model=32, n_heads=2, n_layers=2,
                       d_ff=64, max_seq_len=64)
    oracle = LLOracle(checkpoint_path=fake_ckpt, model_spec=spec)
    grid = oracle.regime_grid()
    assert len(grid) == 15
    assert all(isinstance(r, RegimeSpec) for r in grid)


def test_ll_oracle_extract_pipeline_random_init(tmp_path) -> None:
    """Pipeline LL V2 : extract avec random init produit AttentionDump valide."""
    from catalog.oracles import LLOracle, LLModelSpec
    spec = LLModelSpec(vocab_size=256, d_model=32, n_heads=2, n_layers=2,
                       d_ff=64, max_seq_len=64)
    oracle = LLOracle(model_spec=spec)
    regime = RegimeSpec(omega=2, delta=32, entropy=0.0)
    dump = oracle.extract_regime(regime, n_examples=3)
    dump.validate()
    assert dump.n_layers == 2
    assert dump.attn[0].dtype == torch.float64
    assert dump.tokens.shape == (3, 32)


def test_ll_oracle_nested_template() -> None:
    """Le prompt template produit du texte de longueur seq_len."""
    from catalog.oracles.language import nested_parentheses_template
    for seed in range(3):
        for depth in (0, 1, 2, 4):
            txt = nested_parentheses_template(depth=depth, seq_len=64, seed=seed)
            assert len(txt) == 64
