"""Tests GPT2Oracle — mock HFLanguageBackend pour éviter download réseau en CI."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock

from catalog.oracles.base import RegimeSpec
from catalog.oracles.language import nested_parentheses_template


def _make_mock_backend(n_layers: int = 12, n_heads: int = 12) -> MagicMock:
    backend = MagicMock()
    backend.n_layers = n_layers
    backend.n_heads = n_heads
    backend.tokenize.return_value = torch.zeros(4, 64, dtype=torch.long)
    backend.forward_with_attn.return_value = [
        torch.softmax(torch.randn(4, n_heads, 64, 64), dim=-1)
        for _ in range(n_layers)
    ]
    return backend


@pytest.fixture()
def oracle():
    """GPT2Oracle avec backend mocké (pas de téléchargement HF)."""
    from catalog.oracles.gpt2 import GPT2Oracle
    obj = GPT2Oracle.__new__(GPT2Oracle)
    obj.variant = "small"
    obj.oracle_id = "gpt2_small"
    obj.device = "cpu"
    obj.n_examples_max = 32
    obj.prompt_fn = nested_parentheses_template
    obj._backend = _make_mock_backend(n_layers=12, n_heads=12)
    obj.n_layers = 12
    obj.n_heads = 12
    return obj


# ---------------------------------------------------------------------------
# Métadonnées de classe
# ---------------------------------------------------------------------------


def test_domain() -> None:
    from catalog.oracles.gpt2 import GPT2Oracle
    assert GPT2Oracle.domain == "pretrained_ll"


def test_invalid_variant_raises() -> None:
    from catalog.oracles.gpt2 import GPT2Oracle
    from unittest.mock import patch
    with patch("catalog.oracles.gpt2.HFLanguageBackend"):
        with pytest.raises(ValueError, match="variant GPT-2 inconnu"):
            GPT2Oracle(variant="gpt3")


def test_valid_variants_accepted() -> None:
    from catalog.oracles.gpt2 import GPT2Oracle, GPT2_VARIANTS
    from unittest.mock import patch
    for variant in GPT2_VARIANTS:
        with patch("catalog.oracles.gpt2.HFLanguageBackend") as MockHF:
            MockHF.return_value = _make_mock_backend()
            oracle = GPT2Oracle(variant=variant)
            assert oracle.variant == variant
            assert oracle.oracle_id == f"gpt2_{variant}"


# ---------------------------------------------------------------------------
# regime_grid
# ---------------------------------------------------------------------------


def test_regime_grid_size(oracle) -> None:
    grid = oracle.regime_grid()
    assert len(grid) == 8  # 2 depths × 4 seq_lens


def test_regime_grid_types(oracle) -> None:
    grid = oracle.regime_grid()
    assert all(isinstance(r, RegimeSpec) for r in grid)


def test_regime_grid_seq_lens(oracle) -> None:
    grid = oracle.regime_grid()
    deltas = {r.delta for r in grid}
    assert deltas == {64, 128, 256, 512}


def test_regime_grid_depths(oracle) -> None:
    grid = oracle.regime_grid()
    omegas = {r.omega for r in grid}
    assert omegas == {0, 2}


# ---------------------------------------------------------------------------
# extract_regime
# ---------------------------------------------------------------------------


def test_extract_returns_valid_dump(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=4)
    dump.validate()
    assert dump.n_layers == 12
    assert dump.n_heads == 12
    assert dump.n_examples == 4
    assert dump.seq_len == 64


def test_extract_attn_fp64(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=2)
    for layer_attn in dump.attn:
        assert layer_attn.dtype == torch.float64


def test_extract_seq_len_capped_at_1024(oracle) -> None:
    oracle._backend.tokenize.return_value = torch.zeros(2, 1024, dtype=torch.long)
    oracle._backend.forward_with_attn.return_value = [
        torch.softmax(torch.randn(2, 12, 1024, 1024), dim=-1)
        for _ in range(12)
    ]
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=9999), n_examples=2)
    assert dump.metadata["seq_len"] == 1024


def test_extract_n_examples_capped(oracle) -> None:
    oracle.n_examples_max = 3
    oracle._backend.tokenize.return_value = torch.zeros(3, 64, dtype=torch.long)
    oracle._backend.forward_with_attn.return_value = [
        torch.softmax(torch.randn(3, 12, 64, 64), dim=-1)
        for _ in range(12)
    ]
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=100)
    assert dump.n_examples == 3


def test_extract_metadata_causal(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=2)
    assert dump.metadata["causal"] is True


def test_extract_metadata_oracle_id(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=2)
    assert dump.metadata["oracle_id"] == "gpt2_small"


def test_extract_tokens_shape(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=4)
    assert dump.tokens is not None
    assert dump.tokens.shape == (4, 64)


def test_extract_omegas_are_zero(oracle) -> None:
    """GPT-2 n'a pas de contrôle structurel : omegas = 0."""
    dump = oracle.extract_regime(RegimeSpec(omega=2, delta=64), n_examples=3)
    assert dump.omegas.eq(0).all()


def test_extract_deltas_match_seq_len(oracle) -> None:
    dump = oracle.extract_regime(RegimeSpec(omega=0, delta=64), n_examples=3)
    assert dump.deltas.eq(64).all()
