"""Tests pour la nouvelle API streaming de extract.py (refactor 2026-05-11).

Couvre :
- extract_per_layer (générateur, libération progressive)
- extract_streamed (callback)
- extract_windowed_per_layer (fenêtres K×K, phase 2)
- ExtractorConfig (validation numérique, max_layers, stream_to_disk)
- Backward compat avec extract() préservé
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from phase1_metrologie.oracle.extract import (
    AttentionDump,
    AttentionExtractor,
    ExtractorConfig,
    LayerDump,
)
from phase1_metrologie.oracle.train import find_query_positions
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import (
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
)


def _make_oracle(vocab: Vocab, n_layers: int = 3, n_heads: int = 4) -> OracleTransformer:
    cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=32,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=64,
        max_seq_len=64,
        n_classes=10,
        pad_id=vocab.PAD,
    )
    return OracleTransformer(cfg)


def _make_batch(ds: StructureMNISTDataset, n: int = 2):
    samples = [ds[i] for i in range(n)]
    tokens = torch.stack([s.tokens for s in samples])
    targets = torch.tensor([s.target for s in samples])
    qpos = find_query_positions(tokens, ds.vocab.QUERY)
    omegas = torch.tensor([s.omega for s in samples])
    deltas = torch.tensor([s.delta for s in samples])
    entropies = torch.tensor([s.entropy for s in samples])
    return tokens, qpos, targets, omegas, deltas, entropies


# -----------------------------------------------------------------------------
# extract_per_layer
# -----------------------------------------------------------------------------


def test_extract_per_layer_yields_each_layer() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    layers_seen = []
    for layer_dump in ext.extract_per_layer(
        tokens, qpos, targets, omegas, deltas, entropies
    ):
        assert isinstance(layer_dump, LayerDump)
        layers_seen.append(layer_dump.layer)
        assert layer_dump.attn.dtype == torch.float64
        assert layer_dump.attn.shape == (2, 4, tokens.size(1), tokens.size(1))
        assert not layer_dump.is_windowed

    assert layers_seen == [0, 1, 2]


def test_extract_per_layer_releases_buffers_progressively() -> None:
    """Le buffer .last_attn de chaque block doit être libéré après son yield."""
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    iterator = ext.extract_per_layer(
        tokens, qpos, targets, omegas, deltas, entropies
    )
    # Consomme la première couche
    _ = next(iterator)
    # Buffer de la couche 0 doit être libéré
    assert oracle.blocks[0].attn.last_attn is None
    # Consomme tout
    for _ in iterator:
        pass
    # Tous les buffers libérés à la fin
    for block in oracle.blocks:
        assert block.attn.last_attn is None


def test_extract_per_layer_respects_max_layers() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=4)
    ext = AttentionExtractor(oracle, ExtractorConfig(max_layers=2))
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    layers = [
        d.layer
        for d in ext.extract_per_layer(tokens, qpos, targets, omegas, deltas, entropies)
    ]
    assert layers == [0, 1]


# -----------------------------------------------------------------------------
# extract_streamed
# -----------------------------------------------------------------------------


def test_extract_streamed_invokes_callback_per_layer() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    received: list[int] = []

    def cb(layer_dump: LayerDump) -> None:
        received.append(layer_dump.layer)
        assert layer_dump.attn.dtype == torch.float64

    ext.extract_streamed(tokens, qpos, targets, omegas, deltas, entropies, cb)
    assert received == [0, 1, 2]


# -----------------------------------------------------------------------------
# extract_windowed_per_layer
# -----------------------------------------------------------------------------


def test_extract_windowed_per_layer_yields_diagonal_windows() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=2)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=1, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds, n=1)
    N = tokens.size(1)
    K = max(2, N // 3)

    windows: list[LayerDump] = list(
        ext.extract_windowed_per_layer(
            tokens, qpos, targets, omegas, deltas, entropies, K=K
        )
    )
    assert len(windows) > 0
    for w in windows:
        assert w.is_windowed
        assert w.window_size == K
        assert w.attn.shape == (1, oracle.cfg.n_heads, K, K)
        assert w.attn.dtype == torch.float64
        assert w.window_offset is not None and 0 <= w.window_offset <= N - K


def test_extract_windowed_per_layer_K_too_large_raises() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=2)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=1, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds, n=1)
    N = tokens.size(1)

    with pytest.raises(ValueError, match="K=.+> seq_len"):
        list(
            ext.extract_windowed_per_layer(
                tokens, qpos, targets, omegas, deltas, entropies, K=N + 10
            )
        )


def test_extract_windowed_invalid_K_raises() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=1)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=1, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds, n=1)

    with pytest.raises(ValueError, match="K doit être > 0"):
        list(
            ext.extract_windowed_per_layer(
                tokens, qpos, targets, omegas, deltas, entropies, K=0
            )
        )


# -----------------------------------------------------------------------------
# ExtractorConfig
# -----------------------------------------------------------------------------


def test_extractor_config_fp64_false_keeps_native_dtype() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=2)
    ext = AttentionExtractor(oracle, ExtractorConfig(fp64=False))
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=1, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds, n=1)
    # Avec FP32 model par défaut, on attend du float32
    for layer_dump in ext.extract_per_layer(
        tokens, qpos, targets, omegas, deltas, entropies
    ):
        assert layer_dump.attn.dtype == torch.float32


def test_extractor_config_validate_numerics_passes_for_softmax() -> None:
    """Une attention softmax bien-formée doit passer la validation."""
    v = Vocab()
    oracle = _make_oracle(v, n_layers=2)
    ext = AttentionExtractor(oracle, ExtractorConfig(validate_numerics=True))
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)
    # Pas d'exception attendue
    dumps = list(
        ext.extract_per_layer(tokens, qpos, targets, omegas, deltas, entropies)
    )
    assert len(dumps) == 2


def test_extractor_stream_to_disk_writes_one_file_per_layer(tmp_path: Path) -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(
        oracle, ExtractorConfig(stream_to_disk=tmp_path / "dumps")
    )
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=1, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds, n=1)
    _ = list(ext.extract_per_layer(tokens, qpos, targets, omegas, deltas, entropies))

    files = sorted((tmp_path / "dumps").glob("layer_*.pt"))
    assert len(files) == 3
    for i, f in enumerate(files):
        loaded = torch.load(f, weights_only=False)
        assert loaded["layer"] == i
        assert loaded["attn"].dtype == torch.float64


# -----------------------------------------------------------------------------
# Backward compat
# -----------------------------------------------------------------------------


def test_extract_backward_compat_returns_attention_dump() -> None:
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    dump = ext.extract(tokens, qpos, targets, omegas, deltas, entropies)
    assert isinstance(dump, AttentionDump)
    assert dump.n_layers() == 3
    assert dump.n_heads() == 4
    assert dump.seq_len() == tokens.size(1)
    for a in dump.attn:
        assert a.dtype == torch.float64


def test_extract_per_layer_equivalent_to_extract() -> None:
    """extract_per_layer doit produire les mêmes tenseurs que extract."""
    v = Vocab()
    oracle = _make_oracle(v, n_layers=3)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.5, n_examples=2, seed=0)
    )
    tokens, qpos, targets, omegas, deltas, entropies = _make_batch(ds)

    dump_full = ext.extract(tokens, qpos, targets, omegas, deltas, entropies)
    dumps_stream = list(
        ext.extract_per_layer(tokens, qpos, targets, omegas, deltas, entropies)
    )

    assert len(dumps_stream) == dump_full.n_layers()
    for ell, ld in enumerate(dumps_stream):
        assert torch.allclose(ld.attn, dump_full.attn[ell])
