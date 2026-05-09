"""Tests Oracle Transformer dense + extraction d'attention FP64."""

from __future__ import annotations

import torch

from phase1_metrologie.oracle.extract import AttentionExtractor
from phase1_metrologie.oracle.train import TrainConfig, find_query_positions
from phase1_metrologie.oracle.transformer import (
    DenseAttention,
    OracleConfig,
    OracleTransformer,
)
from phase1_metrologie.ssg.structure_mnist import (
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
)


def _make_oracle(vocab: Vocab, **overrides) -> OracleTransformer:
    cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=overrides.get("d_model", 32),
        n_heads=overrides.get("n_heads", 4),
        n_layers=overrides.get("n_layers", 2),
        d_ff=overrides.get("d_ff", 64),
        max_seq_len=overrides.get("max_seq_len", 64),
        n_classes=10,
        pad_id=vocab.PAD,
    )
    return OracleTransformer(cfg)


def _make_batch(ds: StructureMNISTDataset, n: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [ds[i] for i in range(n)]
    tokens = torch.stack([s.tokens for s in samples])
    targets = torch.tensor([s.target for s in samples])
    qpos = find_query_positions(tokens, ds.vocab.QUERY)
    return tokens, targets, qpos


# ----------------------------------------------------------------
# Forward / shapes
# ----------------------------------------------------------------


def test_oracle_forward_shape() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=2, delta=4, entropy=0.5, n_examples=4, seed=0))
    tokens, _, qpos = _make_batch(ds, n=4)
    logits = oracle(tokens, qpos, capture_attn=False)
    assert logits.shape == (4, 10)


def test_oracle_no_attn_capture_by_default() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=2, delta=4, entropy=0.5, n_examples=2, seed=0))
    tokens, _, qpos = _make_batch(ds, n=2)
    _ = oracle(tokens, qpos, capture_attn=False)
    for block in oracle.blocks:
        assert block.attn.last_attn is None


def test_oracle_capture_attn_materializes_full_matrix() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=2, delta=4, entropy=0.5, n_examples=2, seed=0))
    tokens, _, qpos = _make_batch(ds, n=2)
    _ = oracle(tokens, qpos, capture_attn=True)
    for block in oracle.blocks:
        a = block.attn.last_attn
        assert a is not None
        # (B, H, N, N), softmax doit sommer à 1 (à l'epsilon près) sur le dernier axe
        # (en ignorant les positions PAD masquées qui contribuent 0)
        N = tokens.size(1)
        assert a.shape == (2, oracle.cfg.n_heads, N, N)
        row_sums = a.sum(dim=-1)
        # Les positions non-PAD doivent avoir une somme proche de 1
        assert (row_sums > 0.99).any()


# ----------------------------------------------------------------
# Masquage PAD
# ----------------------------------------------------------------


def test_pad_mask_blocks_attention_to_pad_positions() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    # Séquence avec tokens informatifs au début, PAD à la fin
    N = 16
    tokens = torch.full((1, N), v.PAD, dtype=torch.int64)
    tokens[0, 0] = v.BOS
    tokens[0, 1] = v.digit(3)
    tokens[0, 2] = v.QUERY
    qpos = torch.tensor([2])
    _ = oracle(tokens, qpos, capture_attn=True)
    a = oracle.blocks[0].attn.last_attn
    # Pour chaque (head, query position), l'attention vers les PAD doit être ≈ 0
    pad_positions = (tokens[0] == v.PAD).nonzero(as_tuple=True)[0]
    for pos in pad_positions:
        attn_to_pad = a[0, :, :, pos]   # (H, N)
        assert (attn_to_pad < 1e-3).all(), f"attention vers PAD@{pos} non nulle"


# ----------------------------------------------------------------
# DenseAttention — forward direct
# ----------------------------------------------------------------


def test_dense_attention_capture_softmax_normalized() -> None:
    attn = DenseAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 8, 16)
    _ = attn(x, attn_mask=None, capture_attn=True)
    a = attn.last_attn
    assert a is not None
    assert a.shape == (2, 2, 8, 8)
    # Lignes softmax-normalisées
    row_sums = a.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_dense_attention_no_capture_no_buffer() -> None:
    attn = DenseAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 8, 16)
    _ = attn(x, attn_mask=None, capture_attn=False)
    assert attn.last_attn is None


# ----------------------------------------------------------------
# Extraction FP64
# ----------------------------------------------------------------


def test_extractor_returns_fp64_attention() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=2, delta=4, entropy=0.5, n_examples=2, seed=0))
    tokens, targets, qpos = _make_batch(ds, n=2)
    omegas = torch.tensor([s.omega for s in [ds[0], ds[1]]])
    deltas = torch.tensor([s.delta for s in [ds[0], ds[1]]])
    entropies = torch.tensor([s.entropy for s in [ds[0], ds[1]]])
    dump = ext.extract(tokens, qpos, targets, omegas, deltas, entropies)
    assert dump.n_layers() == oracle.cfg.n_layers
    assert dump.n_heads() == oracle.cfg.n_heads
    for a in dump.attn:
        assert a.dtype == torch.float64
        assert a.shape == (2, oracle.cfg.n_heads, tokens.size(1), tokens.size(1))


def test_extractor_clears_layer_buffers_after_extract() -> None:
    v = Vocab()
    oracle = _make_oracle(v)
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=2, delta=4, entropy=0.5, n_examples=2, seed=0))
    tokens, targets, qpos = _make_batch(ds, n=2)
    omegas = torch.zeros(2, dtype=torch.int64)
    deltas = torch.zeros(2, dtype=torch.int64)
    entropies = torch.zeros(2)
    _ = ext.extract(tokens, qpos, targets, omegas, deltas, entropies)
    # Après extraction, les buffers de chaque DenseAttention doivent être libérés
    for block in oracle.blocks:
        assert block.attn.last_attn is None


def test_extractor_preserves_eval_mode_state() -> None:
    """L'extracteur ne doit pas changer durablement le mode train/eval du modèle."""
    v = Vocab()
    oracle = _make_oracle(v)
    oracle.train()
    ext = AttentionExtractor(oracle)
    ds = StructureMNISTDataset(StructureMNISTConfig(omega=1, delta=2, entropy=0.0, n_examples=1, seed=0))
    tokens, targets, qpos = _make_batch(ds, n=1)
    _ = ext.extract(tokens, qpos, targets, torch.zeros(1, dtype=torch.int64),
                     torch.zeros(1, dtype=torch.int64), torch.zeros(1))
    assert oracle.training is True


# ----------------------------------------------------------------
# Query position detection
# ----------------------------------------------------------------


def test_find_query_positions_returns_correct_index() -> None:
    v = Vocab()
    # Construit une séquence à la main avec QUERY en position 5
    N = 12
    seq = torch.full((2, N), v.PAD, dtype=torch.int64)
    seq[0, 5] = v.QUERY
    seq[1, 8] = v.QUERY
    qpos = find_query_positions(seq, v.QUERY)
    assert qpos.tolist() == [5, 8]


# ----------------------------------------------------------------
# Smoke training (1 époque sur petit dataset CPU)
# ----------------------------------------------------------------


def test_train_oracle_smoke_cpu() -> None:
    """Smoke test : entraînement 1 époque sur 16 exemples. Vérifie que la
    boucle Fabric tourne sans erreur en CPU.
    """
    from lightning.fabric import Fabric

    v = Vocab()
    train_ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.0, n_examples=16, seed=0)
    )
    val_ds = StructureMNISTDataset(
        StructureMNISTConfig(omega=1, delta=2, entropy=0.0, n_examples=8, seed=1)
    )
    model_cfg = OracleConfig(
        vocab_size=v.size, d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_seq_len=16, pad_id=v.PAD,
    )
    train_cfg = TrainConfig(
        batch_size=4, lr=1e-3, max_epochs=1, patience=1, precision="32-true",
    )
    fabric = Fabric(precision=train_cfg.precision)

    from phase1_metrologie.oracle.train import train_oracle

    model, metrics = train_oracle(
        model_cfg=model_cfg, train_ds=train_ds, val_ds=val_ds,
        train_cfg=train_cfg, query_id=v.QUERY, fabric=fabric,
    )
    assert "val_loss" in metrics
    assert metrics["val_loss"] >= 0
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
