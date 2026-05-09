"""Tests SSG Structure-MNIST. Reproductibilité, balayages, tri-partition."""

from __future__ import annotations

import numpy as np
import pytest

from phase1_metrologie.ssg.structure_mnist import (
    SplitConfig,
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
    apply_program,
    split_indices,
    sweep_monovariate,
)


def test_vocab_disjoint() -> None:
    v = Vocab(n_ops=4, n_noise=8)
    digits = {v.digit(d) for d in range(10)}
    ops = {v.op(o) for o in range(4)}
    noise = {v.noise(i) for i in range(8)}
    specials = {v.PAD, v.BOS, v.QUERY}
    all_ids = digits | ops | noise | specials
    assert len(all_ids) == 10 + 4 + 8 + 3
    assert v.size == 10 + 4 + 1 + 8 + 2  # digits + ops + PAD + noise + BOS + QUERY


def test_apply_program_simple() -> None:
    op_table = [(1, 1)]  # x = (x + d) mod 10
    assert apply_program(initial=3, ops=[0], digits=[4], op_table=op_table) == 7
    assert apply_program(initial=3, ops=[0, 0], digits=[4, 5], op_table=op_table) == 2


def test_dataset_reproducible() -> None:
    cfg = StructureMNISTConfig(omega=4, delta=8, entropy=0.5, n_examples=16, seed=42)
    ds = StructureMNISTDataset(cfg)
    s1 = ds[3]
    s2 = ds[3]
    assert (s1.tokens == s2.tokens).all()
    assert s1.target == s2.target


def test_dataset_different_seeds_differ() -> None:
    cfg_a = StructureMNISTConfig(omega=4, delta=8, entropy=0.5, n_examples=16, seed=1)
    cfg_b = StructureMNISTConfig(omega=4, delta=8, entropy=0.5, n_examples=16, seed=2)
    a = StructureMNISTDataset(cfg_a)[0]
    b = StructureMNISTDataset(cfg_b)[0]
    # Très improbable d'être identiques pour deux seeds différents
    assert not (a.tokens == b.tokens).all() or a.target != b.target


def test_dataset_seq_length_consistent() -> None:
    cfg = StructureMNISTConfig(omega=3, delta=5, entropy=0.0, n_examples=8, seed=0)
    ds = StructureMNISTDataset(cfg)
    expected = cfg.expected_seq_len()
    for i in range(len(ds)):
        assert ds[i].tokens.numel() == expected


def test_dataset_omega_zero() -> None:
    cfg = StructureMNISTConfig(omega=0, delta=2, entropy=0.0, n_examples=4, seed=0)
    ds = StructureMNISTDataset(cfg)
    s = ds[0]
    # ω=0 → cible = digit initial
    assert s.target == int(s.tokens[1].item())


def test_entropy_changes_distractor_distribution() -> None:
    cfg_zero = StructureMNISTConfig(omega=2, delta=20, entropy=0.0, n_examples=64, seed=0)
    cfg_one = StructureMNISTConfig(omega=2, delta=20, entropy=1.0, n_examples=64, seed=0)
    v = Vocab(n_ops=cfg_zero.n_ops, n_noise=cfg_zero.n_noise)
    pad_count_zero = sum(int((StructureMNISTDataset(cfg_zero)[i].tokens == v.PAD).sum()) for i in range(64))
    pad_count_one = sum(int((StructureMNISTDataset(cfg_one)[i].tokens == v.PAD).sum()) for i in range(64))
    # ℋ=0 : que des PAD entre tokens info. ℋ=1 : aucun PAD distracteur.
    assert pad_count_zero > pad_count_one * 2


def test_split_indices_disjoint() -> None:
    n = 1000
    parts = split_indices(n, SplitConfig(seed=7))
    s_train = set(parts["train_oracle"].tolist())
    s_audit = set(parts["audit_svd"].tolist())
    s_init = set(parts["init_phase3"].tolist())
    assert s_train.isdisjoint(s_audit)
    assert s_train.isdisjoint(s_init)
    assert s_audit.isdisjoint(s_init)
    assert len(s_train | s_audit | s_init) == n


def test_split_proportions() -> None:
    n = 10_000
    parts = split_indices(n, SplitConfig(train_oracle=0.7, audit_svd=0.2, init_phase3=0.1, seed=0))
    assert abs(len(parts["train_oracle"]) - 7000) < 5
    assert abs(len(parts["audit_svd"]) - 2000) < 5
    assert abs(len(parts["init_phase3"]) - 1000) < 5


def test_sweep_monovariate() -> None:
    base = StructureMNISTConfig(omega=2, delta=8, entropy=0.0, n_examples=16, seed=0)
    omegas = list(sweep_monovariate(axis="omega", values=[1, 2, 4, 8], base=base))
    assert [c.omega for c in omegas] == [1, 2, 4, 8]
    assert all(c.delta == 8 for c in omegas)
    assert all(c.entropy == 0.0 for c in omegas)


def test_invalid_split_raises() -> None:
    with pytest.raises(AssertionError):
        SplitConfig(train_oracle=0.5, audit_svd=0.4, init_phase3=0.2)
