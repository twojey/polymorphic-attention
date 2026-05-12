"""
run_extract.py — driver V2 d'extraction des matrices A FP64 (phase 1, post-training).

Recharge un Oracle entraîné (checkpoint phase 1) puis extrait les matrices
d'attention sur **tout le set audit_svd** (vs un seul batch en V1). Produit
des dumps `.pt` compatibles phase 2 : `{attn, omegas, deltas, entropies}`.

Pipeline :
1. Hydra config phase 1 (oracle_smnist.yaml) + override `oracle_checkpoint=PATH`.
2. Reconstruit le sweep dataset + tri-partition → `audit_svd` indices.
3. Recharge Oracle depuis le checkpoint (réutilise `_load_oracle` de phase 1.5).
4. Groupe les exemples par seq_len exact (déterminé par ω, Δ).
5. Pour chaque bucket : extract_per_layer → empile (B, H, N, N) FP64 par couche.
6. Sauvegarde 1 dump par bucket : `audit_dump_seq{N}.pt`.
7. Log MLflow : dumps comme artifacts + métriques Hankel/entropie par régime.

Robustesse (DOC/feedback_script_robustness) :
- warnings/erreurs sur stderr avec traceback (jamais silencieux) ;
- MLflow log_artifact non-bloquant : si MLflow tombe, le dump reste sur
  disque et le run continue (la source de vérité = le fichier `.pt`) ;
- resume capability : `extraction.resume_skip_existing=true` saute les
  buckets déjà extraits (utile post-crash ou si MLflow upload a échoué) ;
- snapshot RAM/VRAM avant chaque bucket (psutil + torch.cuda.mem_get_info).

Usage :
    PYTHONPATH=CODE uv run python -m phase1_metrologie.run_extract \
        --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist \
        oracle_checkpoint=/path/to/oracle.ckpt

Pourquoi un driver séparé du V1 :
- Le V1 (run.py) couple training + extraction d'un seul batch.
- Le V2 sert UNIQUEMENT à régénérer les dumps phase-2-ready à partir d'un
  Oracle déjà entraîné. Évite de re-entraîner pour 4-6 h juste pour ré-extraire.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from phase1_metrologie.metrics.hankel import hankel_rank_numerical
from phase1_metrologie.metrics.spectral import spectral_entropy
from phase1_metrologie.oracle.extract import AttentionExtractor, ExtractorConfig
from phase1_metrologie.oracle.train import collate, find_query_positions
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.ssg.structure_mnist import (
    SplitConfig,
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
    split_indices,
    sweep_monovariate,
)
from shared.mlflow_helpers import log_yaml_config, start_run
from shared.runner import finalize_manifest, make_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Helpers robustesse
# -----------------------------------------------------------------------------


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_mlflow(operation: str, fn, *args, **kwargs):
    """Exécute une opération MLflow ; warn-but-continue si elle échoue.

    Les artefacts critiques sont écrits sur disque AVANT toute opération
    MLflow → une coupure réseau ne doit pas annuler l'extraction. Pour les
    opérations bloquantes (start_run, set_experiment), ne pas utiliser ce
    wrapper.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _stderr(
            f"[mlflow] {operation} a échoué : {type(exc).__name__}: {exc}\n"
            f"         (artefact disque préservé, run continue)"
        )
        traceback.print_exc(file=sys.stderr, limit=3)
        return None


def _print_resource_snapshot(prefix: str) -> None:
    """RAM disponible + VRAM, pour traçabilité avant pic mémoire."""
    msg = prefix
    try:
        import psutil
        mem = psutil.virtual_memory()
        msg += (
            f" RAM avail={mem.available/1e9:.1f} GB used={mem.percent:.0f}%"
        )
    except ImportError:
        msg += " psutil indisponible"
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        msg += (
            f" | VRAM free={free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB "
            f"(alloc={torch.cuda.memory_allocated()/1e9:.2f} GB)"
        )
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Helpers dataset / modèle
# -----------------------------------------------------------------------------


def _build_sweep_dataset(
    cfg: DictConfig,
) -> ConcatDataset:
    """Reconstruit le ConcatDataset identique à `run.py` (V1) pour cohérence indices."""
    base = StructureMNISTConfig(
        omega=cfg.ssg.axes.omega.reference,
        delta=cfg.ssg.axes.delta.reference,
        entropy=cfg.ssg.axes.entropy.reference,
        n_examples=cfg.ssg.n_examples_per_regime,
        n_ops=cfg.ssg.n_ops,
        n_noise=cfg.ssg.n_noise,
        seed=cfg.ssg.seed_train,
    )
    all_datasets: list[StructureMNISTDataset] = []
    for axis in ("omega", "delta", "entropy"):
        sweep_values = list(cfg.ssg.axes[axis].sweep)
        for c in sweep_monovariate(axis=axis, values=sweep_values, base=base):
            all_datasets.append(StructureMNISTDataset(c))
    return ConcatDataset(all_datasets)


def _load_oracle(
    checkpoint_path: str,
    vocab: Vocab,
    model_cfg: OracleConfig,
    device: str,
) -> OracleTransformer:
    """Recharge l'Oracle depuis le checkpoint Fabric (strip _forward_module/_orig_mod prefixes)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    model = OracleTransformer(model_cfg)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.removeprefix("_forward_module.").removeprefix("_orig_mod.")
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        _stderr(
            f"[load_oracle] state_dict : missing={len(missing)} "
            f"unexpected={len(unexpected)} (peut être bénin selon le checkpoint)"
        )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _expected_seq_len(omega: int, delta: int) -> int:
    """Formule SSG (cf. structure_mnist.py + phase1b §_calibrate_kl_baseline).

    seq_len = (2Δ+2)·ω + Δ + 3
    Indépendant de ℋ (ℋ change la distribution NOISE, pas la longueur).
    """
    return (2 * delta + 2) * omega + delta + 3


def _adaptive_batch_size(seq_len: int, *, L: int, H: int, target_gb: float = 15.0,
                        cap: int = 8) -> int:
    """B tel que B·L·H·N²·8 ≤ target_gb (peak FP64 attention).

    cap conservateur : les activations FFN ne sont pas comptées dans cette
    estimation (cf. crash Run 3 du 2026-05-11). Mieux vaut sous-estimer.
    """
    if seq_len <= 0:
        return 1
    n_max = max(1, int(target_gb * 1e9 / (L * H * seq_len * seq_len * 8)))
    return min(cap, n_max)


def _group_by_seq_len(
    indices: list[int],
    dataset: ConcatDataset | torch.utils.data.Dataset,
) -> dict[int, list[int]]:
    """Pré-scan : groupe les indices par seq_len exact.

    Itère sur chaque item pour récupérer la longueur de tokens. Coût O(|indices|)
    sans matérialiser de tensors lourds (pas de forward). Le dataset retourne un
    `StructureMNISTSample` (dataclass) — on lit `item.tokens.shape[0]`.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        item = dataset[idx]
        n = item.tokens.shape[0]
        groups[n].append(idx)
    return dict(groups)


def _apply_bucket_caps(
    seq_len_groups: dict[int, list[int]],
    *,
    L: int,
    H: int,
    skip_seq_len_above: int | None,
    max_bucket_size_gb: float | None,
    max_examples_per_bucket: int | None,
    seed: int,
) -> dict[int, list[int]]:
    """Applique les bornes disque/compute sur les buckets seq_len.

    Trois bornes mutuellement compatibles, dans l'ordre :
    1. `skip_seq_len_above` : drop pur des buckets seq_len > N. Évite les
       extractions impossibles (e.g., seq_len=5127 avec FP64 attention =
       ~8 TB pour 819 ex).
    2. `max_bucket_size_gb` : cap auto par bucket en fonction de la taille
       FP64 estimée. cap_auto = floor(GB·1e9 / (L·H·N²·8)).
    3. `max_examples_per_bucket` : cap uniforme dur (sécurité).

    Le sous-échantillonnage est déterministe (seeded) — pas d'ordre par-index
    aveugle qui privilégierait certains régimes. Sur chaque bucket conservé,
    on tire `cap` indices avec rng.choice(replace=False).

    Reporte explicitement à stderr ce qui a été droppé / capped (pas de
    silence).
    """
    if skip_seq_len_above is None and max_bucket_size_gb is None and max_examples_per_bucket is None:
        return seq_len_groups

    import numpy as _np
    rng = _np.random.default_rng(seed)

    def _cap_for(n_examples_avail: int, seq_len: int) -> int:
        cap = n_examples_avail
        if max_bucket_size_gb is not None:
            cap_size = max(1, int(
                max_bucket_size_gb * 1e9 / (L * H * seq_len * seq_len * 8)
            ))
            cap = min(cap, cap_size)
        if max_examples_per_bucket is not None:
            cap = min(cap, max_examples_per_bucket)
        return cap

    out: dict[int, list[int]] = {}
    for seq_len in sorted(seq_len_groups.keys()):
        indices = seq_len_groups[seq_len]
        if skip_seq_len_above is not None and seq_len > skip_seq_len_above:
            _stderr(
                f"[bucket_cap] DROP seq_len={seq_len} ({len(indices)} ex) — "
                f"au-dessus de skip_seq_len_above={skip_seq_len_above}"
            )
            continue
        cap = _cap_for(len(indices), seq_len)
        if cap < len(indices):
            chosen = rng.choice(len(indices), size=cap, replace=False)
            sub = sorted(indices[i] for i in chosen)
            size_est_gb = (cap * L * H * seq_len * seq_len * 8) / 1e9
            _stderr(
                f"[bucket_cap] seq_len={seq_len} : {len(indices)} → {cap} ex "
                f"(estim. {size_est_gb:.2f} GB FP64)"
            )
            out[seq_len] = sub
        else:
            out[seq_len] = list(indices)
    return out


def _validate_existing_dump(
    dump_path: Path,
    *,
    expected_seq_len: int,
    expected_L: int,
) -> bool:
    """Vérifie qu'un dump existant a la bonne structure pour être réutilisé.

    Retourne True si le dump peut servir tel quel (skip extraction), False
    sinon (re-extraction nécessaire). Le contenu est validé en lecture
    légère (clés présentes, seq_len, L) — pas de vérification numérique.
    """
    try:
        d = torch.load(str(dump_path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        _stderr(
            f"[resume] Dump {dump_path.name} illisible ({exc}) → "
            f"re-extraction."
        )
        return False
    needed = {"attn", "omegas", "deltas", "entropies"}
    missing = needed - set(d.keys())
    if missing:
        _stderr(f"[resume] Dump {dump_path.name} clés manquantes {missing} → re-extraction.")
        return False
    if len(d["attn"]) != expected_L:
        _stderr(
            f"[resume] Dump {dump_path.name} a {len(d['attn'])} couches, "
            f"attendu {expected_L} → re-extraction."
        )
        return False
    actual_seq_len = d["attn"][0].size(2)
    if actual_seq_len != expected_seq_len:
        _stderr(
            f"[resume] Dump {dump_path.name} seq_len={actual_seq_len}, "
            f"attendu {expected_seq_len} → re-extraction."
        )
        return False
    return True


# -----------------------------------------------------------------------------
# Extraction par bucket
# -----------------------------------------------------------------------------


def _extract_bucket(
    extractor: AttentionExtractor,
    subset: Subset,
    *,
    pad_id: int,
    query_id: int,
    batch_size: int,
    n_layers: int,
) -> dict[str, object]:
    """Extrait les matrices A pour un bucket (seq_len uniforme).

    Retourne le dict format phase 2 :
        {"attn": list[L]((B, H, N, N) FP64),
         "omegas": (B,), "deltas": (B,), "entropies": (B,),
         "tokens": (B, N)}

    Streaming via extract_per_layer : pic FP64 ≈ 1 couche × batch_size.
    """
    from functools import partial as _partial
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=_partial(collate, pad_id=pad_id),
        num_workers=0, pin_memory=False,
    )

    per_layer_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    omegas_chunks: list[torch.Tensor] = []
    deltas_chunks: list[torch.Tensor] = []
    entropies_chunks: list[torch.Tensor] = []
    tokens_chunks: list[torch.Tensor] = []

    n_batches = (len(subset) + batch_size - 1) // batch_size
    for b_idx, batch in enumerate(loader):
        tokens = batch["tokens"]
        qpos = find_query_positions(tokens, query_id)
        print(
            f"    [bucket] batch {b_idx+1}/{n_batches} shape={tuple(tokens.shape)} "
            f"; extract_per_layer streaming…",
            flush=True,
        )
        for layer_dump in extractor.extract_per_layer(
            tokens, qpos, batch["targets"], batch["omegas"], batch["deltas"], batch["entropies"],
        ):
            per_layer_chunks[layer_dump.layer].append(layer_dump.attn.cpu())
        omegas_chunks.append(batch["omegas"].cpu())
        deltas_chunks.append(batch["deltas"].cpu())
        entropies_chunks.append(batch["entropies"].cpu())
        tokens_chunks.append(tokens.cpu())

    attn_stacked = [torch.cat(chunks, dim=0) for chunks in per_layer_chunks]
    return {
        "attn": attn_stacked,
        "omegas": torch.cat(omegas_chunks, dim=0),
        "deltas": torch.cat(deltas_chunks, dim=0),
        "entropies": torch.cat(entropies_chunks, dim=0),
        "tokens": torch.cat(tokens_chunks, dim=0),
    }


def _log_bucket_metrics(
    dump: dict[str, object],
    *,
    seq_len: int,
    hankel_tau: float,
    max_examples_metrics: int = 32,
) -> None:
    """Agrégats Hankel rank + entropie spectrale par couche (tous régimes confondus).

    Volontairement minimal : on loggue uniquement le mean global par couche,
    pour monitoring temps réel de l'extraction. Le breakdown formel par
    régime (ω, Δ, ℋ) et le calcul de `min_portion` se font en post-processing
    à partir des dumps `.pt` (qui embarquent omegas/deltas/entropies). Cela
    évite de polluer MLflow avec O(L × n_régimes) métriques par bucket et
    centralise l'analyse régime dans un script séparé.

    Sous-échantillonnage : `hankel_rank_of_attention` a une boucle Python
    double (B × N) appelant SVD pour chaque ligne. Pour le bucket seq=87
    avec ~5000 ex, ça représente ~3.6M SVDs → minutes par couche. Comme
    on ne loggue qu'une moyenne globale pour monitoring, on subsample à
    `max_examples_metrics` (défaut 32) exemples — la moyenne reste
    représentative avec un IC large mais suffisant pour suivre la
    progression. Le calcul exact reste possible en post-processing sur
    le dump complet.

    Tout échec sur une couche est reporté à stderr avec traceback (jamais
    silencieux) mais ne stoppe pas l'extraction — les dumps complets sont
    déjà sur disque.
    """
    attn: list[torch.Tensor] = dump["attn"]  # type: ignore[assignment]

    for ell, A in enumerate(attn):  # A : (B, H, N, N)
        # Sous-échantillonnage déterministe (seed = ell + seq_len pour stabilité)
        B = A.size(0)
        if B > max_examples_metrics:
            g = torch.Generator().manual_seed(ell * 10007 + seq_len)
            idx = torch.randperm(B, generator=g)[:max_examples_metrics]
            A_sub = A[idx]
        else:
            A_sub = A
        try:
            hk_mean = float(hankel_rank_numerical(A_sub, tau=hankel_tau).item())
            se_mean = float(spectral_entropy(A_sub).mean().item())
        except Exception as exc:  # noqa: BLE001
            _stderr(
                f"[bucket_metrics] couche {ell} seq={seq_len} : "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc(file=sys.stderr, limit=3)
            continue

        _safe_mlflow(
            f"log_metric hankel_rank seq{seq_len} layer{ell}",
            mlflow.log_metric,
            f"hankel_rank_seq{seq_len}_layer{ell}_mean", hk_mean,
        )
        _safe_mlflow(
            f"log_metric spectral_entropy seq{seq_len} layer{ell}",
            mlflow.log_metric,
            f"spectral_entropy_seq{seq_len}_layer{ell}_mean", se_mean,
        )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../OPS/configs/phase1",
            config_name="oracle_smnist")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _print_resource_snapshot("[startup]")

    oracle_checkpoint = OmegaConf.select(cfg, "oracle_checkpoint")
    if oracle_checkpoint is None:
        raise SystemExit(
            "Override requis : `oracle_checkpoint=path/to/oracle.ckpt` "
            "(produit par phase1_metrologie.run V1)"
        )
    if not Path(oracle_checkpoint).is_file():
        raise SystemExit(
            f"Checkpoint introuvable : {oracle_checkpoint}. "
            f"Vérifier le path passé en override."
        )

    # output dir local pour les dumps (sera log comme MLflow artifact)
    output_dir = Path(OmegaConf.select(cfg, "extraction.output_dir")
                      or "/tmp/phase1_extract_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dumps locaux → {output_dir}", flush=True)

    resume_skip_existing = bool(
        OmegaConf.select(cfg, "extraction.resume_skip_existing") or False
    )
    if resume_skip_existing:
        print(
            "[resume] resume_skip_existing=true : buckets déjà extraits "
            "et valides seront sautés.",
            flush=True,
        )

    vocab = Vocab(n_ops=cfg.ssg.n_ops, n_noise=cfg.ssg.n_noise)

    # --- Datasets : reconstruit sweep + audit_svd ---
    sweep_full = _build_sweep_dataset(cfg)
    n_total = len(sweep_full)
    split = SplitConfig(
        train_oracle=cfg.dataset_split_smnist.train_oracle,
        audit_svd=cfg.dataset_split_smnist.audit_svd,
        init_phase3=cfg.dataset_split_smnist.init_phase3,
        seed=cfg.dataset_split_smnist.seed,
    )
    parts = split_indices(n_total, split)
    audit_indices = parts["audit_svd"].tolist()
    print(
        f"Tri-partition : audit_svd={len(audit_indices)} exemples"
        f" (sur {n_total} total)",
        flush=True,
    )

    # --- Modèle ---
    model_cfg = OracleConfig(
        vocab_size=vocab.size,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
        n_classes=cfg.model.n_classes, pad_id=vocab.PAD,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = _load_oracle(str(oracle_checkpoint), vocab, model_cfg, device=device)
    print(f"Oracle chargé depuis {oracle_checkpoint} sur device {device}", flush=True)

    extractor_cfg = ExtractorConfig(
        fp64=True,
        validate_numerics=False,
        empty_cache_per_layer=True,
        max_layers=None,
        stream_to_disk=None,
    )
    extractor = AttentionExtractor(oracle, config=extractor_cfg)

    # --- Pré-grouping par seq_len ---
    print("Pré-grouping audit_svd par seq_len (formule SSG)…", flush=True)
    t_group = time.perf_counter()
    seq_len_groups_raw = _group_by_seq_len(audit_indices, sweep_full)
    print(
        f"  {len(seq_len_groups_raw)} buckets seq_len bruts en "
        f"{time.perf_counter()-t_group:.1f}s : "
        f"{sorted(seq_len_groups_raw.keys())[:10]}…",
        flush=True,
    )
    # Application des caps (skip + size + count). Logge à stderr les buckets
    # affectés.
    skip_seq_len_above = OmegaConf.select(cfg, "extraction.skip_seq_len_above")
    max_bucket_size_gb = OmegaConf.select(cfg, "extraction.max_bucket_size_gb")
    max_examples_per_bucket = OmegaConf.select(cfg, "extraction.max_examples_per_bucket")
    seq_len_groups = _apply_bucket_caps(
        seq_len_groups_raw,
        L=cfg.model.n_layers,
        H=cfg.model.n_heads,
        skip_seq_len_above=(int(skip_seq_len_above) if skip_seq_len_above is not None else None),
        max_bucket_size_gb=(float(max_bucket_size_gb) if max_bucket_size_gb is not None else None),
        max_examples_per_bucket=(int(max_examples_per_bucket) if max_examples_per_bucket is not None else None),
        seed=cfg.ssg.seed_train,
    )
    n_examples_kept = sum(len(v) for v in seq_len_groups.values())
    n_examples_raw = sum(len(v) for v in seq_len_groups_raw.values())
    print(
        f"  Après caps : {len(seq_len_groups)} buckets, "
        f"{n_examples_kept}/{n_examples_raw} exemples conservés "
        f"({100*n_examples_kept/max(n_examples_raw,1):.1f}%)",
        flush=True,
    )

    # --- Manifest + MLflow run ---
    manifest = make_manifest(
        phase="1", sprint=cfg.sprint, domain=cfg.domain, seed=cfg.ssg.seed_train,
        config_path="OPS/configs/phase1/oracle_smnist.yaml",
        short_description="extract_v2",
        repo=REPO_ROOT,
    )
    write_manifest(manifest, REPO_ROOT)
    if manifest.git_dirty:
        _stderr(
            "[manifest] git_dirty=True → status=exploratory. Le run ne "
            "comptera pas comme registered strict. Commit avant relance "
            "pour obtenir registered."
        )

    t0 = time.perf_counter()
    hankel_tau = float(cfg.thresholds_phase1.hankel.tau)
    cap_batch = int(OmegaConf.select(cfg, "extraction.batch_size_cap") or 8)
    target_peak_gb = float(OmegaConf.select(cfg, "extraction.target_peak_gb") or 15.0)

    with start_run(
        experiment="phase1",
        run_name=manifest.run_id,
        phase="1", sprint=cfg.sprint, domain=cfg.domain,
        status=manifest.status,
    ):
        _safe_mlflow(
            "log_yaml_config",
            log_yaml_config,
            OmegaConf.to_container(cfg, resolve=True),
        )
        _safe_mlflow(
            "log_params",
            mlflow.log_params,
            {
                "oracle_checkpoint": str(oracle_checkpoint),
                "n_audit_examples": len(audit_indices),
                "n_seq_len_buckets_raw": len(seq_len_groups_raw),
                "n_seq_len_buckets_kept": len(seq_len_groups),
                "n_examples_after_caps": n_examples_kept,
                "skip_seq_len_above": str(skip_seq_len_above),
                "max_bucket_size_gb": str(max_bucket_size_gb),
                "max_examples_per_bucket": str(max_examples_per_bucket),
                "batch_size_cap": cap_batch,
                "extractor_fp64": extractor_cfg.fp64,
                "resume_skip_existing": str(resume_skip_existing),
                "git_dirty": str(manifest.git_dirty),
            },
        )

        n_dumps_saved = 0
        n_dumps_skipped = 0
        total_bytes = 0
        for seq_len in sorted(seq_len_groups.keys()):
            indices = seq_len_groups[seq_len]
            dump_path = output_dir / f"audit_dump_seq{seq_len}.pt"

            # Resume : skip si dump existant et valide
            if resume_skip_existing and dump_path.exists():
                if _validate_existing_dump(
                    dump_path, expected_seq_len=seq_len,
                    expected_L=cfg.model.n_layers,
                ):
                    size_b = dump_path.stat().st_size
                    total_bytes += size_b
                    print(
                        f">>> Bucket seq_len={seq_len} : SKIP (dump existant "
                        f"valide, {size_b / 1e6:.1f} MB)",
                        flush=True,
                    )
                    if bool(OmegaConf.select(cfg, "extraction.mlflow_log_dumps") or False):
                        _safe_mlflow(
                            f"log_artifact existing {dump_path.name}",
                            mlflow.log_artifact,
                            str(dump_path), artifact_path="audit_dumps",
                        )
                    n_dumps_skipped += 1
                    continue

            subset = Subset(sweep_full, indices)
            batch_size = _adaptive_batch_size(
                seq_len, L=cfg.model.n_layers, H=cfg.model.n_heads,
                target_gb=target_peak_gb, cap=cap_batch,
            )
            print(
                f"\n>>> Bucket seq_len={seq_len} : {len(indices)} ex, "
                f"batch_size={batch_size} (cap={cap_batch})",
                flush=True,
            )
            _print_resource_snapshot(f"  [resources pre seq={seq_len}]")
            t_bucket = time.perf_counter()
            dump = _extract_bucket(
                extractor, subset, pad_id=vocab.PAD, query_id=vocab.QUERY,
                batch_size=batch_size, n_layers=cfg.model.n_layers,
            )
            elapsed = time.perf_counter() - t_bucket
            print(
                f"    bucket seq={seq_len} : terminé en {elapsed:.1f}s "
                f"({len(indices)/elapsed:.2f} ex/s)",
                flush=True,
            )

            # Sauvegarde disque AVANT MLflow log_artifact : disque = source de
            # vérité, MLflow upload est optionnel.
            torch.save(dump, dump_path)
            size_b = dump_path.stat().st_size
            total_bytes += size_b
            print(
                f"    dump sauvegardé : {dump_path} "
                f"({size_b / 1e6:.1f} MB)",
                flush=True,
            )
            # Pour les dumps multi-bucket V2, on ÉVITE log_artifact qui copie
            # le fichier dans /mlruns (cf. carnet 2026-05-12 pod disk blowup :
            # MLflow file backend a doublé l'usage disque sur 89 GB). Les
            # dumps sont sur disque, rapatriés via rsync séparé. Override via
            # `extraction.mlflow_log_dumps=true` si nécessaire.
            if bool(OmegaConf.select(cfg, "extraction.mlflow_log_dumps") or False):
                _safe_mlflow(
                    f"log_artifact {dump_path.name}",
                    mlflow.log_artifact,
                    str(dump_path), artifact_path="audit_dumps",
                )
            n_dumps_saved += 1

            _log_bucket_metrics(dump, seq_len=seq_len, hankel_tau=hankel_tau)

            # Libérer activement le dump avant le bucket suivant (gros tensors FP64)
            del dump
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _safe_mlflow(
            "log_metric n_dumps_saved",
            mlflow.log_metric, "n_dumps_saved", n_dumps_saved,
        )
        _safe_mlflow(
            "log_metric n_dumps_skipped",
            mlflow.log_metric, "n_dumps_skipped", n_dumps_skipped,
        )
        _safe_mlflow(
            "log_metric total_bytes_dumps",
            mlflow.log_metric, "total_bytes_dumps", total_bytes,
        )
        finalize_manifest(manifest, duration_s=time.perf_counter() - t0,
                          repo_root=REPO_ROOT)
        print(
            f"\n=== Extraction V2 terminée === "
            f"{n_dumps_saved} dumps écrits, {n_dumps_skipped} sautés, "
            f"{total_bytes/1e9:.2f} GB total, "
            f"{time.perf_counter()-t0:.1f}s wall-clock",
            flush=True,
        )


if __name__ == "__main__":
    main()
