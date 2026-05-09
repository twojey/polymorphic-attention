"""
test_5c_se_ht.py — Test 5c (SE et HT).

Spec : DOC/05 §3 (V3.5 : SE et HT séparés), ROADMAP 5.3.

- SE = Accuracy / Average_Rank        (Structural Efficiency)
- HT = Accuracy / Inference_Time      (Hardware Throughput)

Cible : SE_ASP ≥ 2 × SE_Transformer ET HT_ASP ≥ HT_SSM_pur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class SEHTResult:
    accuracy: float
    avg_rank: float           # pour ASP ; pour Transformer dense, == seq_len
    inference_time_s: float   # par exemple
    se: float
    ht: float


def measure_inference_time(
    *,
    forward_callable,
    n_warmup: int = 5,
    n_repeat: int = 20,
    sync_cuda: bool = True,
) -> float:
    """Mesure le temps moyen d'un forward (en secondes par appel).

    Pour mesure réelle (pas analytique, cf. ROADMAP 5.3) : warmup + N appels
    chronométrés. CUDA sync si GPU dispo.
    """
    for _ in range(n_warmup):
        forward_callable()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        forward_callable()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeat


def compute_se_ht(
    *,
    accuracy: float,
    avg_rank: float,
    inference_time_s: float,
    eps: float = 1e-9,
) -> SEHTResult:
    se = accuracy / max(avg_rank, eps)
    ht = accuracy / max(inference_time_s, eps)
    return SEHTResult(
        accuracy=accuracy, avg_rank=avg_rank, inference_time_s=inference_time_s,
        se=se, ht=ht,
    )


def passes_se_ht_targets(
    *,
    asp_result: SEHTResult,
    transformer_result: SEHTResult,
    ssm_result: SEHTResult,
    se_target_factor: float = 2.0,
    ht_min_ratio: float = 1.0,
) -> dict[str, bool]:
    return {
        "se_passed": asp_result.se >= se_target_factor * transformer_result.se,
        "ht_passed": asp_result.ht >= ht_min_ratio * ssm_result.ht,
    }
