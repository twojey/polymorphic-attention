"""bench_cpu_vs_gpu_properties.py — Benchmark wall-clock CPU vs GPU sur properties représentatives.

Objectif : décider per-property si GPU vaut le coût (règle ≥3× speedup ET fit VRAM).
Sortie : table Markdown dans OPS/logs/benchmarks/bench_<UTC>.md.

Usage typique sur pod chaud :
    PYTHONPATH=CODE uv run python OPS/scripts/bench_cpu_vs_gpu_properties.py \\
        --dumps-dir OPS/logs/sprints/B/dumps \\
        --repeats 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import logging
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CODE"))

import catalog.properties  # noqa: F401 — déclenche register_property
from catalog.properties.base import PropertyContext
from catalog.properties.registry import REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bench")

# Properties représentatives par pattern (1-2 par catégorie attendue) ────────
REPRESENTATIVES = [
    # (property_name, expected_pattern, note)
    ("L1_fft2d_energy", "FFT 2D", "GPU candidat fort"),
    ("L3_quasi_periodicity", "FFT 1D", "GPU candidat"),
    ("W2_dependence_proxy", "cdist", "GPU candidat (post-fix)"),
    ("D1_head_cosine", "cdist/cosine", "GPU candidat"),
    ("B8_sylvester_rank", "SVD FP64", "CPU obligatoire (Blackwell FP64 limité)"),
    ("O1_toeplitz_displacement_rank", "SVD FP64", "CPU obligatoire"),
    ("B1_toeplitz_distance", "scalaire", "petit, à mesurer"),
    ("B4_sparse_fraction", "scalaire", "petit, à mesurer"),
    ("C3_shannon_entropy", "scalaire", "petit, CPU OK"),
    ("I1_head_diversity", "einsum/cosine", "à mesurer"),
    ("R3_bochner_stationarity", "FFT + SVD", "mix, à mesurer"),
]


def time_compute(prop, A: torch.Tensor, ctx: PropertyContext, repeats: int) -> dict:
    """Run prop.compute() warmup + repeats × et retourne stats wall-clock."""
    try:
        # Warmup
        _ = prop.compute(A, ctx)
        if A.device.type == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return {"status": "OOM", "msg": str(e).split("\n")[0][:80]}
    except Exception as e:
        return {"status": "ERROR", "msg": f"{type(e).__name__}: {e}"[:120]}

    times = []
    for _ in range(repeats):
        gc.collect()
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            _ = prop.compute(A, ctx)
            if A.device.type == "cuda":
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            return {"status": "OOM", "msg": str(e).split("\n")[0][:80]}
        except Exception as e:
            return {"status": "ERROR", "msg": f"{type(e).__name__}: {e}"[:120]}
        times.append(time.perf_counter() - t0)

    return {
        "status": "OK",
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "n_repeats": len(times),
    }


def load_dump(path: Path, max_samples: int, layer_idx: int = 0) -> torch.Tensor:
    """Charge un dump .pt, retourne A en (B, H, N, N) pour layer_idx donné.

    Format dump Sprint B : dict avec key "attn" = list[Tensor(B,H,N,N)] (un par layer).
    Bench sur un seul layer = représentatif du coût per-prop.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "attn" in obj:
        attn = obj["attn"]
        if isinstance(attn, list):
            A = attn[layer_idx]
        else:
            A = attn
    elif isinstance(obj, dict):
        A = next(v for v in obj.values() if isinstance(v, torch.Tensor))
    else:
        A = obj

    if A.ndim == 5:  # (B, L, H, N, N) → flatten L into B
        B, L, H, N, _ = A.shape
        A = A.reshape(B * L, H, A.shape[-2], A.shape[-1])
    if A.ndim != 4:
        raise ValueError(f"Dump {path.name} format inattendu: shape={A.shape}")
    if A.shape[0] > max_samples:
        A = A[:max_samples]
    return A.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dumps-dir",
        type=Path,
        default=Path("OPS/logs/sprints/B/dumps"),
        help="Répertoire contenant dump_*.pt",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Cap B sur chaque dump pour bench reproductible (default 4)",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--properties",
        nargs="+",
        default=[name for name, _, _ in REPRESENTATIVES],
        help="Liste des property names à bencher",
    )
    args = ap.parse_args()

    if args.output is None:
        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / "OPS/logs/benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"bench_cpu_vs_gpu_{ts}.md"

    # Sélectionner les régimes : 1 petit (Δ=16), 1 moyen (Δ=64), 1 gros (Δ=256)
    dumps_dir = args.dumps_dir
    selected = []
    for tag in ("delta16", "delta64", "delta256"):
        candidates = sorted(dumps_dir.glob(f"*omega0*{tag}*.pt"))
        if candidates:
            selected.append((tag, candidates[0]))

    if not selected:
        log.error("Aucun dump trouvé dans %s", dumps_dir)
        sys.exit(1)

    log.info("Régimes sélectionnés : %s", [t for t, _ in selected])
    log.info("Properties à bencher : %d", len(args.properties))
    log.info("Sortie : %s", args.output)

    rows = []  # rows[i] = (prop_name, regime_tag, cpu_result, gpu_result, A_shape)
    cuda_avail = torch.cuda.is_available()

    for regime_tag, dump_path in selected:
        log.info("=== Régime %s (%s) ===", regime_tag, dump_path.name)
        A_cpu = load_dump(dump_path, max_samples=args.max_samples)
        log.info("  A shape = %s, dtype = %s", tuple(A_cpu.shape), A_cpu.dtype)

        # Pré-charger une copie GPU si dispo
        A_gpu = A_cpu.to("cuda") if cuda_avail else None

        for prop_name in args.properties:
            try:
                cls = REGISTRY.get(prop_name)
            except KeyError:
                log.warning("Property '%s' introuvable dans REGISTRY, skip", prop_name)
                rows.append((prop_name, regime_tag, {"status": "NOT_FOUND"}, {"status": "NOT_FOUND"}, tuple(A_cpu.shape)))
                continue
            prop = cls()
            requires_fp64 = getattr(prop, "requires_fp64", False)

            # CPU run
            ctx_cpu = PropertyContext(
                device="cpu", dtype=torch.float64 if requires_fp64 else torch.float32
            )
            A_for_cpu = A_cpu.to(ctx_cpu.dtype)
            log.info("  [CPU] %s ...", prop_name)
            cpu_res = time_compute(prop, A_for_cpu, ctx_cpu, args.repeats)

            # GPU run
            if cuda_avail:
                ctx_gpu = PropertyContext(
                    device="cuda",
                    dtype=torch.float64 if requires_fp64 else torch.float32,
                )
                A_for_gpu = A_gpu.to(ctx_gpu.dtype)
                log.info("  [GPU] %s ...", prop_name)
                gpu_res = time_compute(prop, A_for_gpu, ctx_gpu, args.repeats)
                torch.cuda.empty_cache()
            else:
                gpu_res = {"status": "NO_CUDA"}

            rows.append((prop_name, regime_tag, cpu_res, gpu_res, tuple(A_cpu.shape)))

        # Free GPU after régime done
        if A_gpu is not None:
            del A_gpu
            torch.cuda.empty_cache()
        del A_cpu
        gc.collect()

    # Format markdown ──────────────────────────────────────────────────────
    lines = [
        f"# Benchmark CPU vs GPU properties — {dt.datetime.utcnow().isoformat()}Z",
        "",
        f"- Pod: `{torch.cuda.get_device_name(0) if cuda_avail else 'CPU-only'}` "
        f"({torch.cuda.get_device_properties(0).total_memory // 1024**3 if cuda_avail else 'N/A'} GB VRAM)",
        f"- PyTorch: `{torch.__version__}`",
        f"- Repeats: {args.repeats} (median)",
        f"- Max samples per régime: {args.max_samples}",
        "",
        "## Résultats",
        "",
        "| Property | Régime | A shape | CPU (s) | GPU (s) | Speedup CPU/GPU | Décision |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for prop_name, regime, cpu_r, gpu_r, shape in rows:
        def fmt(r):
            if r["status"] == "OK":
                return f"{r['median_s']:.3f}"
            return r["status"]

        cpu_s = fmt(cpu_r)
        gpu_s = fmt(gpu_r)

        if cpu_r["status"] == "OK" and gpu_r["status"] == "OK":
            speedup = cpu_r["median_s"] / gpu_r["median_s"]
            decision = "🟢 GPU" if speedup >= 3.0 else "🟡 CPU (overhead)"
            speedup_s = f"{speedup:.2f}×"
        elif gpu_r["status"] == "OOM":
            decision = "🔴 CPU obligatoire (OOM)"
            speedup_s = "—"
        elif gpu_r["status"] == "NO_CUDA":
            decision = "—"
            speedup_s = "—"
        else:
            decision = "⚠ ERROR"
            speedup_s = "—"
        lines.append(
            f"| `{prop_name}` | {regime} | {shape} | {cpu_s} | {gpu_s} | {speedup_s} | {decision} |"
        )

    lines.extend(["", "## Notes", ""])
    for name, pattern, note in REPRESENTATIVES:
        if name in args.properties:
            lines.append(f"- **{name}** ({pattern}) — {note}")

    args.output.write_text("\n".join(lines) + "\n")
    log.info("Rapport écrit : %s", args.output)
    print(args.output)


if __name__ == "__main__":
    main()
