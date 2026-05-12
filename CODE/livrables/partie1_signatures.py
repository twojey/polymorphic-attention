"""
partie1_signatures.py — Signature textuelle par Oracle.

Spec : DOC/CATALOGUE §5.2 "Signatures par Oracle".

Pour chaque Oracle, produit un résumé textuel des classes mathématiques
identifiées :
  "Llama 3.2 1B est compatible avec : low-rank effectif (A1), pas Mercer
  (R1 ❌), pas Toeplitz (B1 ❌)"

Heuristiques par famille :
- Famille A (spectral) : low-rank ssi r_eff_median < N / 4
- Famille B : Toeplitz ssi B1_toeplitz_distance < 0.10
- Famille U : structure compressible ssi epsilon_* < 0.10
- etc.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _median_metric(
    results_dict: dict[str, Any], prop: str, metric: str
) -> float | None:
    """Médiane cross-régime d'une (prop, metric)."""
    vals: list[float] = []
    for regime_out in results_dict.get("per_regime", {}).values():
        prop_out = regime_out.get(prop, {})
        v = prop_out.get(metric) if isinstance(prop_out, dict) else None
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return statistics.median(vals) if vals else None


def _check(results: dict, prop: str, metric: str, op: str, thresh: float) -> bool | None:
    v = _median_metric(results, prop, metric)
    if v is None:
        return None
    if op == "<":
        return v < thresh
    if op == "<=":
        return v <= thresh
    if op == ">":
        return v > thresh
    if op == ">=":
        return v >= thresh
    raise ValueError(f"op invalide : {op}")


def build_signature_for_oracle(results_dict: dict[str, Any], oracle_id: str,
                                N_max: int = 64) -> dict[str, Any]:
    """Construit la signature textuelle pour un Oracle.

    Returns
    -------
    {oracle_id, properties_satisfied : list[str], properties_refuted : list[str],
     properties_unknown : list[str], summary_text}
    """
    # Heuristiques (prop, metric, op, threshold, description)
    rules: list[tuple[str, str, str, float, str]] = [
        ("A1_r_eff_theta099", "r_eff_median", "<", N_max // 4, "low-rank effectif (A1)"),
        ("A3_condition_number", "log10_kappa_median", "<", 6.0, "bien conditionné (A3)"),
        ("A4_spectral_entropy", "spectral_entropy_norm_median", "<", 0.5, "concentré spectralement (A4)"),
        ("B1_toeplitz_distance", "toeplitz_distance_relative_median", "<", 0.10, "Toeplitz (B1)"),
        ("B5_block_diagonal_distance", "block_diag_relative_distance_median", "<", 0.15, "block-diagonal (B5)"),
        ("B6_banded_distance", "rel_distance_median", "<", 0.15, "banded (B6)"),
        ("B4_sparse_fraction", "sparse_fraction_median", ">", 0.50, "sparse (B4)"),
        ("O1_toeplitz_displacement_rank", "fraction_rank_le_2_strict", ">", 0.50,
         "Toeplitz-like rang déplacement (O1)"),
        ("P1_hankel_realization", "fraction_low_order_le_3", ">", 0.50,
         "Ho-Kalman ordre faible (P1)"),
        ("Q2_hss_rank", "fraction_hss_rank_le_3", ">", 0.50, "HSS rang faible (Q2)"),
        ("U1_butterfly_distance", "epsilon_butterfly_median", "<", 0.20, "Butterfly (U1)"),
        ("U2_monarch_distance", "epsilon_monarch_median", "<", 0.20, "Monarch (U2)"),
        ("U5_sparse_plus_lowrank", "epsilon_splr_median", "<", 0.20, "Sparse+LowRank (U5)"),
        ("R1_mercer_psd", "fraction_psd", ">", 0.50, "Mercer PSD (R1)"),
        ("R3_bochner_stationarity", "fraction_pass_bochner", ">", 0.50, "Bochner stationnaire (R3)"),
    ]
    satisfied: list[str] = []
    refuted: list[str] = []
    unknown: list[str] = []
    for prop, metric, op, thresh, desc in rules:
        result = _check(results_dict, prop, metric, op, thresh)
        if result is None:
            unknown.append(desc)
        elif result:
            satisfied.append(desc)
        else:
            refuted.append(desc)

    summary = f"### {oracle_id}\n\n"
    if satisfied:
        summary += "**Compatible avec** : " + ", ".join(satisfied) + ".\n\n"
    if refuted:
        summary += "**Non compatible avec** : " + ", ".join(refuted) + ".\n\n"
    if unknown:
        summary += f"**Indéterminé** ({len(unknown)} propriétés) : " + ", ".join(unknown) + ".\n\n"

    return {
        "oracle_id": oracle_id,
        "properties_satisfied": satisfied,
        "properties_refuted": refuted,
        "properties_unknown": unknown,
        "summary_text": summary,
    }


def build_all_signatures(
    results_per_oracle: dict[str, dict[str, Any]], N_max: int = 64,
) -> tuple[list[dict[str, Any]], str]:
    """Build signatures pour tous les Oracles."""
    sigs = [build_signature_for_oracle(rd, oid, N_max=N_max)
            for oid, rd in results_per_oracle.items()]
    md = "# Signatures par Oracle — Partie 1\n\n"
    md += "Heuristiques de classification : voir CODE/livrables/partie1_signatures.py.\n\n"
    for s in sigs:
        md += s["summary_text"]
    return sigs, md


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="livrables.partie1_signatures")
    p.add_argument("--results", nargs="+", required=True,
                  help="path1.json:oracle_id1 ...")
    p.add_argument("--output", required=True)
    p.add_argument("--N-max", type=int, default=64)
    args = p.parse_args()

    results_per_oracle: dict[str, dict] = {}
    for spec in args.results:
        path, oid = spec.rsplit(":", 1)
        with open(path) as f:
            results_per_oracle[oid] = json.load(f)

    sigs, md = build_all_signatures(results_per_oracle, N_max=args.N_max)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signatures_per_oracle.json").write_text(json.dumps(sigs, indent=2))
    (out_dir / "signatures_per_oracle.md").write_text(md)
    print(f"=== Signatures écrites : {out_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
