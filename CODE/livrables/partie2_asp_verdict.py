"""
partie2_asp_verdict.py — Verdict ASP final (Partie 2).

Spec : DOC/FALSIFIABILITE + DOC/05_phase_pareto.

Le verdict ASP est GO / NO-GO basé sur les critères de phase 5 :
- 5a identifiabilité : R_max ↔ r_target prediction valide
- 5b élasticité : ASP maintient qualité quand r_target varie
- 5c self-emergent half-truth : Spectromètre converge
- 5d anti-fraud : pas de "cheating" sur signaux faciles
- 5e OOD robustness : tient hors distribution
- 6c R_max = r_med / 2 : ≥ 95 % qualité Oracle

ASP est validé ssi 5a + 5c + 6c passent (5b, 5d, 5e bonus).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VerdictResult:
    verdict: str  # "GO" | "NO-GO" | "PARTIAL"
    mandatory_passed: int
    mandatory_total: int
    bonus_passed: int
    bonus_total: int
    details: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "mandatory_passed": self.mandatory_passed,
            "mandatory_total": self.mandatory_total,
            "bonus_passed": self.bonus_passed,
            "bonus_total": self.bonus_total,
            "details": self.details,
        }


MANDATORY_TESTS = ("5a", "5c", "6c")
BONUS_TESTS = ("5b", "5d", "5e")


def build_verdict(test_results: dict[str, dict[str, Any]]) -> VerdictResult:
    """Calcule verdict global.

    Args
    ----
    test_results : {"5a": {"passed": True, ...}, "5b": {...}, ...}

    Returns
    -------
    VerdictResult
    """
    details: dict[str, dict[str, Any]] = {}
    m_passed = 0
    b_passed = 0
    for tid in MANDATORY_TESTS + BONUS_TESTS:
        r = test_results.get(tid, {"passed": None, "note": "missing"})
        details[tid] = r
        if r.get("passed") is True:
            if tid in MANDATORY_TESTS:
                m_passed += 1
            else:
                b_passed += 1

    if m_passed == len(MANDATORY_TESTS):
        if b_passed == len(BONUS_TESTS):
            verdict = "GO"
        else:
            verdict = "PARTIAL"  # mandatory OK mais bonus partiel
    else:
        verdict = "NO-GO"

    return VerdictResult(
        verdict=verdict,
        mandatory_passed=m_passed,
        mandatory_total=len(MANDATORY_TESTS),
        bonus_passed=b_passed,
        bonus_total=len(BONUS_TESTS),
        details=details,
    )


def render_markdown(verdict: VerdictResult) -> str:
    icon = {"GO": "✅", "PARTIAL": "🟡", "NO-GO": "❌"}.get(verdict.verdict, "❓")
    md = f"# Verdict ASP — Partie 2\n\n"
    md += f"## {icon} Verdict global : **{verdict.verdict}**\n\n"
    md += f"- Tests mandatory passés : **{verdict.mandatory_passed} / {verdict.mandatory_total}**\n"
    md += f"- Tests bonus passés : {verdict.bonus_passed} / {verdict.bonus_total}\n\n"
    md += "## Détail par test\n\n"
    md += "| Test | Type | Statut | Note |\n|---|---|---|---|\n"
    for tid in MANDATORY_TESTS + BONUS_TESTS:
        r = verdict.details.get(tid, {})
        ttype = "mandatory" if tid in MANDATORY_TESTS else "bonus"
        passed = r.get("passed")
        status = "✅" if passed is True else ("❌" if passed is False else "❓")
        note = r.get("note", "")
        md += f"| {tid} | {ttype} | {status} | {note} |\n"
    md += "\n"
    if verdict.verdict == "GO":
        md += "**Conclusion** : l'hypothèse ASP est validée empiriquement sur "
        md += "l'Oracle considéré. La Partie 1 a montré une structure exploitable, "
        md += "la Partie 2 démontre qu'elle peut être routée dynamiquement avec "
        md += "qualité préservée.\n"
    elif verdict.verdict == "PARTIAL":
        md += "**Conclusion** : ASP est viable sur les critères mandatory mais "
        md += "des limites apparaissent sur les tests bonus (élasticité, OOD, "
        md += "anti-fraud). Approfondir avant publication.\n"
    else:
        md += "**Conclusion** : l'hypothèse ASP est rejetée sur cet Oracle. "
        md += "Soit la structure n'est pas exploitable dynamiquement, soit "
        md += "l'architecture testée est sous-dimensionnée. Documenter pour "
        md += "publication négative.\n"
    return md


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="livrables.partie2_asp_verdict")
    p.add_argument("--test-results", required=True,
                  help="JSON {test_id: {passed, note, ...}}")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.test_results) as f:
        test_results = json.load(f)
    verdict = build_verdict(test_results)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2))
    (out_dir / "verdict.md").write_text(render_markdown(verdict))
    print(f"=== Verdict {verdict.verdict} écrit dans {out_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
