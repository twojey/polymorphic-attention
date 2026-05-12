"""
checkpoint.py — wrapper léger sur shared.checkpoint.Checkpoint.

Maintenu pour backward compat. L'implémentation est déléguée à
`shared.checkpoint.Checkpoint` ; ce module construit juste le fingerprint
spécifique au pipeline phase 2 (SVD + SRM + transfer law + batteries).

Étapes sauvegardées par convention :
1. svd_r_eff
2. srm_monovariate
3. transfer_law
4. head_spec
5. batteries_a / batteries_b / batteries_d
6. decoupling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.checkpoint import Checkpoint

# Alias backward-compat — Phase2State == Checkpoint avec fingerprint phase 2
Phase2State = Checkpoint


def create_or_resume(
    state_dir: Path | str,
    *,
    seq_lens: list[int],
    n_examples_total: int,
    svd_device: str,
    svd_precision: str,
) -> tuple[Checkpoint, bool]:
    """Crée ou reprend un checkpoint phase 2 avec fingerprint typed."""
    fingerprint: dict[str, Any] = {
        "seq_lens": sorted(seq_lens),
        "n_examples_total": n_examples_total,
        "svd_device": svd_device,
        "svd_precision": svd_precision,
    }
    return Checkpoint.create_or_resume(state_dir, fingerprint=fingerprint)
