"""Fixtures partagées tests phase 1.5.

Garde-fou BLAS : `compute_s_spectral` refuse de tourner si les variables
`OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `OMP_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` ne sont pas à `"1"` — protection contre le deadlock
eigvalsh batché (cf. DOC/carnet 2026-05-11). En production, `launch_phase1b.sh`
les exporte avant l'import torch. En test, on les set en fixture autouse pour
que le check runtime passe ; le check lui-même reste en place côté production.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_blas_single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(var, "1")
