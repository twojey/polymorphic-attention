"""
sprint_b_re_extract.py — Sprint B : re-extraction phase 1 V2 dumps SMNIST.

Spec : DOC/ROADMAP §3.9 + DOC/01_phase_metrologie.

Objectif : générer les 9 dumps multi-bucket (ω × Δ × ℋ) sur SMNIST Oracle
V2 conforme phase 1 V2. Pré-requis Sprint C (catalog level_research).

Robustesse :
- Mem check via shared.mem_guard avant chaque régime (abort si RAM < seuil).
- Timeout par régime (configurable, défaut 600s) via subprocess.Process —
  le thread/process est tué pour de vrai, la RAM est libérée.
- Skip explicite + log error pour chaque régime qui échoue, jamais silencieux.
- Logs INFO à chaque étape pour suivre l'avancement (visible dans sprint.log).

Pipeline :
1. Charger checkpoint SMNIST Oracle (issu phase 1 entraîné)
2. Sélectionner sweep régimes 3 × 3 (omega ∈ {0, 2, 4}, delta ∈ {16, 64, 256})
3. Pour chaque (ω, Δ) : extract_regime(n_examples=512) → AttentionDump FP64
4. Sauvegarder en NPZ pour reuse Sprint C
"""

from __future__ import annotations

import gc
import multiprocessing as mp
from pathlib import Path

import torch

from shared.mem_guard import MemoryGuardAbort, check_memory
from sprints.base import SprintBase


def _extract_worker(
    checkpoint_path: str,
    device: str,
    omega: int,
    delta: int,
    n_examples: int,
    out_path: str,
    queue: "mp.Queue",
) -> None:
    """Worker subprocess : charge oracle, extrait un régime, save dump.

    Tourne dans un Process séparé pour pouvoir le killer net en cas de
    timeout. Communique son statut via la queue (ok/error + traceback).
    """
    try:
        from catalog.oracles import SMNISTOracle  # noqa: PLC0415
        from catalog.oracles.base import RegimeSpec  # noqa: PLC0415

        oracle = SMNISTOracle(checkpoint_path=checkpoint_path, device=device)
        regime = RegimeSpec(omega=omega, delta=delta, entropy=0.0)
        dump = oracle.extract_regime(regime, n_examples)
        dump.validate()
        torch.save({
            "attn": dump.attn,
            "omegas": dump.omegas,
            "deltas": dump.deltas,
            "entropies": dump.entropies,
            "tokens": dump.tokens,
            "query_pos": dump.query_pos,
            "metadata": dump.metadata,
        }, out_path)
        queue.put(("ok", out_path))
    except BaseException as exc:  # noqa: BLE001
        import traceback as tb  # noqa: PLC0415
        queue.put(("error", f"{type(exc).__name__}: {exc}\n{tb.format_exc()}"))


class SprintBReExtract(SprintBase):
    """Sprint B — re-extraction dumps phase 1 V2."""

    sprint_id = "B_re_extract"
    expected_duration_hint = "30 min - 1 h sur pod CPU"
    expected_compute_cost = "$0.30 (RunPod CPU)"
    requires_pod = True  # pod recommandé (CPU suffit)

    def __init__(
        self,
        *,
        oracle_checkpoint: str | Path,
        omegas: tuple[int, ...] = (0, 2, 4),
        deltas: tuple[int, ...] = (16, 64, 256),
        n_examples_per_regime: int = 512,
        device: str = "cpu",
        regime_timeout_s: float = 600.0,
        min_available_gb: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_checkpoint = Path(oracle_checkpoint)
        self.omegas = omegas
        self.deltas = deltas
        self.n_examples = n_examples_per_regime
        self.device = device
        self.regime_timeout_s = regime_timeout_s
        self.min_available_gb = min_available_gb

    def _run_inner(self) -> None:
        self._check_go_nogo(
            "oracle_checkpoint_exists",
            self.oracle_checkpoint.is_file(),
            skip_if_failed=True,
        )

        # Mem check global au démarrage
        check_memory(
            min_available_gb=self.min_available_gb,
            label="sprint-B start",
            logger=self.logger,
            abort=True,
        )

        # Probe oracle infos pour log (charge puis libère)
        self._log_oracle_info()

        # Boucle régimes avec checkpoint/resume + subprocess + mem check
        dumps_dir = self.output_dir / "dumps"
        dumps_dir.mkdir(exist_ok=True)
        regimes = [(o, d) for o in self.omegas for d in self.deltas]
        n_total = len(regimes)
        n_done = 0
        n_skipped = 0
        n_resumed = 0

        for idx, (omega, delta) in enumerate(regimes, start=1):
            key = f"dump_omega{omega}_delta{delta}"
            dump_path = dumps_dir / f"{key}.pt"
            self.logger.info(
                "[%s] régime %d/%d : ω=%d Δ=%d (key=%s)",
                self.sprint_id, idx, n_total, omega, delta, key,
            )

            if self._checkpoint_has(key) and dump_path.is_file():
                self.logger.info("[%s] %s : déjà extrait — resume", self.sprint_id, key)
                n_resumed += 1
                n_done += 1
                continue

            try:
                check_memory(
                    min_available_gb=self.min_available_gb,
                    label=f"avant régime ω={omega} Δ={delta}",
                    logger=self.logger,
                    abort=True,
                )
            except MemoryGuardAbort as exc:
                self.logger.error("[%s] abort sur mémoire insuffisante : %s",
                                  self.sprint_id, exc)
                n_skipped += 1
                # On arrête tout : si la RAM est saturée, les régimes suivants
                # crasheront pareil. Mieux vaut sortir proprement.
                break

            status, info = self._extract_one_regime(
                omega=omega, delta=delta, out_path=str(dump_path),
            )

            if status == "ok":
                self._checkpoint_save(key, str(dump_path))
                self._add_artifact(dump_path, f"dump (ω={omega}, Δ={delta})")
                n_done += 1
                self.logger.info(
                    "[%s] ✓ régime ω=%d Δ=%d extrait → %s",
                    self.sprint_id, omega, delta, dump_path,
                )
            elif status == "timeout":
                self.logger.error(
                    "[%s] ✗ régime ω=%d Δ=%d TIMEOUT après %.0fs — skip",
                    self.sprint_id, omega, delta, self.regime_timeout_s,
                )
                n_skipped += 1
            else:
                self.logger.error(
                    "[%s] ✗ régime ω=%d Δ=%d FAILED : %s",
                    self.sprint_id, omega, delta, info,
                )
                n_skipped += 1

            # Libère explicitement avant le prochain régime
            gc.collect()

        self._log_metric("n_dumps_produced", n_done)
        self._log_metric("n_regimes_target", n_total)
        self._log_metric("n_regimes_skipped", n_skipped)
        self._log_metric("n_regimes_resumed", n_resumed)

        self._check_go_nogo(
            "min_dumps_produced",
            n_done >= n_total - 1,
            skip_if_failed=False,
        )

    def _log_oracle_info(self) -> None:
        """Charge l'oracle juste pour logger ses dimensions, puis le libère."""
        from catalog.oracles import SMNISTOracle  # noqa: PLC0415

        oracle = SMNISTOracle(
            checkpoint_path=str(self.oracle_checkpoint), device=self.device,
        )
        self._log_metric("oracle_id", oracle.oracle_id)
        self._log_metric("n_layers", oracle.n_layers)
        self._log_metric("n_heads", oracle.n_heads)
        del oracle
        gc.collect()

    def _extract_one_regime(
        self, *, omega: int, delta: int, out_path: str,
    ) -> tuple[str, str]:
        """Lance l'extraction d'un régime dans un subprocess séparé.

        Returns
        -------
        (status, info) où status ∈ {"ok", "timeout", "error"}.
        info est le path du dump (status=ok) ou un message d'erreur.
        """
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_extract_worker,
            kwargs=dict(
                checkpoint_path=str(self.oracle_checkpoint),
                device=self.device,
                omega=omega,
                delta=delta,
                n_examples=self.n_examples,
                out_path=out_path,
                queue=queue,
            ),
            name=f"extract-ω{omega}Δ{delta}",
            daemon=False,
        )
        self.logger.info(
            "[%s] subprocess start ω=%d Δ=%d (timeout %.0fs)",
            self.sprint_id, omega, delta, self.regime_timeout_s,
        )
        proc.start()
        proc.join(timeout=self.regime_timeout_s)

        if proc.is_alive():
            self.logger.error(
                "[%s] subprocess ω=%d Δ=%d still alive after timeout, killing PID %d",
                self.sprint_id, omega, delta, proc.pid,
            )
            proc.terminate()
            proc.join(timeout=10.0)
            if proc.is_alive():
                self.logger.error(
                    "[%s] subprocess ω=%d Δ=%d refused SIGTERM, SIGKILL PID %d",
                    self.sprint_id, omega, delta, proc.pid,
                )
                proc.kill()
                proc.join(timeout=5.0)
            return "timeout", f"pid {proc.pid}"

        # Process terminé : récupérer le statut depuis la queue
        if not queue.empty():
            status, info = queue.get_nowait()
            return status, info
        # Queue vide mais process mort → crash sans message
        exit_code = proc.exitcode
        return "error", f"exit code {exit_code}, queue empty"
