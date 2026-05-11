#!/usr/bin/env bash
# run_phase1b_skl.sh — Wrapper Run 3 phase 1.5 (S_KL option C amendement V1.5).
#
# Origine : remplace l'ancien /root/run3_skl.sh (hors repo) qui a crashé
# silencieusement le 2026-05-11 12:12:05 pendant calibration baseline.
# Causes identifiées :
#   (1) stderr non capturé par le wrapper externe (`> log` sans `2>&1`)
#   (2) batch_size=4 trop ambitieux à seq_len=4371 (~14.7 GB d'attentions FP64,
#       sans compter les activations FFN du forward).
#
# Patches appliqués :
#   - CODE/phase1b_calibration_signal/run.py : cap batch_size=2 + logging défensif
#   - launch_phase1b.sh refactoré (common.sh, OPS/logs/, trap ERR, env capture)
#   - Ce wrapper : thin pass-through avec arguments S_KL pré-configurés
#
# Usage :
#   bash OPS/scripts/run_phase1b_skl.sh                # foreground
#   bash OPS/scripts/run_phase1b_skl.sh --nohup        # background détaché
#
# Logs : OPS/logs/run3_skl_<UTC_TS>.log (géré par launch_phase1b.sh).
#
# Codes de sortie :
#   0   succès Python
#   1   options invalides
#   2   pré-condition manquante
#   N   exit code Python (propagé)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PASSTHRU=()
case "${1:-}" in
    --nohup)
        PASSTHRU+=(--nohup)
        ;;
    --watch)
        PASSTHRU+=(--watch)
        ;;
    --help|-h)
        sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 0
        ;;
    "")
        ;;
    *)
        echo "Erreur: argument inconnu '$1' (utiliser --nohup, --watch ou --help)" >&2
        exit 1
        ;;
esac

# Délègue à launch_phase1b.sh — c'est lui qui :
#   - init le logging persistant (OPS/logs/run3_skl_<TS>.log)
#   - capture l'env (commit, GPU, BLAS, PYTHONPATH)
#   - vérifie les pré-conditions (uv, checkpoint, repo)
#   - injecte les env vars BLAS / MLflow / PYTHONUNBUFFERED
#   - gère le mode --nohup proprement (PID + log path affichés)
exec bash "$REPO_ROOT/OPS/env/launch_phase1b.sh" \
    "${PASSTHRU[@]}" \
    --log-prefix run3_skl \
    -- \
    bench.n_examples=2000 \
    s_kl.enabled=true \
    "bench.structured_deltas=[16,64,256]" \
    s_kl.baseline.n_calibration_examples=256
