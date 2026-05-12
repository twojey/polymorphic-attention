#!/usr/bin/env bash
# launch_phase1b.sh — Lancer phase 1.5 avec safeguards BLAS + traces robustes.
#
# Refactor 2026-05-11 (carnet : Refactor scripts robustes) suite Run 3 silent crash :
# - Logs persistants dans OPS/logs/ (PAS /tmp/) via lib/common.sh
# - stderr capturé via exec 2>&1
# - trap ERR avec line number et commande
# - Env capture (commit, GPU, BLAS, PYTHONPATH) en début
# - PID + LOG_FILE affichés explicitement pour récupération
#
# Usage:
#   bash OPS/setup/launch_phase1b.sh [OPTIONS] -- [HYDRA_OVERRIDES]
#
# Options:
#   --nohup              Lancer en background (nohup + disown)
#   --watch              Afficher les logs en temps réel (tail -f)
#   --checkpoint PATH    Override le path checkpoint Oracle (sinon valeur par défaut)
#   --log-prefix PFX     Préfixe du log (défaut : "phase1b")
#   --help               Afficher cette aide
#
# Exemples:
#   bash OPS/setup/launch_phase1b.sh -- bench.n_examples=2000
#   bash OPS/setup/launch_phase1b.sh --nohup --log-prefix run4 -- s_spectral.K=32
#   bash OPS/setup/launch_phase1b.sh --nohup --watch -- s_kl.enabled=true
#
# Codes de sortie :
#   0   succès complet (Python a return 0)
#   1   erreur de parsing options
#   2   pré-condition manquante (uv, checkpoint, repo)
#   N   exit code Python (propagation)
#
# Logs : OPS/logs/<prefix>_<UTC_TS>.log

# shellcheck source=../scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/lib/common.sh"
asp::init_strict_mode

# --- Parse options -----------------------------------------------------------

NOHUP_MODE=0
WATCH_MODE=0
LOG_PREFIX="phase1b"
CHECKPOINT_PATH_DEFAULT="OPS/checkpoints/oracle_e2f0b5e.ckpt"
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup)
            NOHUP_MODE=1
            shift
            ;;
        --watch)
            WATCH_MODE=1
            shift
            ;;
        --checkpoint)
            CHECKPOINT_PATH="${2:?--checkpoint requiert un argument}"
            shift 2
            ;;
        --log-prefix)
            LOG_PREFIX="${2:?--log-prefix requiert un argument}"
            shift 2
            ;;
        --help|-h)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            asp::die 1 "option inconnue : $1 (--help pour usage)"
            ;;
        *)
            asp::die 1 "argument inattendu : $1 (utiliser -- avant les overrides Hydra)"
            ;;
    esac
done

HYDRA_OVERRIDES=("$@")
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_PATH_DEFAULT}"

# --- Init logging (foreground only — pour nohup on relance bash dans le child) ---

ASP_REPO_ROOT="$(asp::_resolve_repo_root)"

if [[ "$NOHUP_MODE" -eq 0 ]]; then
    asp::init_logging "$LOG_PREFIX"
    asp::log_env
fi

# --- Pré-conditions ---------------------------------------------------------

asp::section "Pré-conditions"
asp::require_cmd uv
asp::require_file "$CHECKPOINT_PATH"
[[ -d "$ASP_REPO_ROOT/CODE" ]] || asp::die 2 "CODE/ introuvable dans $ASP_REPO_ROOT"
asp::log "checkpoint : $CHECKPOINT_PATH"
asp::log "hydra_overrides : ${HYDRA_OVERRIDES[*]:-<aucun>}"

# --- Env vars (BLAS + MLflow + PYTHONPATH) ---------------------------------

asp::section "Env vars phase 1.5"

# Anti-deadlock BLAS (cf. carnet 2026-05-11 entrée 06:31 deadlock 38h)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python stdout unbuffered (sans ça, on perd les prints si SIGKILL/OOM)
export PYTHONUNBUFFERED=1

# MLflow (sinon start_run crashe — cf. carnet 2026-05-11 06:31)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

# PYTHONPATH (CODE n'est pas installé en editable)
export PYTHONPATH="$ASP_REPO_ROOT/CODE${PYTHONPATH:+:$PYTHONPATH}"

# Bookkeeping pour le subprocess Python
export ASP_LOG_FILE
export ASP_LOG_PREFIX

asp::log "BLAS=1 (anti-deadlock) PYTHONUNBUFFERED=1 MLFLOW=$MLFLOW_TRACKING_URI"

# --- Exécution --------------------------------------------------------------

asp::section "Lancement"

cd "$ASP_REPO_ROOT"

PYTHON_CMD=(
    uv run python -u -m phase1b_calibration_signal.run
    --config-path=../../OPS/configs/phase1b --config-name=signals
    "oracle_checkpoint=$CHECKPOINT_PATH"
    "${HYDRA_OVERRIDES[@]}"
)

if [[ "$NOHUP_MODE" -eq 1 ]]; then
    # En mode nohup, on init_logging dans le child shell pour qu'il ait son log
    # propre (le parent va exit immédiatement et perdre sa session).
    ASP_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
    CHILD_LOG="$ASP_REPO_ROOT/OPS/logs/${LOG_PREFIX}_${ASP_TS}.log"
    mkdir -p "$(dirname "$CHILD_LOG")"

    # Env capture en début du log child (avant le nohup detach)
    {
        printf '[%s] [init] log=%s pid=$$ (background)\n' "$(asp::_ts)" "$CHILD_LOG"
        printf '[%s] commit=%s\n' "$(asp::_ts)" "$(git -C "$ASP_REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo n/a)"
        printf '[%s] cmd=%s\n' "$(asp::_ts)" "${PYTHON_CMD[*]}"
        printf '[%s] BLAS=1 PYTHONPATH=%s MLFLOW=%s\n' "$(asp::_ts)" "$PYTHONPATH" "$MLFLOW_TRACKING_URI"
    } > "$CHILD_LOG"

    # Lancement nohup : stdin /dev/null, stdout+stderr → fichier (append)
    nohup "${PYTHON_CMD[@]}" >> "$CHILD_LOG" 2>&1 < /dev/null &
    PID=$!
    disown

    echo "Lancé en background :"
    echo "  PID      : $PID"
    echo "  Log      : $CHILD_LOG"
    echo "  Suivre   : tail -f $CHILD_LOG"
    echo "  Killer   : kill $PID"

    if [[ "$WATCH_MODE" -eq 1 ]]; then
        echo ""
        echo "[watch] tail -f démarré (Ctrl-C pour arrêter le tail, pas le process)"
        sleep 2
        tail -f "$CHILD_LOG"
    fi
else
    # Foreground : tout passe par le log + terminal via tee (init_logging)
    asp::log "cmd: ${PYTHON_CMD[*]}"
    "${PYTHON_CMD[@]}"
fi
