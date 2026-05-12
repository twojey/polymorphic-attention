#!/usr/bin/env bash
# launch_extract.sh — Lancer phase 1 V2 extract (run_extract.py) avec safeguards
# BLAS + logs persistants + tunnel MLflow + checkpoint Oracle.
#
# Reprise du pattern launch_phase1b.sh (carnet 2026-05-11 refactor scripts robustes).
#
# Usage:
#   bash OPS/env/launch_extract.sh [OPTIONS] -- [HYDRA_OVERRIDES]
#
# Options:
#   --nohup              Lancer en background (nohup + disown)
#   --watch              Afficher les logs en temps réel (tail -f)
#   --checkpoint PATH    Override le path checkpoint Oracle (sinon valeur par défaut)
#   --output-dir PATH    Override extraction.output_dir (défaut : /workspace/phase1_extract)
#   --log-prefix PFX     Préfixe du log (défaut : "extract")
#   --help               Afficher cette aide
#
# Exemples:
#   bash OPS/env/launch_extract.sh --nohup --watch
#   bash OPS/env/launch_extract.sh -- extraction.batch_size_cap=4 extraction.target_peak_gb=10
#
# Codes de sortie :
#   0   succès complet
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
LOG_PREFIX="extract"
CHECKPOINT_PATH_DEFAULT="OPS/checkpoints/oracle_e2f0b5e.ckpt"
CHECKPOINT_PATH=""
OUTPUT_DIR_DEFAULT="/workspace/phase1_extract"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup)        NOHUP_MODE=1; shift ;;
        --watch)        WATCH_MODE=1; shift ;;
        --checkpoint)   CHECKPOINT_PATH="${2:?--checkpoint requiert un argument}"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="${2:?--output-dir requiert un argument}"; shift 2 ;;
        --log-prefix)   LOG_PREFIX="${2:?--log-prefix requiert un argument}"; shift 2 ;;
        --help|-h)      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        --)             shift; break ;;
        -*)             asp::die 1 "option inconnue : $1 (--help pour usage)" ;;
        *)              asp::die 1 "argument inattendu : $1 (utiliser -- avant les overrides Hydra)" ;;
    esac
done

HYDRA_OVERRIDES=("$@")
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_PATH_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

# --- Init logging -----------------------------------------------------------

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
mkdir -p "$OUTPUT_DIR"
asp::log "checkpoint  : $CHECKPOINT_PATH"
asp::log "output_dir  : $OUTPUT_DIR"
asp::log "hydra_overrides : ${HYDRA_OVERRIDES[*]:-<aucun>}"

# Vérifier que git est clean → status=registered (sinon warning)
DIRTY_COUNT="$(cd "$ASP_REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l)"
if [[ "$DIRTY_COUNT" -gt 0 ]]; then
    asp::log "[WARN] git working tree DIRTY ($DIRTY_COUNT fichiers) → manifest aura status=exploratory."
    asp::log "[WARN] Pour status=registered : git stash ou commit avant relance."
fi

# --- Env vars ---------------------------------------------------------------

asp::section "Env vars phase 1 extract"

# Anti-deadlock BLAS (cf. carnet 2026-05-11 deadlock 38h, garde-fou compute_s_spectral)
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

export ASP_LOG_FILE
export ASP_LOG_PREFIX

asp::log "BLAS=1 PYTHONUNBUFFERED=1 MLFLOW=$MLFLOW_TRACKING_URI"

# --- Exécution --------------------------------------------------------------

asp::section "Lancement run_extract.py"

cd "$ASP_REPO_ROOT"

PYTHON_CMD=(
    uv run python -u -m phase1_metrologie.run_extract
    --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist
    "oracle_checkpoint=$CHECKPOINT_PATH"
    "extraction.output_dir=$OUTPUT_DIR"
    "${HYDRA_OVERRIDES[@]}"
)

if [[ "$NOHUP_MODE" -eq 1 ]]; then
    ASP_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
    CHILD_LOG="$ASP_REPO_ROOT/OPS/logs/${LOG_PREFIX}_${ASP_TS}.log"
    mkdir -p "$(dirname "$CHILD_LOG")"

    {
        printf '[%s] [init] log=%s pid=$$ (background)\n' "$(asp::_ts)" "$CHILD_LOG"
        printf '[%s] commit=%s\n' "$(asp::_ts)" "$(git -C "$ASP_REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo n/a)"
        printf '[%s] cmd=%s\n' "$(asp::_ts)" "${PYTHON_CMD[*]}"
        printf '[%s] BLAS=1 PYTHONPATH=%s MLFLOW=%s OUTPUT_DIR=%s\n' "$(asp::_ts)" "$PYTHONPATH" "$MLFLOW_TRACKING_URI" "$OUTPUT_DIR"
    } > "$CHILD_LOG"

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
    asp::log "cmd: ${PYTHON_CMD[*]}"
    "${PYTHON_CMD[@]}"
fi
