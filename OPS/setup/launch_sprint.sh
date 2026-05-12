#!/usr/bin/env bash
# launch_sprint.sh — lancer un Sprint avec safeguards + logs persistants.
#
# Wrappe `python -m sprints.run --sprint <ID>` avec :
# - lib/common.sh (strict mode, logs horodatés OPS/logs/sprints/, traps)
# - env BLAS=1 (anti-deadlock), PYTHONUNBUFFERED, MLflow URI
# - mode --nohup pour pod détaché
# - propagation ASP_LOG_FILE pour cohérence shell/Python
#
# Usage :
#   bash OPS/setup/launch_sprint.sh <SPRINT_ID> [OPTIONS] -- [SPRINT_ARGS]
#
# Sprints supportés : B, C, D, E, F, G, S4, S5, S6, S7.
#
# Options :
#   --nohup              Lancer en background (nohup + disown)
#   --watch              Afficher logs en temps réel (tail -f) — avec nohup
#   --output DIR         Override output_dir (défaut : OPS/logs/sprints/<ID>/)
#   --mlflow-uri URI     MLflow tracking URI (défaut : $MLFLOW_TRACKING_URI ou null)
#   --seed N             Seed (défaut 0)
#   --device DEV         cpu | cuda (défaut cpu)
#   --help               Afficher cette aide
#
# Exemples :
#   bash OPS/setup/launch_sprint.sh B -- --oracle-checkpoint OPS/checkpoints/oracle_smnist.pt
#   bash OPS/setup/launch_sprint.sh C --nohup -- --dumps-dir OPS/logs/sprints/B_re_extract/dumps
#   bash OPS/setup/launch_sprint.sh S5 --device cuda --nohup --watch -- --use-hf-dinov2
#
# Codes de sortie :
#   0   succès complet
#   1   parsing options KO
#   2   pré-condition manquante
#   N   exit code Python (propagation)

# shellcheck source=../scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/lib/common.sh"
asp::init_strict_mode

# --- Parse options ---

SPRINT_ID=""
NOHUP_MODE=0
WATCH_MODE=0
OUTPUT_DIR=""
MLFLOW_URI=""
SEED=0
DEVICE="cpu"

if [[ $# -lt 1 ]]; then
    asp::die 1 "Sprint ID requis. Usage: bash $0 <SPRINT_ID> [OPTIONS] -- [SPRINT_ARGS]"
fi

# Handle --help avant la validation Sprint ID
case "$1" in
    --help|-h|help)
        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 0
        ;;
esac

SPRINT_ID="$1"
shift

# Valider sprint_id
case "$SPRINT_ID" in
    B|C|D|E|F|G|S4|S5|S6|S7) ;;
    *) asp::die 1 "Sprint inconnu : $SPRINT_ID (attendu : B|C|D|E|F|G|S4|S5|S6|S7)" ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup) NOHUP_MODE=1; shift ;;
        --watch) WATCH_MODE=1; shift ;;
        --output) OUTPUT_DIR="${2:?--output requiert un argument}"; shift 2 ;;
        --mlflow-uri) MLFLOW_URI="${2:?}"; shift 2 ;;
        --seed) SEED="${2:?}"; shift 2 ;;
        --device) DEVICE="${2:?}"; shift 2 ;;
        --help|-h)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        --) shift; break ;;
        -*) asp::die 1 "option inconnue : $1 (--help)" ;;
        *) asp::die 1 "argument inattendu : $1 (utiliser -- avant les Sprint args)" ;;
    esac
done

SPRINT_ARGS=("$@")

# --- Init logging ---

ASP_REPO_ROOT="$(asp::_resolve_repo_root)"
LOG_PREFIX="sprint_${SPRINT_ID}"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$ASP_REPO_ROOT/OPS/logs/sprints/${SPRINT_ID}"
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$NOHUP_MODE" -eq 0 ]]; then
    asp::init_logging "$LOG_PREFIX"
    asp::log_env
fi

# --- Pré-conditions ---

asp::section "Pré-conditions"
asp::require_cmd uv
[[ -d "$ASP_REPO_ROOT/CODE" ]] || asp::die 2 "CODE/ introuvable dans $ASP_REPO_ROOT"
asp::log "sprint_id=$SPRINT_ID device=$DEVICE seed=$SEED output=$OUTPUT_DIR"

# --- Env vars ---

asp::section "Env vars Sprint"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ASP_REPO_ROOT/CODE${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$MLFLOW_URI" ]]; then
    export MLFLOW_TRACKING_URI="$MLFLOW_URI"
fi
export ASP_LOG_FILE ASP_LOG_PREFIX

asp::log "PYTHONPATH=$PYTHONPATH"
asp::log "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-<unset>}"

# --- Exécution ---

asp::section "Lancement Sprint $SPRINT_ID"

cd "$ASP_REPO_ROOT"

PYTHON_CMD=(
    uv run python -u -m sprints.run
    --sprint "$SPRINT_ID"
    --output "$OUTPUT_DIR"
    --seed "$SEED"
    --device "$DEVICE"
    "${SPRINT_ARGS[@]}"
)
if [[ -n "$MLFLOW_URI" ]]; then
    PYTHON_CMD+=(--mlflow-uri "$MLFLOW_URI")
fi

if [[ "$NOHUP_MODE" -eq 1 ]]; then
    ASP_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
    CHILD_LOG="$ASP_REPO_ROOT/OPS/logs/sprints/${LOG_PREFIX}_${ASP_TS}.log"
    mkdir -p "$(dirname "$CHILD_LOG")"
    {
        printf '[%s] [init] log=%s sprint=%s (background)\n' "$(asp::_ts)" "$CHILD_LOG" "$SPRINT_ID"
        printf '[%s] commit=%s\n' "$(asp::_ts)" "$(git -C "$ASP_REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo n/a)"
        printf '[%s] cmd=%s\n' "$(asp::_ts)" "${PYTHON_CMD[*]}"
    } > "$CHILD_LOG"
    nohup "${PYTHON_CMD[@]}" >> "$CHILD_LOG" 2>&1 < /dev/null &
    PID=$!
    disown
    echo "Sprint $SPRINT_ID lancé :"
    echo "  PID      : $PID"
    echo "  Log      : $CHILD_LOG"
    echo "  Suivre   : tail -f $CHILD_LOG"
    echo "  Killer   : kill $PID"
    if [[ "$WATCH_MODE" -eq 1 ]]; then
        sleep 2
        tail -f "$CHILD_LOG"
    fi
else
    asp::log "cmd: ${PYTHON_CMD[*]}"
    "${PYTHON_CMD[@]}"
fi
