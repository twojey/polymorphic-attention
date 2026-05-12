#!/usr/bin/env bash
# launch_phase3.sh — Lancer phase 3 (ASPLayer training) avec safeguards.
#
# Reprise du pattern launch_phase1b.sh / launch_phase2.sh.
#
# Usage:
#   bash OPS/setup/launch_phase3.sh [OPTIONS] -- [HYDRA_OVERRIDES]
#
# Options:
#   --nohup              Lancer en background (nohup + disown)
#   --watch              tail -f le log
#   --checkpoint PATH    Oracle checkpoint (défaut : OPS/checkpoints/oracle_e2f0b5e.ckpt)
#   --dump-dir PATH      Dossier des dumps phase 2 (pour Smart Init optionnel)
#   --backbone CLASS     Override backbone.class_name (toeplitz|hankel|identity|linear|composite)
#   --smart-init         Active init_strategy=smart (requiert --dump-dir)
#   --smoke              max_epochs=1 batch_size=8 (smoke check rapide ~5 min)
#   --log-prefix PFX     Préfixe log (défaut : "phase3")
#   --help               Aide
#
# Exemples:
#   bash OPS/setup/launch_phase3.sh --smoke --backbone identity
#   bash OPS/setup/launch_phase3.sh --nohup --watch --backbone toeplitz \
#       --dump-dir /workspace/phase1_extract --smart-init
#
# Logs : OPS/logs/<prefix>_<UTC_TS>.log

# shellcheck source=../scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/lib/common.sh"
asp::init_strict_mode

NOHUP_MODE=0
WATCH_MODE=0
SMOKE_MODE=0
SMART_INIT=0
LOG_PREFIX="phase3"
CHECKPOINT_PATH_DEFAULT="OPS/checkpoints/oracle_e2f0b5e.ckpt"
CHECKPOINT_PATH=""
DUMP_DIR=""
BACKBONE_CLASS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup)        NOHUP_MODE=1; shift ;;
        --watch)        WATCH_MODE=1; shift ;;
        --smoke)        SMOKE_MODE=1; shift ;;
        --smart-init)   SMART_INIT=1; shift ;;
        --checkpoint)   CHECKPOINT_PATH="${2:?--checkpoint requiert un argument}"; shift 2 ;;
        --dump-dir)     DUMP_DIR="${2:?--dump-dir requiert un argument}"; shift 2 ;;
        --backbone)     BACKBONE_CLASS="${2:?--backbone requiert un argument}"; shift 2 ;;
        --log-prefix)   LOG_PREFIX="${2:?--log-prefix requiert un argument}"; shift 2 ;;
        --help|-h)      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        --)             shift; break ;;
        -*)             asp::die 1 "option inconnue : $1 (--help)" ;;
        *)              asp::die 1 "argument inattendu : $1 (utiliser -- avant les overrides Hydra)" ;;
    esac
done

HYDRA_OVERRIDES=("$@")
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_PATH_DEFAULT}"

ASP_REPO_ROOT="$(asp::_resolve_repo_root)"

if [[ "$NOHUP_MODE" -eq 0 ]]; then
    asp::init_logging "$LOG_PREFIX"
    asp::log_env
fi

asp::section "Pré-conditions"
asp::require_cmd uv
asp::require_file "$CHECKPOINT_PATH"
[[ -d "$ASP_REPO_ROOT/CODE" ]] || asp::die 2 "CODE/ introuvable"

DIRTY_COUNT="$(cd "$ASP_REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l)"
if [[ "$DIRTY_COUNT" -gt 0 ]]; then
    asp::log "[WARN] git working tree DIRTY ($DIRTY_COUNT fichiers) → manifest=exploratory."
fi

asp::log "checkpoint  : $CHECKPOINT_PATH"
asp::log "dump_dir    : ${DUMP_DIR:-<none, random Xavier init>}"
asp::log "backbone    : ${BACKBONE_CLASS:-<default identity>}"
asp::log "smart_init  : $SMART_INIT"
asp::log "smoke       : $SMOKE_MODE"

asp::section "Env vars phase 3"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export PYTHONPATH="$ASP_REPO_ROOT/CODE${PYTHONPATH:+:$PYTHONPATH}"

export ASP_LOG_FILE
export ASP_LOG_PREFIX

asp::log "BLAS=1 PYTHONUNBUFFERED=1 MLFLOW=$MLFLOW_TRACKING_URI"

asp::section "Overrides Hydra"

OVERRIDES_AUTO=()
OVERRIDES_AUTO+=("+oracle_checkpoint=$CHECKPOINT_PATH")
if [[ -n "$DUMP_DIR" ]]; then
    OVERRIDES_AUTO+=("+phase2_dump_dir=$DUMP_DIR")
fi
if [[ -n "$BACKBONE_CLASS" ]]; then
    OVERRIDES_AUTO+=("backbone.class_name=$BACKBONE_CLASS")
fi
if [[ "$SMART_INIT" -eq 1 ]]; then
    if [[ -z "$DUMP_DIR" ]]; then
        asp::die 1 "--smart-init requiert --dump-dir"
    fi
    OVERRIDES_AUTO+=("asp_layer.init_strategy=smart")
fi
if [[ "$SMOKE_MODE" -eq 1 ]]; then
    OVERRIDES_AUTO+=("training.max_epochs=1")
    OVERRIDES_AUTO+=("training.batch_size=8")
    OVERRIDES_AUTO+=("training.log_interval_steps=2")
    OVERRIDES_AUTO+=("loss.matriochka.n_rank_samples=2")
    OVERRIDES_AUTO+=("loss.consistency.n_samples=2")
    OVERRIDES_AUTO+=("ssg.n_examples_per_regime=128")  # ~10x moins de data
    asp::log "smoke mode : 1 epoch, batch=8, n_examples_per_regime=128"
fi

asp::log "auto  : ${OVERRIDES_AUTO[*]:-<aucun>}"
asp::log "user  : ${HYDRA_OVERRIDES[*]:-<aucun>}"

asp::section "Lancement phase3_kernel_asp.run_train"

cd "$ASP_REPO_ROOT"

PYTHON_CMD=(
    uv run python -u -m phase3_kernel_asp.run_train
    --config-path=../../OPS/configs/phase3 --config-name=asp_layer
    "${OVERRIDES_AUTO[@]}"
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
    } > "$CHILD_LOG"
    nohup "${PYTHON_CMD[@]}" >> "$CHILD_LOG" 2>&1 < /dev/null &
    PID=$!
    disown
    echo "Lancé en background :"
    echo "  PID    : $PID"
    echo "  Log    : $CHILD_LOG"
    echo "  Suivre : tail -f $CHILD_LOG"
    if [[ "$WATCH_MODE" -eq 1 ]]; then sleep 2; tail -f "$CHILD_LOG"; fi
else
    asp::log "cmd: ${PYTHON_CMD[*]}"
    "${PYTHON_CMD[@]}"
fi
