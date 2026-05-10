#!/usr/bin/env bash
# monitor_run.sh — état du dernier run MLflow + processus pod + GPU
#
# Usage : bash OPS/scripts/monitor_run.sh [experiment_id]
#         (experiment_id défaut = 2 = phase 1)
#
# Variables d'environnement :
#   MLFLOW_URL    (défaut: http://127.0.0.1:5000)
#   POD_IP        (défaut: 149.36.1.181)
#   POD_PORT      (défaut: 16279)
#   POD_USER      (défaut: root)
#   SSH_KEY       (défaut: ~/.ssh/id_ed25519)
#   POD_PROC_RE   (défaut: phase1_metrologie\\.run)
#
# Sortie : bloc lisible humain. Codes de sortie :
#   0  : run RUNNING normal
#   1  : aucune donnée trouvée
#   2  : run FINISHED (succès)
#   3  : run FAILED ou process Dead alors que MLflow dit RUNNING

set -euo pipefail

EXPERIMENT_ID="${1:-2}"
MLFLOW_URL="${MLFLOW_URL:-http://127.0.0.1:5000}"
POD_IP="${POD_IP:-149.36.1.181}"
POD_PORT="${POD_PORT:-16279}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
POD_PROC_RE="${POD_PROC_RE:-phase1_metrologie\\.run}"

# ---------------------------------------------------------------
# 1. Heure Paris
# ---------------------------------------------------------------
PARIS_TIME="$(TZ=Europe/Paris date +'%H:%M')"

# ---------------------------------------------------------------
# 2. MLflow : dernier run sur l'expérience
# ---------------------------------------------------------------
MLFLOW_PAYLOAD=$(curl -s --max-time 5 \
  "${MLFLOW_URL}/api/2.0/mlflow/runs/search" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"experiment_ids\":[\"${EXPERIMENT_ID}\"],\"max_results\":1,\"order_by\":[\"attributes.start_time DESC\"]}" || echo '{"runs":[]}')

# Parse via Python pour robustesse (jq pas garanti). JSON passé en env var
# pour ne pas confondre avec le script Python lui-même sur stdin.
PARSED=$(MLFLOW_DATA="$MLFLOW_PAYLOAD" python3 - <<'PY'
import json, os, sys, time
data_raw = os.environ.get("MLFLOW_DATA", "")
try:
    d = json.loads(data_raw)
except Exception as e:
    print("STATE=PARSE_ERROR")
    sys.stderr.write(f"JSON parse failed: {e}\n")
    sys.exit(0)
runs = d.get("runs", [])
if not runs:
    print("STATE=NORUNS")
    sys.exit(0)
r = runs[0]
i = r["info"]
m = r["data"].get("metrics", [])
print(f"RUN_NAME={i.get('run_name','?')}")
print(f"STATE={i.get('status','?')}")
start_ms = i.get("start_time", 0) or 0
print(f"START_MS={start_ms}")
end_ms = i.get("end_time")
if end_ms:
    print(f"END_MS={end_ms}")
    print(f"DURATION_S={(end_ms - start_ms) // 1000}")
else:
    print(f"DURATION_S={int(time.time() - start_ms / 1000)}")

def latest(key):
    items = sorted([x for x in m if x["key"] == key], key=lambda x: x["step"])
    return items[-1] if items else None

for k in ("train_loss", "val_loss", "val_acc"):
    v = latest(k)
    if v:
        print(f"{k.upper()}_STEP={v['step']}")
        print(f"{k.upper()}_VAL={v['value']:.4f}")

hk = [x for x in m if x["key"].startswith("hankel_rank_layer")]
sp = [x for x in m if x["key"].startswith("spectral_entropy_layer")]
print(f"HANKEL_N={len(hk)}")
print(f"SPECTRAL_N={len(sp)}")
PY
)

# Charge les variables PARSED dans le shell
eval "$PARSED"

if [[ "${STATE:-}" == "NORUNS" ]]; then
    echo "Heure Paris : ${PARIS_TIME}"
    echo "Aucun run MLflow trouvé sur experiment ${EXPERIMENT_ID}."
    exit 1
fi

# ---------------------------------------------------------------
# 3. Pod : processus + GPU
# ---------------------------------------------------------------
POD_OUT=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no \
    -p "${POD_PORT}" -i "${SSH_KEY}" "${POD_USER}@${POD_IP}" \
    "pgrep -f '${POD_PROC_RE}' >/dev/null && echo ALIVE || echo DEAD; \
     nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader" \
    2>&1 || echo "POD_UNREACHABLE")

POD_ALIVE=$(echo "$POD_OUT" | head -1)
POD_GPU=$(echo "$POD_OUT" | sed -n '2p')

# ---------------------------------------------------------------
# 4. Format du rapport
# ---------------------------------------------------------------
DURATION_MIN=$(( ${DURATION_S:-0} / 60 ))
DURATION_SEC=$(( ${DURATION_S:-0} % 60 ))

echo "Heure Paris : ${PARIS_TIME} · run elapsed ≈ ${DURATION_MIN} min ${DURATION_SEC} s"
echo "Run         : ${RUN_NAME:-?}"
echo "Status      : ${STATE:-?}"
if [[ -n "${TRAIN_LOSS_VAL:-}" ]]; then
    echo "Epoch       : ${TRAIN_LOSS_STEP:-?}"
    echo "  train_loss : ${TRAIN_LOSS_VAL}"
    echo "  val_loss   : ${VAL_LOSS_VAL:-?}"
    echo "  val_acc    : ${VAL_ACC_VAL:-?}"
else
    echo "Epoch       : aucune métrique encore"
fi
echo "Hankel/Spec : ${HANKEL_N:-0} / ${SPECTRAL_N:-0} (post-extraction)"
echo "Pod process : ${POD_ALIVE}"
echo "GPU         : ${POD_GPU:-?}"

# ---------------------------------------------------------------
# 5. Verdict + code retour
# ---------------------------------------------------------------
echo
case "${STATE:-?}" in
    RUNNING)
        if [[ "${POD_ALIVE:-?}" == "DEAD" ]]; then
            echo "Verdict     : ÉCHEC (process pod mort alors que MLflow dit RUNNING)"
            exit 3
        fi
        echo "Verdict     : continue"
        exit 0
        ;;
    FINISHED)
        if [[ "${HANKEL_N:-0}" -gt 0 ]] && [[ "${SPECTRAL_N:-0}" -gt 0 ]]; then
            echo "Verdict     : TERMINÉ avec succès (Hankel + spectral présents)"
        else
            echo "Verdict     : TERMINÉ (mais métriques d'extraction absentes — à investiguer)"
        fi
        exit 2
        ;;
    FAILED)
        echo "Verdict     : ÉCHEC (status MLflow = FAILED)"
        exit 3
        ;;
    *)
        echo "Verdict     : ÉTAT INCONNU (${STATE:-?})"
        exit 3
        ;;
esac
