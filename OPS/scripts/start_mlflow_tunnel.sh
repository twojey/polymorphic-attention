#!/usr/bin/env bash
# start_mlflow_tunnel.sh — ouvre un tunnel SSH inverse VPS → pod
#
# **À lancer depuis le VPS** (où MLflow tourne sur 127.0.0.1:5000).
# Le tunnel rend MLflow accessible depuis le pod sur localhost:5000 :
#
#     pod $ curl http://localhost:5000   →   VPS:5000 (MLflow)
#
# Avantage vs forward tunnel pod→VPS : pas besoin de configurer une clé
# SSH du pod vers le VPS, on utilise la clé du VPS qui peut déjà
# atteindre le pod.
#
# Usage :
#   POD_IP=149.36.1.181 POD_PORT=16279 bash OPS/scripts/start_mlflow_tunnel.sh
#
# Variables d'environnement :
#   POD_IP     (requis)         IP publique du pod RunPod
#   POD_PORT   (requis)         port SSH exposé du pod
#   POD_USER   (défaut: root)   user SSH du pod
#   SSH_KEY    (défaut: ~/.ssh/id_ed25519)
#   MLFLOW_PORT (défaut: 5000)
#
# Le tunnel s'ouvre en background (-N -f). Pour le fermer :
#   pgrep -af "ssh -N -f.*${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT}" | head -1 | awk '{print $1}' | xargs kill

set -euo pipefail

POD_IP="${POD_IP:?Variable POD_IP requise (ex: 149.36.1.181)}"
POD_PORT="${POD_PORT:?Variable POD_PORT requise (ex: 16279)}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERREUR : clé SSH introuvable : $SSH_KEY" >&2
  exit 1
fi

# Vérifier que MLflow tourne côté VPS
if ! curl -s -o /dev/null --max-time 3 "http://127.0.0.1:${MLFLOW_PORT}/"; then
  echo "AVERTISSEMENT : MLflow ne répond pas sur 127.0.0.1:${MLFLOW_PORT}." >&2
  echo "                Lancer 'bash OPS/scripts/start_mlflow_server.sh' d'abord." >&2
fi

# Tuer un éventuel tunnel existant pour ce port
EXISTING=$(pgrep -af "ssh -N -f.*-R ${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT}" | awk '{print $1}' | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "==> Tunnel existant détecté (PID $EXISTING), je le ferme."
  kill "$EXISTING" 2>/dev/null || true
  sleep 1
fi

echo "==> Ouverture tunnel inverse VPS → ${POD_USER}@${POD_IP}:${POD_PORT}"
echo "    MLflow ${MLFLOW_PORT}/VPS  →  pod localhost:${MLFLOW_PORT}"

ssh -N -f \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=no \
  -R "${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT}" \
  -p "${POD_PORT}" -i "${SSH_KEY}" \
  "${POD_USER}@${POD_IP}"

# Vérification
sleep 1
NEW_PID=$(pgrep -af "ssh -N -f.*-R ${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT}" | awk '{print $1}' | head -1 || true)
if [[ -z "$NEW_PID" ]]; then
  echo "ERREUR : tunnel non détecté après lancement." >&2
  exit 1
fi

echo "==> Tunnel actif (PID $NEW_PID)"
echo "    Pour fermer : kill $NEW_PID"
echo "    Sur le pod : export MLFLOW_TRACKING_URI=http://localhost:${MLFLOW_PORT}"
