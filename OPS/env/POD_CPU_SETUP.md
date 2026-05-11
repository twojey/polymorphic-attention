# POD_CPU_SETUP.md — pod RunPod CPU pour phase 1.5 (sans GPU)

Procédure pour lancer phase 1.5 sur un pod CPU éphémère (vs SETUP.md qui cible le pod RTX 5090).

> **Pourquoi un pod CPU plutôt que le VPS** : phase 1.5 = compute eigvalsh batché parallélisable. Le VPS a 4 vCPUs ; un pod CPU 32 vCPUs donne ~10× speedup sur S_Spectral pour ~$0.30 le run 2000-ex. Bonus : RAM ≥ 32 GB permet de relancer le bench original (Δ≤256) au lieu du bench réduit (Δ≤64) imposé par la contrainte VPS.

---

## TL;DR — 7 commandes

```bash
# === Côté VPS (préparation) ===
bash OPS/scripts/start_mlflow_server.sh   # tmux conseillé, voir SETUP.md §1.1

# === Louer un pod RunPod ===
# Choisir un template "Ubuntu 22.04 + Python 3.11+" (pas besoin de CUDA).
# Recommandé : 32 vCPUs / 64 GB RAM (~$0.20-0.30/h).
# Récupérer POD_IP et POD_PORT (SSH).

# === Sync code + oracle checkpoint depuis VPS ===
POD_IP=xx.xx.xx.xx POD_PORT=NNNNN
rsync -e "ssh -p $POD_PORT -o StrictHostKeyChecking=no" -avz \
  --exclude '.venv' --exclude 'OPS/logs' --exclude '__pycache__' \
  /root/polymorphic-attention/ root@$POD_IP:/workspace/polymorphic-attention/

ORACLE_REL='OPS/logs/mlflow/artifacts/2/7e1e859818b64b85b6c2c88433767c48/artifacts/oracle/s1_smnist_oracle_e2f0b5e_oracle.ckpt'
ssh -p $POD_PORT root@$POD_IP "mkdir -p /workspace/polymorphic-attention/$(dirname $ORACLE_REL)"
rsync -e "ssh -p $POD_PORT" -avz "/root/polymorphic-attention/$ORACLE_REL" \
  "root@$POD_IP:/workspace/polymorphic-attention/$ORACLE_REL"

# === Setup pod (idempotent) ===
ssh -p $POD_PORT root@$POD_IP \
  "cd /workspace/polymorphic-attention && bash OPS/scripts/setup_pod_cpu.sh"

# === Tunnel MLflow inverse VPS → pod ===
POD_IP=$POD_IP POD_PORT=$POD_PORT bash OPS/scripts/start_mlflow_tunnel.sh

# === Lancer phase 1.5 sur le pod (bench original Δ≤256) ===
ssh -p $POD_PORT root@$POD_IP "cd /workspace/polymorphic-attention && \
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  ./OPS/env/launch_phase1b.sh --nohup -- bench.n_examples=2000 s_kl.enabled=false"
```

---

## Notes importantes

### Bench original vs bench réduit
- Sur VPS (9 GB RAM) : bench réduit obligatoire (`structured_deltas: [16, 64]`, seq max 1043).
- Sur pod 64 GB : bench **original** OK (`structured_deltas: [16, 64, 128, 256]`, seq max ~4100). Pas d'override Hydra à passer — c'est la config par défaut dans `OPS/configs/phase1b/signals.yaml` si tu n'as pas modifié le fichier ; vérifier `git diff` côté pod si doute.

### Garde-fou BLAS (toujours actif)
Le code `compute_s_spectral()` raise `RuntimeError` si `OPENBLAS_NUM_THREADS != 1` (cf. carnet 2026-05-11 deadlock). `launch_phase1b.sh` exporte les 4 vars (OPENBLAS, MKL, OMP, NUMEXPR). Ne pas bypasser.

### Parallélisation
`compute_s_spectral()` utilise `multiprocessing.Pool` avec `os.cpu_count() - 1` workers (chacun BLAS=1). Sur 32 vCPUs → 31 workers. Speedup attendu vs séquentiel : **~10-15×** (limité par bande passante mémoire après ~16 workers).

Pas de variable d'environnement à passer pour activer — c'est le mode par défaut.

### Estimation wall-clock
| Config | VPS 4 vCPUs | Pod 32 vCPUs |
|---|---|---|
| 2000-ex / Δ≤64 | ~6-12h | ~30-60 min |
| 2000-ex / Δ≤256 | impossible (OOM) | ~1.5-3h |

### Suivi du run
Depuis le VPS, si tunnel MLflow up :
```bash
# UI MLflow accessible sur le VPS
curl http://127.0.0.1:5000/api/2.0/mlflow/experiments/search -X POST -d '{}' -H "Content-Type: application/json" | jq

# Logs live du run pod
ssh -p $POD_PORT root@$POD_IP "tail -f /tmp/phase1b_*.log | head -200"
```

### Avant de supprimer le pod
1. Vérifier que le run est terminé (status MLflow = FINISHED, pas RUNNING).
2. Vérifier que les artefacts MLflow sont bien sur le VPS (`ls /root/polymorphic-attention/OPS/logs/mlflow/artifacts/3/<run_id>/`).
3. Si OK → fermer le tunnel + delete pod. Pas besoin de rsync retour : tout passe par MLflow.

---

## Différences avec SETUP.md (pod GPU RTX 5090)

| Étape | Pod GPU | Pod CPU |
|---|---|---|
| Source `blackwell_env.sh` | Oui (sm_120 ENV vars) | **Non** (pas de CUDA) |
| `install_python_deps.sh` | Auto-détecte CUDA via nvidia-smi | Force `ASP_CUDA=0` |
| `verify_torch.sh` | Vérifie CUDA capability | Smoke eigvalsh CPU dans `setup_pod_cpu.sh` |
| Phase à lancer | phase 1 (oracle) ou phase 2+ (GPU obligatoire) | phase 1.5 uniquement (eigvalsh CPU, pas de gradient) |
| Coût/h | ~$0.50-1.50 (RTX 5090) | ~$0.10-0.30 (32 vCPUs) |
