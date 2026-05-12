# OPS/setup/ — Bootstrap machines

Scripts et docs pour **passer d'une machine vide à un environnement projet utilisable**. Distinct de `OPS/env/` qui décrit l'environnement runtime (variables, conventions, stack).

## Contenu

### Documentation (procédure step-by-step)

- [SETUP.md](SETUP.md) — Bootstrap pod RunPod RTX 5090 (Blackwell, sm_120, contraintes FP64)
- [POD_CPU_SETUP.md](POD_CPU_SETUP.md) — Bootstrap pod RunPod CPU (alternative pour phases qui ne nécessitent pas GPU, ex: phase 1.5 BLAS eigvalsh)

### Scripts de launching (lance une phase complète)

- `launch_extract.sh` — phase 1 extract sur pod
- `launch_phase1b.sh` — phase 1.5 calibration signaux
- `launch_phase2.sh` — phase 2 audit spectral
- `launch_phase3.sh` — phase 3 training ASPLayer

Note : les scripts d'**installation pure** (uv, deps Python, torch+cuda) restent dans
`OPS/scripts/` (install_uv.sh, install_python_deps.sh, setup_pod.sh, etc.) — ils sont
appelés par les procédures de ce dossier.

## Quand utiliser quoi

| Besoin | Aller à |
|---|---|
| Démarrer un nouveau pod RunPod RTX 5090 | [SETUP.md](SETUP.md) |
| Démarrer un pod CPU pour phase 1.5 ou eigvalsh-heavy | [POD_CPU_SETUP.md](POD_CPU_SETUP.md) |
| Lancer une phase déjà installée | `launch_phaseN.sh` du dossier courant |
| Comprendre la stack (PyTorch, BLAS, MLflow) | [OPS/env/STACK.md](../env/STACK.md) |
| Variables d'environnement Blackwell | [OPS/env/HARDWARE.md](../env/HARDWARE.md) |
| Conventions MLflow runs | [OPS/env/LOGGING.md](../env/LOGGING.md) |
