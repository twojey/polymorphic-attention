# blackwell_env.sh — variables d'environnement pour RTX 5090 / Blackwell sm_120
#
# **À sourcer**, pas à exécuter :
#     source OPS/scripts/blackwell_env.sh
#
# (Si exécuté en sous-shell les exports sont perdus.)
#
# Ces variables doivent être set AVANT toute compilation/import qui peut
# produire un kernel CUDA, sinon l'arch n'est pas ciblée correctement.
#
# Source de vérité : configuration validée par /root/lumis/OPS sur 5090
# réelle + STACK.md § Blackwell.

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"                                 # anti-freeze sur compilation parallèle
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"                 # bénin en single-GPU
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Flag de présence — permet aux autres scripts de vérifier que ces vars sont set
export ASP_BLACKWELL_ENV_LOADED=1

# Affichage uniquement si on n'est pas dans un script silencieux
if [[ -z "${ASP_BLACKWELL_ENV_QUIET:-}" ]]; then
  echo "==> Blackwell ENV chargé : TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"
fi
