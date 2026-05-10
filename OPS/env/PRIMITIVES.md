# PRIMITIVES.md — Validation Stage 0.2 sur RTX 5090

**Statut** : ✅ exécuté le 2026-05-10 sur le pod LUMIS-SFT-3 (RTX 5090, sm_120). 6/6 checks passés.

## Procédure

```bash
# Sur le pod, après bash OPS/scripts/setup_env.sh :
PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py
```

Le script produit :
- Une table Markdown sur `stdout` (à coller dans la section ci-dessous)
- `OPS/env/primitives_results.json` (machine-readable, **gitignored**)

## Critères de passage Stage 0.2

| Check | Critère minimal |
|---|---|
| `svd_fp64_batched` | passed, `max_recon_err < 1e-10` |
| `svd_lowrank_randomized` | passed sans erreur |
| `fft_rfft_long` | passed, `max_inverse_err < 1e-5` |
| `lstsq_batched` | passed sans erreur |
| `vmap` | passed sans erreur |
| `dense_attention_export` | passed, `A_fp64_bytes_mb` cohérent avec attendu |

**Si un check échoue** : documenter l'erreur dans la section "Régressions" et basculer sur un fallback (cf. STACK.md § Évolutions prévisibles).

## Système (post-run, 2026-05-10)

```
- python : 3.12.3
- torch : 2.11.0+cu128
- cuda_available : True
- cuda_version : 12.8
- device_name : NVIDIA GeForce RTX 5090
- device_capability : (12, 0)
- device_vram_gb : 31.367
- platform : Linux-6.11.0-26-generic-x86_64-with-glibc2.39
- is_blackwell : True
```

## Résultats (sortie validate_primitives.py)

| Check | Statut | Durée | Détails |
|---|---|---|---|
| `svd_fp64_batched` | ✅ | 1.980s | batch=8, N=256, max_recon_err=2.5e-14, S_dtype=float64 |
| `svd_lowrank_randomized` | ✅ | 0.318s | K=128, q=16, S_top=1.006, S_bot=0.153 |
| `fft_rfft_long` | ✅ | 0.650s | N=8192, max_inverse_err=1.4e-6 |
| `lstsq_batched` | ✅ | 0.214s | batch=16, M=64, N=32 |
| `vmap` | ✅ | 0.104s | input_shape=(32,16,16), output_shape=(32,) |
| `dense_attention_export` | ✅ | 0.065s | B=4, H=8, N=256, d=64, A_bytes_mb=8.0, A_fp64_bytes_mb=16.0 |

JSON complet sur le pod : `/workspace/polymorphic-attention/OPS/env/primitives_results.json`.

## Verdict Stage 0.2

- [x] 6/6 checks passés sur RTX 5090
- [x] Pas d'erreur Blackwell (sm_120) sur les builds CUDA des wheels installés
- [x] `A_fp64_bytes_mb` cohérent (16 MB pour N=256, B=4, H=8 → extrapolable à N=2¹⁶ avec slicing par batch)

✅ Stage 0.2 final passé. Sprint 1 phase 1 ouvrable.

## Régressions / surprises (post-run)

- ⚠️ Warning ENV vars Blackwell manquantes lors de l'exécution dans une session SSH non-héritée
  (TORCH_CUDA_ARCH_LIST, PYTORCH_CUDA_ALLOC_CONF, CUDA_MODULE_LOADING, MAX_JOBS).
  Sans impact sur les 6 checks (passent quand même), mais nécessaire pour les compilations
  CUDA en runtime. À régler en sourçant un fichier env avant chaque run sur le pod
  (ou en ajoutant à `~/.bashrc`).
