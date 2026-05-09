# PRIMITIVES.md — Validation Stage 0.2 sur RTX 5090

**Statut** : ⏸ en attente d'exécution sur le pod RunPod RTX 5090.

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

## Système (à remplir post-run)

```
- python : ...
- torch : ...
- cuda_available : ...
- cuda_version : ...
- device_name : NVIDIA GeForce RTX 5090
- device_capability : ...
- device_vram_gb : ...
- platform : ...
```

## Résultats (à coller depuis stdout)

> _Coller ici la sortie du script `validate_primitives.py` exécuté sur le pod._

```
[à remplir]
```

## Verdict Stage 0.2

- [ ] 6/6 checks passés sur RTX 5090
- [ ] Pas d'erreur Blackwell (sm_120) sur les builds CUDA des wheels installés
- [ ] `A_fp64_bytes_mb` extrapolable à `N=2¹²` et `N=2¹⁶` sans dépassement VRAM (cf. HARDWARE.md § Phase 2)

Si tout est ✅ → cocher Stage 0.2 dans `ROADMAP.md` et ouvrir Sprint 1 (phase 1 sur Structure-MNIST).

## Régressions / surprises (post-run)

> _Documenter ici tout comportement inattendu : warning Blackwell, performance dégradée, primitive manquante, etc._
