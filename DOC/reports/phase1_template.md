# Rapport phase 1 — métrologie RCP

> Spec : [DOC/01_phase_metrologie.md](../01_phase_metrologie.md). Driver : `phase1_metrologie.run`.

## Métadonnées

- **Run ID** : `<run_id>`
- **Git hash** : `<short_hash>` (commit complet : `<full_hash>`)
- **Sprint** : `<N>`
- **Domaine** : `smnist | code | vision`
- **Date** : `YYYY-MM-DD`
- **Hardware** : `<gpu, vram, cuda>`
- **Status** : `registered | exploratory`

## Verdict go/no-go (DOC/01 §6)

**Décision** : `GO | NO-GO`

**Raison** : `<reason from phase1_metrologie.report.evaluate_go_no_go>`

## Recommandation R_max préliminaire

`R_max ≈ <r_max>` — heuristique 1.5 × max(p90 rang Hankel par régime).
À raffiner par phase 2 sur la base de r_eff.

## Statistiques par régime

### Rang de Hankel (monovariate)
*(table autogénérée — voir `phase1_metrologie.report.render_markdown_report`)*

### Entropie spectrale (monovariate)

### Précision Oracle par régime

## Figures

- `<run_id>_hankel_omega.png`
- `<run_id>_hankel_delta.png`
- `<run_id>_hankel_entropy.png`
- `<run_id>_entropy_omega.png`
- `<run_id>_entropy_delta.png`
- `<run_id>_entropy_entropy.png`
- `<run_id>_hankel_2d_omega_delta.png`
- `<run_id>_entropy_2d_omega_delta.png`

## Lessons learned (manuel)

> _À remplir après revue. Ce qui a surpris, ce qui a confirmé, ce qui doit
> être changé phase 2._
