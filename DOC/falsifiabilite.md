# Falsifiabilité — critères go/no-go par phase

Tableau de synthèse. Chaque ligne décrit une hypothèse, le test qui la confronte, et la condition d'échec qui interromprait le protocole. Aucun ajustement post-hoc des seuils n'est admis.

## Synthèse

| Phase | Hypothèse testée | Test | Condition d'échec |
|------:|------------------|------|-------------------|
| 1 | L'attention a un profil de rang/entropie variable selon (ω, Δ, ℋ) | Cartographie SSG → Structure-MNIST | Rang Hankel ≈ N et entropie spectrale ≈ log N **partout** |
| 1.5 | Il existe au moins un signal observable token par token qui corrèle avec le stress structurel ET reste insensible au bruit pur | Banc 50% SSG + 50% bruit, corrélation Spearman des 3 candidats (S_KL, S_Grad, S_Spectral) sur (ω, Δ, ℋ) — par Oracle si multi-Oracle | Aucun signal ne passe `max(ρ_ω, ρ_Δ) > 0.70` ET `\|ρ_ℋ\| < 0.20` → l'allocation dynamique par token est une illusion statistique, arrêt du protocole |
| 2 | SCH (V3.5) — il existe une **distribution** conditionnelle `P(r_eff \| stress)` reproductible et structurée, dont les statistiques sont prédictibles | SVD, r_eff(0,95) et r_eff(0,99), distribution complète (médiane/IQR/p10/p90), batterie de tests structurels (DOC/02 5c) | Médiane sature à N partout, **ou** médiane ne croît pas avec le stress, **ou** IQR comparable à médiane sur portion majoritaire des régimes (SCH rejetée ou faiblement corroborée) |
| 2b | Catalogue d'opérateurs structurés couvre la majorité des régimes | Batterie A+B+D (résidu structurel, FFT, PCA cross-régimes du résidu) | Résidu relatif > 30% sur portion non-triviale → catalogue insuffisant, flag d'extension |
| 3 | L'ASPLayer (Backbone + correction Matriochka) atteint la borne de qualité phase 1 à R_max calibré, avec décroissance monotone en r_t | Sanity checks saturation / effondrement / monotonie | Plafond strictement sous la borne phase 1, **ou** décroissance non monotone (Matriochka non effective) |
| 4 | Le Spectromètre, sous λ_budget et curriculum de stress, alloue R_target en fonction croissante du stress local | Diagramme de Phase + courbe Pareto en λ_budget | Diagramme de Phase plat, **ou** R_target décorrélé du stress local, **ou** courbe Pareto plate (un seul régime déguisé) |
| 5a.i | Identifiabilité — R_target reste au plancher sur du bruit blanc | Injection de bruit, mesure de la distribution de R_target | R_target monte sur du bruit → modèle frauduleux |
| 5a.ii | Activation différentielle — Spectromètre catégorise activement, pas silencieux | 4 conditions (bruit, null, trivial, structuré), KL divergence entre paires | diff_score < seuil_silent → Spectromètre passif, ne fait rien d'utile |
| 5b | Élasticité — R_target répond en escalier aux sauts de structure (montée et descente) | Séquences sandwich, mesure du Lag de Réaction | Lag > seuil pré-enregistré, **ou** absence de descente après la zone structurée |
| 5c | SE et HT — l'ASP domine simultanément le Transformer en SE (≥ 2× : Accuracy/Average_Rank) et le SSM pur en HT (parité : Accuracy/Inference_Time) | Mesure SE et HT séparément, multi-baseline, multi-machine si possible | SE ASP < 2× SE Transformer, **ou** HT ASP < HT SSM (échec d'une seule des deux dimensions) |
| 5d | Pareto — l'ASP domine strictement la frontière (qualité, FLOPs) sur ≥ une famille de tâches | Suite multi-baseline, multi-tâche, multi-N | Aucune dominance stricte |
| 5e | OOD — le Spectromètre a appris la *loi* de la structure (SCH), pas des mots-clés liés à un axe | Train sur récursion (ω) → eval sur binding (Δ) et inversement ; train mixte → eval tâches algorithmiques tierces | R_target s'effondre ou reste figé hors axe d'entraînement → polymorphisme superficiel |

## Règles d'application

1. **Séquentialité** — une phase n'est lancée que si la précédente a passé son go-criterion. Une phase entamée puis interrompue documente son échec dans le rapport correspondant et arrête la chaîne.
2. **Conjonction en phase 5** — les sous-tests 5a à 5e sont conjoints. L'échec d'un seul est l'échec de la phase. Pas de compensation entre eux.
3. **Pas de double standard** — les baselines de phase 5 reçoivent les mêmes ressources d'entraînement, les mêmes données, les mêmes seeds que l'ASP. Si l'ASP demande un traitement particulier (warmup, scheduling) pour fonctionner, ce traitement doit être justifié et documenté.
4. **Pré-enregistrement** — seuils (ε_target, R_max cible, formes de loss, valeurs de λ_budget, métriques de phase 5, valeurs de N, plages OOD) fixés *avant* l'expérimentation, publiés dans le rapport correspondant. Tout changement post-hoc déprécie la conclusion.
5. **Échecs documentés** — un no-go n'est pas une non-information. L'arrêt du protocole produit un rapport négatif qui est un résultat scientifique : il ferme une voie et oriente la suivante.
6. **Trois sets disjoints (règle des sets)** — les données du SSG sont partitionnées en `train_oracle`, `audit_svd`, `init_phase3`, partitionnement pré-enregistré en début de phase 1 et jamais modifié. Toute mesure faite sur des données vues ailleurs dans le protocole est invalide.
7. **Intégrité de l'Oracle** — pas de modification post-entraînement (pruning, distillation, quantization). Pas d'optimisations qui modifient les patterns d'attention (GQA, sliding window). Cf. DOC/01 section 8.
8. **Reporting honnête de l'init** — si la phase 3 utilise un Smart Init exploitant l'Oracle, l'ablation random init doit être reportée et l'écart quantifié en phase 5. Cf. DOC/05 section 6b.
9. **No-preconception import (Discovery > Reproduction)** — aucune architecture pré-existante (Mamba, S4, Hyena, MoD, etc.) n'est importée comme composant du protocole. Le Backbone phase 3 est dérivé du dictionnaire SCH phase 2. Les signaux phase 1.5 sont calculés directement, pas via un proxy importé. Cf. DOC/00 section 4b.
10. **Multi-Oracle obligatoire si extension multi-domaine** — toute claim de validité au-delà du domaine cœur (Structure-MNIST) exige un Oracle supplémentaire entraîné from scratch sur ce domaine. Pas de "transfer learning narratif".
11. **Scope attention-only** — le projet étudie l'attention comme sous-module. MLP/FFN/résiduels conservés tels quels. Les comparateurs phase 5 sont évalués à *architecture FFN/résiduel/normalisation strictement identique* à l'ASP. Cf. DOC/00 section 0.
12. **Oracle = borne supérieure, pas cible** — toute claim "ASP plus efficace que l'Oracle" exige le test 6c (R_max réduit). Sans ce test, le rapport peut seulement revendiquer "ASP atteint l'Oracle à coût comparable". Cf. DOC/00 section 3b.
13. **Batterie de tests structurels vivante** — la batterie de phase 2 (DOC/02 section 5c) est une liste ouverte. Tout nouveau test pertinent est documenté (motivation + critère + ID) et propagé. Pas un livrable final, une tâche permanente.

## Hiérarchie des risques

Du plus à risque vers le moins :

1. **Phase 1.5 (Identification Gate)** — c'est *le* point où le projet entier peut s'effondrer prématurément. Si aucun signal ne passe les deux conditions, il n'y a pas de Spectromètre possible et donc pas d'ASP. Sévérité maximale, et le test arrive **avant** tout investissement architectural.
2. **5a (Identifiabilité)** — un modèle apparemment performant peut être démasqué comme une fraude statistique. Risque : Spectromètre sur-paramétré qui encode des heuristiques de tâche. Mitigation : Spectromètre volontairement petit + signal validé en phase 1.5.
3. **5e (OOD croisé)** — extrapolation à un axe de stress non vu ; révèle si le Spectromètre a appris la loi (SCH) ou des mots-clés axe-spécifiques.
4. **Phase 4 (corrélation R_target ↔ stress)** — Spectromètre peut apprendre des proxies non causaux malgré 1.5. Test fragile, surveillé par le Diagramme de Phase et la dérive 4a → 4b.
5. **Phase 2 (SCH)** — le rang effectif peut être bon en moyenne mais rater systématiquement les cas critiques.
6. **5d (Pareto)** — risque de seeding bias : trois seeds sur des tâches longues coûtent cher.
7. **Phase 3 (Matriochka + Consistency)** — instabilité d'optimisation conjointe ; le terme Consistency mitige les sauts de qualité mais peut introduire son propre bruit d'optimisation.
8. **Phase 1 (RCP)** — risque méthodologique faible ; principal danger : sur-spécification du SSG.
