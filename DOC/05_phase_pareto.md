# 05 — Phase 5 : Validation et Falsification

L'heure de vérité. On prouve que l'ASP n'est pas un gros modèle déguisé, mais un **système cybernétique** qui adapte sa géométrie relationnelle à la complexité réelle de l'information.

Cette phase produit la section « Résultats » du rapport final. Cinq tests, ordonnés du plus discriminant au plus contextuel.

## 1. Test d'Identifiabilité — l'anti-fraude

**Question.** Le Spectromètre réagit-il à la **structure**, ou simplement à la présence de données ?

**Protocole de base (5a.i).** Soumettre au modèle entraîné des entrées **non structurées** :
- bruit blanc gaussien dans l'espace d'entrée,
- séquences de tokens uniformément aléatoires,
- séquences à entropie ℋ maximale du SSG.

Mesurer la distribution de R_target sur ces entrées.

**Résultat attendu.** R_target reste à son niveau plancher (r_min, idéalement 0).

**Falsification.** Si R_target augmente sur du bruit, le Spectromètre **hallucine** de la structure ou utilise le rang pour mémoriser du bruit (overfitting). Le modèle est déclaré **non-identifiable** et le protocole échoue, indépendamment des autres résultats.

### 1b. Test différentiel d'activation (5a.ii)

Ajouté V3.5. Le test 5a.i seul ne distingue pas deux comportements :

- **Spectromètre silencieux** : R_target = 0 partout par défaut. Non-allocation passive.
- **Spectromètre intelligent** : R_target = 1 ou 2, allocation explicitement basse parce que le modèle *reconnaît* le bruit. Catégorisation active.

Les deux passent 5a.i. Mais le second est ce qu'on veut, le premier est inutile (et trompeur — il passerait aussi tous les tests sur structure si le défaut était maintenant haut).

**Protocole.** Soumettre quatre conditions distinctes au Spectromètre :

| Condition | Description | R_target attendu |
|-----------|-------------|------------------|
| `(a) bruit blanc` | Bruit gaussien aléatoire | bas |
| `(b) null/empty input` | Token de padding répété, ou input nul | très bas (idéalement = 0) |
| `(c) répétition triviale` | Même token répété sur toute la séquence | bas mais > (b) |
| `(d) input structuré` | Exemple SSG à stress moyen | élevé |

**Calcul du score différentiel.** Pour chaque condition c ∈ {a, b, c, d}, distribution `D_c` de R_target sur N exemples. Score :

```
diff_score = (1/6) Σ_{i<j} KL( D_i || D_j )    (somme sur les 6 paires)
```

**Critères.**

- **Spectromètre actif** (souhaité) : `diff_score > seuil_actif` pré-enregistré (ex. 0.5 nats). Le Spectromètre catégorise les conditions différemment.
- **Spectromètre silencieux** : `diff_score < seuil_silent` pré-enregistré (ex. 0.1 nats). Le Spectromètre produit la même distribution partout — il ne fait rien.
- **Zone grise** : entre les deux, le comportement est ambigu.

Critère de validation 5a complet : 5a.i passe **ET** diff_score > seuil_actif **ET** R_target sur (d) > R_target sur (a, b, c). La conjonction garantit à la fois qu'il n'y a pas de fraude (5a.i) et qu'il y a une réelle catégorisation active (5a.ii).

## 2. Test d'Élasticité — réponse impulsionnelle

**Question.** Le Spectromètre adapte-t-il rapidement et précisément le rang à un changement brutal de régime ?

**Protocole.** Construire des séquences « sandwich » :

```
[bruit] → [structure forte (Dyck-12, induction profonde)] → [bruit]
```

Tracer R_target en fonction de la position du token.

**Résultat attendu.** Une **réponse en escalier** (step response) : montée nette dès le premier token de la zone structurée, plateau pendant la zone, chute immédiate après le dernier token structuré.

**Métrique : le Lag de Réaction.** Distance (en tokens) entre l'apparition de la structure et le moment où R_target atteint la moitié de sa montée. Un bon ASP : Lag ≈ 0–2 tokens. Un Lag > 10 trahit un Spectromètre paresseux ou un signal de surprise mal choisi.

**Variante symétrique.** Vérifier aussi que R_target *redescend* à la fin de la zone structurée. Un Spectromètre qui ouvre vite mais ne ferme jamais est incomplet — il sait monter, pas allouer.

## 3. SE et HT — métriques d'efficience décomposées

L'ancien SRI composite (`Accuracy / (Average_Rank × Inference_Time)`) confondait deux grandeurs hétérogènes : un proxy mathématique (rang) et une réalité matérielle (latence). Un modèle peut bien classer en SRI tout en étant lent en pratique, ou inversement. La V2 du protocole les sépare.

### 3.1 Structural Efficiency (SE)

```
SE = Accuracy / Average_Rank
```

- **Hardware-indépendant** — c'est un ratio mathématique pur.
- Mesure : à quel point chaque unité de rang allouée se traduit en performance.
- Comparable d'une machine à l'autre, d'une implémentation à l'autre.

### 3.2 Hardware Throughput (HT)

```
HT = Accuracy / Inference_Time     (ou tokens·s⁻¹ · accuracy_normalisée)
```

- **Hardware-dépendant** — capture la latence et la friendliness mémoire.
- Mesure : à quel point le modèle exploite efficacement le silicium disponible.
- Doit être rapporté avec le contexte hardware (GPU, batch size, kernel utilisé).

### 3.3 Pourquoi les rapporter séparément

- Un modèle low-rank avec accès mémoire dispersés peut avoir un SE excellent et un HT médiocre. Le rapporter en composite cache cette asymétrie.
- Inversement, un modèle dense bien optimisé peut avoir un HT élevé et un SE médiocre — il « gaspille » du calcul mais le hardware le supporte.
- Le composite ancien classait dans le mauvais sens dans ces cas-là.

**Lecture combinée.**
- Un Transformer plein : Accuracy élevée, Average_Rank constant (≈ N) → SE faible. Inference_Time élevé → HT faible. Cohérent, le Transformer paie partout.
- Un SSM pur : Average_Rank ≈ 1 → SE potentiellement très élevé. Inference_Time bas → HT élevé. Mais Accuracy borne supérieure plus basse.
- L'**ASP** doit viser à **dominer simultanément** sur les deux axes : SE 2–3× supérieur au Transformer (justification : si le Spectromètre alloue la moitié du rang max en moyenne, gain théorique 2×, on demande au moins ce minimum) **et** HT au moins équivalent au SSM pur (sinon le polymorphisme est cosmétique).

Aucune des deux métriques n'est rapportée en valeur absolue isolée — comparaison relative entre modèles, même suite, même machine.

## 4. Frontière de Pareto

Graphique central de la section Résultats. Comparaison ASP vs baselines sur (qualité, complexité) :

- **Axe X** — coût de calcul : FLOPs / token, mesurés (pas estimés analytiquement).
- **Axe Y** — qualité : accuracy ou loss, sur tâches Structure-MNIST + LRA + sous-échantillon LM.

Comparateurs entraînés à **ressources strictement appariées** (FLOPs d'entraînement, données, optim, seeds), choisis **selon le domaine** évalué.

### Sequence-domain (Structure-MNIST séquentialisé, code synthétique, langue)

| Modèle | Régime | Granularité d'allocation | Coût |
|--------|--------|--------------------------|------|
| Transformer plein | Attention quadratique | Aucune (toujours plein) | O(N²) |
| Mamba2 (ou SSM SOTA) | Récurrent sélectif | Aucune (toujours minimal) | O(N) |
| Linear Attention | Attention linéaire | Aucune | O(N) |
| Hyena / RWKV | Hybride | Aucune | O(N log N) ou O(N) |
| **MoD** (Mixture of Depths) | Allocation par couche | **Binaire** (passer / sauter la couche) | O(N · k/L), k = nb tokens activés |
| **ASP** (notre) | Allocation par token | **Spectrale** (rang continu r_t ∈ [0, R_max]) | O(N · C̄) |

### Vision-domain (si évalué : Structure-MNIST 2D, MNIST/CIFAR par patches)

| Modèle | Régime | Granularité d'allocation | Coût |
|--------|--------|--------------------------|------|
| ViT (Vision Transformer) | Attention quadratique sur patches | Aucune | O(P²), P = nb patches |
| ConvNet (ResNet ou ConvNeXt) | Conv hiérarchique | Aucune (équivariance par construction) | O(P) |
| MLP-Mixer | Mélange spatial + canal | Aucune | O(P²) |
| **ASP** (notre) | Allocation par patch | **Spectrale** | O(P · C̄) |

### Code-domain (si évalué : Dyck-k, mini-DSL synthétique)

| Modèle | Régime | Granularité d'allocation | Coût |
|--------|--------|--------------------------|------|
| Transformer plein | Attention quadratique | Aucune | O(N²) |
| MoD | Allocation par couche | Binaire | O(N · k/L) |
| **ASP** | Allocation par token | Spectrale | O(N · C̄) |

**Principe** : ne jamais comparer ASP à des baselines d'un autre domaine. Comparer un ASP-vision à un Mamba (sequence-only) serait une comparaison déloyale dans les deux sens (Mamba n'est pas conçu pour la vision ; les conv ne le sont pas pour les séquences arbitraires).

### Pourquoi MoD est le baseline principal

MoD (Raposo et al., DeepMind 2024) est l'art récent le plus proche conceptuellement : c'est aussi de l'allocation dynamique de calcul par token. Si l'ASP ne se distingue pas de MoD, le projet n'apporte rien.

**Distinction structurelle** :
- MoD fait un **routage binaire** : pour chaque couche, chaque token soit traverse la couche soit la saute. C'est un choix discret 0/1.
- ASP fait une **allocation spectrale** : pour chaque token, le rang de la correction varie continûment de 0 à R_max. C'est une décision graduée.

**Hypothèse de supériorité** : la difficulté structurelle du langage n'est pas binaire (utile / inutile). Elle est *spectrale* — un token peut nécessiter peu, modérément, ou beaucoup de capacité relationnelle. Si cette hypothèse est vraie, ASP doit dominer MoD à coût comparable. Sinon, MoD suffit et l'ASP n'est qu'une complication.

**Test décisif** : sur les tâches de récursion profonde (où la difficulté monte progressivement avec ω), ASP devrait montrer une montée progressive de R_target alors que MoD n'a que deux états (passer / sauter). Si MoD performe aussi bien sur ces tâches, la granularité spectrale est cosmétique.

**Critère de succès.** L'ASP doit se situer **sur la frontière de Pareto** — point le plus haut pour chaque budget de calcul donné. À iso-FLOPs avec le Transformer, qualité comparable et FLOPs plats quand N croît. À iso-FLOPs avec Mamba2, gain mesurable de qualité sur les tâches récursives.

## 5. Test de généralisation hors distribution (OOD)

**Question.** Le Spectromètre a-t-il appris **la loi** de la structure (SCH), ou des **mots-clés** spécifiques à l'axe de stress vu à l'entraînement ?

**Protocole — croisement d'axes (V2)**. La V1 du test extrapolait sur le même axe (train ω∈{2,4,6}, test ω∈{8,10,12}) — c'était trop facile : un Spectromètre qui interpole linéairement passe sans avoir appris quoi que ce soit. La V2 **croise les familles de stress** :

| Régime d'entraînement | Régime d'évaluation OOD |
|----------------------|------------------------|
| Récursion dominante (ω varié, Δ minimal) | Binding dominant (Δ varié, ω minimal) |
| Binding dominant (Δ varié, ω minimal) | Récursion dominante (ω varié, Δ minimal) |
| Récursion + Binding mixte | Tâches algorithmiques tierces (induction, copy) |

### Opérationnalisation de "Binding"

Au sens cognitif strict, *binding* désigne la liaison entre features distinctes (variable binding, anaphora, coreference). Dans le SSG actuel, on opérationnalise binding comme **régime Δ-dominé** : tâches où le défi structurel est de relier deux tokens éloignés (référence, copie, association à distance), sans composition récursive significative.

Cette opérationnalisation est **un raccourci**, pas une équivalence. Elle suffit pour ce test parce que la distinction critique entre récursion et binding dans le SSG est l'axe de stress dominant (ω vs Δ), pas la nature exacte de la tâche. Pour un test plus rigoureux, l'enrichissement futur du SSG devrait inclure :

- **Vraies tâches de liaison** : variable assignment + later use (programmation simple), résolution de références pronominales sur séquences synthétiques.
- **Tâches de composition non-récursive** : combinaison de features par règles, sans appels imbriqués.

Dans la version actuelle du SSG, ces enrichissements ne sont pas requis pour passer le test 5e — mais leur absence est documentée comme limite.

L'idée : le Spectromètre doit allouer du rang correctement sur des **mécanismes structurels qu'il n'a jamais vus dans cette intensité**, en s'appuyant uniquement sur les invariants de la SCH (énergie spectrale, signal de surprise structurel).

**Résultat attendu.**
- R_target s'élève sur les zones structurées même si la *forme* du stress diffère du training.
- L'accuracy se dégrade gracieusement, pas brutalement.
- Le Lag de Réaction reste comparable.
- Le R_target moyen sur la tâche cross-axe est cohérent avec ce que la loi de transfert phase 2 prédirait.

**Test secondaire (V1 conservé).** L'extrapolation sur le même axe (ω restreint → ω étendu) reste rapportée comme **diagnostic complémentaire** mais n'est plus le critère canonique de 5e.

**Falsification.** Si R_target s'effondre dès qu'on change d'axe — ou s'il reste constant à la valeur vue à l'entraînement — le Spectromètre a appris une *fonction* (de l'axe d'entraînement) mais pas une *loi* (de la structure). Le polymorphisme est superficiel.

C'est le test de **robustesse ultime** : il décide si l'ASP est un système cybernétique ou un look-up table sophistiqué.

## 6. Verdict — conjonction stricte

Les cinq tests sont **conjoints**.

| Test | Échec → conséquence |
|------|---------------------|
| 1. Identifiabilité | Modèle frauduleux. Pas de retour amont sans repenser le Spectromètre. |
| 2. Élasticité | Polymorphisme non temps réel. Retour à la phase 4 (curriculum, signal de surprise). |
| 3. SE / HT | Pas de gain d'efficience structurelle ou matérielle. ASP n'est pas Pareto-utile. |
| 4. Pareto | Pas de dominance stricte. Conclusion négative globale. |
| 5. OOD | Polymorphisme superficiel. ASP est un look-up déguisé. |

Le succès complet (les cinq tests passés) constitue la preuve du concept. Tout autre résultat appelle un retour amont documenté ou l'arrêt du protocole.

## 6c. Test décisif — l'Oracle comme borne supérieure

Test ajouté V3.5. La loi de transfert phase 2 décrit ce que l'**Oracle** consomme en rang, pas ce qui est *nécessaire*. L'objectif d'ASP est d'atteindre la qualité de l'Oracle à *strictement moins* de rang. Ce test le mesure directement.

**Protocole.**

1. Prendre la médiane de r_eff de l'Oracle sur le régime de stress moyen : `r_med = médiane(r_eff_oracle | régime moyen)`.
2. Entraîner un ASPLayer avec `R_max = r_med / 2` (la moitié de ce que l'Oracle utilise en moyenne).
3. Évaluer la qualité sur la suite phase 5.

**Critères.**

- **Succès** : l'ASP avec R_max réduit atteint au moins 95 % de la qualité de l'Oracle. **Cas le plus fort possible** : ASP a structurellement dépassé l'Oracle (utilise moins, qualité comparable).
- **Succès partiel** : 80–95 % de la qualité. ASP est compétitif sans être strictement supérieur.
- **Échec** : < 80 %. L'ASP a besoin d'au moins autant de rang que l'Oracle ; pas de gain structurel intrinsèque, seulement le polymorphisme dynamique.

**Statut** : test optionnel pour Pareto strict, **obligatoire pour toute claim "ASP dépasse l'Oracle"**. Si le test n'est pas exécuté, le rapport ne peut pas revendiquer plus que "ASP atteint l'Oracle à coût comparable" — pas "ASP est plus efficace structurellement".

## 6b. Reporting honnête de l'initialisation

Si la version de l'ASP rapportée en phase 5 utilise le **Smart Init Matriochka** (cf. DOC/03 section 5.2), les conditions de reporting sont :

1. **Déclaration explicite** : préciser dans le rapport qu'un init smart est utilisé, et quels têtes/colonnes sont initialisés à partir de l'Oracle.
2. **Ablation random obligatoire** : reporter aussi les chiffres de l'ASP entraîné en random init, sur la même suite et avec les mêmes hyperparamètres.
3. **Quantification de l'écart** : reporter la différence (qualité, FLOPs, SE, HT) entre les deux versions.
4. **Comparaison équitable** : si smart init donne un gain significatif, les baselines (Transformer, Mamba2, MoD, etc.) doivent recevoir une comparaison à *random init ASP* aussi, sinon on compare un ASP boosté à des baselines vanilla.

Sans ces conditions, les chiffres Pareto de l'ASP ne sont pas comparables aux baselines : on aurait injecté de l'information de l'Oracle uniquement dans l'ASP, en violation du principe "ressources strictement appariées" (section 4).

## 7. Livrables de l'étape

1. **Heatmap de Rang** — visualisation de R_target sur des séquences typées, montrant l'allumage sélectif sur les zones structurées.
2. **Tableau comparatif** — ASP vs baselines sur Accuracy, FLOPs/token, Inference_Time, SE, HT (rapportés séparément).
3. **Courbes des tests d'Identifiabilité et d'Élasticité** avec données brutes.
4. **Frontière de Pareto consolidée**, multi-tâches, multi-N.
5. **Rapport OOD** — courbes de généralisation, comportement de R_target en extrapolation.
6. **Rapport de Falsification** — déclaration explicite, par test, succès / échec contre les conditions pré-enregistrées.

Le rapport ne contient pas de spéculation sur les extensions tant que les cinq tests ne sont pas tous validés.

## 8. Note méthodologique

La tentation, à ce stade, est d'ajuster les seuils ou les définitions de SE / HT pour faire passer le résultat. Le protocole interdit cet ajustement : seuils SE et HT, conditions de succès, valeurs de N, plages de stress OOD sont arrêtés *avant* l'expérimentation, et tout changement post-hoc invalide la phase.
