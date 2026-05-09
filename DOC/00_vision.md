# 00 — Vision

## 0. Scope — ce que le projet étudie, et ce qu'il ne touche pas

Le projet étudie l'**attention** comme sous-module du Transformer. Toutes les analyses (SVD, r_eff, SCH, dictionnaire structurel) portent sur la matrice d'attention A et l'opérateur qui la calcule.

Le **MLP/FFN** (qui contient ~2/3 des paramètres d'un Transformer moderne), les **résiduels**, les **LayerNorms**, les **embeddings** et les **têtes de sortie** sont conservés tels quels. Toute amélioration ou dégradation venant de ces composants est **hors-scope**.

Conséquences pratiques :
- L'ASPLayer est un *drop-in replacement* d'une couche d'attention dense, pas d'un Transformer entier.
- Les comparateurs phase 5 sont évalués à *architecture FFN/résiduel/normalisation strictement identique* à l'ASP. Si un Transformer avec MLP plus large obtient mieux, ce n'est pas un échec d'ASP — c'est hors périmètre.
- La prémisse implicite — que la structure intéressante vit dans A — est une **hypothèse de cadrage non interrogée**. Si le travail computationnel important se fait majoritairement dans le MLP, optimiser l'attention seule a un plafond intrinsèque que l'ASP ne pourra pas dépasser.

Ce cadrage est revendiqué, pas masqué. Il limite l'ambition mais rend le projet exécutable.

## 1. Le problème

L'attention pleine est en O(N²) en temps et mémoire. Les alternatives linéaires (SSM, attention linéaire, RNN modernes) règlent le coût mais perdent la qualité sur les tâches qui demandent de la **récursion** ou des **dépendances longues**. Le débat actuel oppose deux positions :

- **Attention pleine** — qualité maximale, coût rédhibitoire à grand N.
- **SSM / linéaire** — coût plat, plafond de qualité observable empiriquement sur tâches structurées.

Le protocole ASP rejette l'opposition : il fait l'hypothèse que la qualité de l'attention ne se distribue pas uniformément dans la séquence. Elle vit dans des **moments critiques** rares ; le reste du temps, une structure linéaire suffit.

## 2. La thèse

> La quantité de calcul utile par token est variable et mesurable. Un modèle qui alloue dynamiquement son rang d'interaction peut atteindre la qualité d'une attention pleine en payant le coût d'un SSM en moyenne.

Trois sous-hypothèses, falsifiables individuellement :

1. **Hétérogénéité mesurable** — les tâches admettent un profil (rang Hankel, entropie spectrale) variable selon paramètres structurels (ω récursion, Δ distance, ℋ entropie). *Testée en phase 1 via le SSG.*
2. **SCH — correspondance structurelle** — il existe une correspondance reproductible (ω, Δ, ℋ) → r_eff sur le rang structurel effectif (mesuré par SVD de la matrice d'attention des Oracles), résumée par une loi de transfert ajustable. *Question ouverte : la loi est-elle universelle entre domaines (vision, code, langue) ou domain-spécifique ? Testée en phase 2 sur plusieurs Oracles ; statut épistémique d'hypothèse, pas de théorème.*
3. **Allocation contrôlable** — un mini-réseau (le Spectromètre) peut, sous pression d'une loss de budget λ_budget, prédire et allouer le rang token par token de manière causalement reliée à la difficulté locale. *Construite en phases 3 et 4, validée par les cinq tests de phase 5 (Identifiabilité, Élasticité, SE/HT, Pareto, OOD).*

## 3. Le livrable

Une frontière de Pareto **strictement dominée** sur l'axe (qualité, complexité), validée en phase 5. Les conditions précises :

- À iso-qualité du Transformer plein : FLOPs/token et mémoire restent **plats** quand N croît.
- À iso-complexité d'un SSM pur : gain mesurable de qualité sur les tâches récursives et longue distance.

Si aucun de ces deux régimes n'est atteint, le protocole a échoué.

### 3b. L'Oracle est une *borne supérieure*, pas une cible à reproduire

Précision méthodologique critique. L'Oracle est un Transformer dense entraîné — **il n'est pas le comportement optimal**, c'est le comportement *appris sous contraintes de softmax non-régularisée*. Si l'Oracle alloue r_eff = 30 sur un régime, c'est peut-être parce que :

- l'attention dense **sur-utilise sa capacité par défaut** (toujours allouer le maximum disponible est le minimum local naturel d'une softmax),
- les têtes redondantes ne peuvent pas se "désactiver" individuellement même si elles n'apportent rien,
- l'optimisation Adam/AdamW pousse vers des solutions denses sans pression budgétaire explicite.

Une architecture vraiment optimale pourrait faire la même tâche à r_eff = 5. L'objectif d'ASP n'est pas de **matcher la consommation de rang de l'Oracle** ; c'est d'**atteindre la qualité de l'Oracle à strictement moins de rang**.

Conséquences concrètes :
- La loi de transfert phase 2 décrit ce que l'Oracle utilise, pas ce qui est nécessaire.
- La distillation phase 4a est un *prior souple*, pas un plafond.
- Phase 5 inclut un test "R_max réduit" (DOC/05 section 6c) où l'ASP est entraîné avec `R_max = médiane(r_eff_oracle) / 2`. Si la qualité tient, l'ASP a *dépassé* l'Oracle structurellement.

## 4. Pourquoi ce n'est pas une optimisation

L'argument n'est pas « on peut compresser un Transformer en gardant 99 % de la qualité ». L'argument est que l'attention **dilue** sa capacité sur des tokens qui n'en avaient pas besoin, et qu'une mesure préalable révèle où elle aurait pu être économisée. La compression n'est pas un effet recherché ; c'est la conséquence d'un meilleur modèle de la demande.

## 4b. Principe fondateur — Discovery, pas Reproduction

Le protocole ne vise pas à *reproduire* une architecture existante (Mamba, RetNet, Hyena, MoD, etc.) en y ajoutant une touche de polymorphisme. Il vise à **découvrir** la structure que l'attention exhibe sur des données réelles, et à dériver un opérateur qui *implémente* cette structure mesurée.

Conséquences strictes :

1. **Aucune architecture n'est pré-sélectionnée**. Le Backbone de l'ASPLayer est construit *à partir* du dictionnaire SCH issu de la phase 2 — pas importé d'un projet académique antérieur. Si la phase 2 révèle que les régimes faciles sont Toeplitz de rang 2, le Backbone est un opérateur Toeplitz paramétré. Pas un Mamba2 entraîné à se comporter comme.
2. **Aucun signal n'est emprunté à une architecture pré-existante**. Le Spectromètre se calibre sur des signaux mesurés directement (KL local, gradient, rang spectral), pas sur des résidus calculés via un proxy importé.
3. **Aucun benchmark n'est sacralisé sans audit**. LRA, induction, copy, etc. sont des comparateurs, pas des cibles. La validation se fait d'abord sur le SSG (où la structure est connue par construction), ensuite sur des domaines extérieurs.
4. **L'Oracle n'est pas unique**. Différents domaines (vision, code, langue, math) exhibent des signatures structurelles différentes — la vision a tendance à la commutativité (héritée des conv), le code à la hiérarchisation (binding lourd), la langue à un mélange. Le protocole utilise **plusieurs Oracles**, un par domaine, pour découvrir si la SCH est universelle ou domain-spécifique.

C'est cette discipline du *fresh start* qui distingue le projet d'une optimisation marginale parmi d'autres. Le succès n'est pas "ASP bat MoD de 5 %" — c'est "voici la structure mathématique que l'attention exhibe sur ces domaines, et voici l'opérateur dérivé qui l'implémente avec un coût plat".

## 5. Carte du protocole

```
Phase 1   (RCP / SSG + multi-Oracle) →  Oracles entraînés (1 par domaine), baselines de difficulté
       ↓
Phase 1.5 (Calibration Signal)       →  IDENTIFICATION GATE : 3 signaux candidats (S_KL, S_Grad, S_Spectral), ≥ 1 validé ou arrêt
       ↓
Phase 2   (Audit Spectral)           →  SCH corroborée par domaine : r_eff, Stress-Rank Map, loi de transfert + comparaison cross-domain
       ↓
Phase 3   (ASPLayer)                 →  Backbone *dérivé* du dictionnaire SCH (pas importé) + correction Matriochka, Soft-Mask, Loss Consistency
       ↓
Phase 4   (Spectromètre)             →  Politique d'allocation R_target apprise sous λ_budget : 4a (warm-up + distillation) → 4b (autonome)
       ↓
Phase 5   (Validation)               →  5 tests : Identifiabilité, Élasticité, SE/HT, Pareto (comparateurs domain-aware), OOD croisé
```

Chaque flèche est conditionnelle au passage du critère go/no-go de la phase amont (cf. [falsifiabilite.md](falsifiabilite.md)).
