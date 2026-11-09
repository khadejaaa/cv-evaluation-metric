# Métriques d'évaluation en vision par ordinateur : des formules au code

Projet du cours **Vision par ordinateur** – Master Data & IA, année 2025–2026.
Auteure : Khadija ABOUZAKY

Chaque métrique est programmée « à la main » avec NumPy, puis **vérifiée avec scikit-learn** quand c'est possible.
Le but : comprendre ce que mesure chaque chiffre, et surtout quand il peut tromper.

## Contenu du dépôt

- `metriques_evaluation_vision.ipynb` : le code source (notebook commenté, en français)
- `requirements.txt` : les bibliothèques à installer avec `pip install -r requirements.txt`
- `README.md` : cette documentation

## Installation et lancement

- Prérequis : Python 3.9 ou plus récent
- Cloner le dépôt : `git clone https://github.com/khadejaaa/cv-evaluation-metrics.git`
- Aller dans le dossier : `cd cv-evaluation-metrics`
- **Installer les dépendances (obligatoire) : `pip install -r requirements.txt`**
- Ouvrir le notebook : `jupyter notebook metriques_evaluation_vision.ipynb`
- Exécuter toutes les cellules (Kernel → Restart & Run All)
- Aucun téléchargement de données n'est nécessaire : tout est généré ou fourni par scikit-learn (jeu `digits`)
- Les résultats sont reproductibles (graine aléatoire fixe : `default_rng(42)`)

## Idée centrale

Un seul chiffre peut mentir. Un modèle qui répond toujours « négatif » sur des données où 95 % des cas sont négatifs a **95,5 % d'exactitude** mais **0 % de rappel**. Évaluer, c'est choisir la métrique qui révèle l'échec, pas celle qui rassure.

## Étiquettes utilisées dans le notebook

-  **JOUET** : petites données synthétiques, construites pour vérifier les chiffres à la main
- **RÉEL** : un vrai modèle sur un vrai jeu de données (`digits` de scikit-learn)

La détection et la segmentation n'ont pas de cellule « réelle » (il faudrait un réseau pré-entraîné). Les fonctions marchent avec n'importe quel modèle : voir la section « Brancher un vrai modèle ».

## Documentation du code, partie par partie

### 0. Initialisation
- Imports : NumPy, Matplotlib, scikit-learn
- `rng = np.random.default_rng(42)` : graine fixe pour obtenir toujours les mêmes résultats

### 1. Classification d'images

Formules clés :
- Exactitude = (VP+VN) / (VP+VN+FP+FN)
- Précision = VP / (VP+FP)
- Rappel = VP / (VP+FN)
- F1 = 2VP / (2VP+FP+FN)
- AUC = probabilité qu'un positif ait un meilleur score qu'un négatif

Fonctions programmées :
- `binary_metrics(y, p)` : calcule VP, FP, FN, VN, exactitude, précision, rappel et F1
- `auc_rank(y, s)` : calcule le ROC-AUC par comparaison des scores (sans scikit-learn)

Cellule JOUET (2 000 images, environ 5 % de positifs) :
- Les résultats sont comparés à scikit-learn avec des `assert` : tout coïncide
- Résultat : exactitude 0,837 · précision 0,188 · rappel 0,780 · F1 0,303 · AUC 0,889
- Piège du déséquilibre : le modèle « toujours négatif » a une exactitude de 0,955 mais un rappel et un F1 de 0
- Graphiques : matrice de confusion, courbe ROC, comparaison exactitude / rappel / F1

Cellule RÉEL (régression logistique sur `digits`, 10 classes) :
- `classification_report` complet
- Moyennes **macro** (simple sur les classes), **micro** (on cumule les comptes) et **weighted** (pondérée par le nombre d'exemples)
- Résultat : F1 macro 0,972 · ROC-AUC macro (un contre tous) 0,999
- Le F1 macro est aussi recalculé à la main depuis la matrice de confusion et vérifié

### 2. Détection d'objets

Règle : une prédiction est un **VP** si la classe est bonne et IoU ≥ seuil avec une vérité terrain pas encore appariée (par confiance décroissante). Sinon c'est un **FP**. Une vérité terrain non appariée est un **FN**.

Formules clés :
- IoU = aire de l'intersection / aire de l'union
- AP = aire sous la courbe précision-rappel (interpolation COCO à 101 points)
- mAP = moyenne des AP sur les classes
- AP50:95 = moyenne des AP pour les seuils d'IoU 0,50 ; 0,55 ; … ; 0,95

Fonctions programmées :
- `box_iou(a, b)` : IoU de deux boîtes au format `[x1, y1, x2, y2]`
- `match(preds, gts, iou_fn, thr)` : appariement glouton pour une classe, renvoie les indicateurs VP triés par score
- `pr_curve(tp, n_gt)` : courbe précision-rappel cumulée
- `average_precision(rec, prec)` : AP avec enveloppe de la précision et 101 points
- `evaluate_ap(preds, gts, iou_fn)` : AP par classe et par seuil, puis AP50, AP75 et AP50:95

Tests de cohérence : détections parfaites donnent AP = 1, aucune détection donne AP = 0.

Cellule JOUET (3 images, 2 classes : bonne boîte, boîte approximative, mauvaise classe, doublon, fausse alarme) :
- À IoU 0,5 et seuil de score 0,5 : VP = 5, FP = 1, FN = 0, précision 0,83, rappel 1,00
- AP50 = 1,00 · AP75 = 0,25 · AP50:95 = 0,43
- Lecture : le critère souple (IoU 0,5) cache de mauvaises boîtes, que l'AP50:95 révèle
- Vérification optionnelle avec `torchmetrics` (ignorée si le package n'est pas installé)

### 3. Segmentation sémantique et d'instance

Tout part d'une matrice de confusion `cm[i, j]` (pixels de vraie classe i prédits j).

Formules clés :
- Précision pixel = somme de la diagonale / total des pixels
- IoU(c) = VP / (VP+FP+FN), et mIoU = moyenne sur les classes
- Dice(c) = 2VP / (2VP+FP+FN), avec Dice = 2·IoU / (1+IoU)

Fonctions programmées :
- `confusion_seg(gt, pred, n_classes, ignore_index)` : matrice de confusion sur les pixels (option pour ignorer une étiquette)
- `seg_metrics(cm)` : précision pixel, IoU, mIoU, Dice et Dice moyen
- `mask_iou(a, b)` : IoU entre deux masques booléens

Cellule JOUET sémantique (fond, disque, carré, prédiction décalée avec un peu de bruit) :
- Précision pixel 0,927 · mIoU 0,763 · Dice moyen 0,861
- L'identité Dice = 2·IoU / (1+IoU) est vérifiée par `assert`
- Piège : une prédiction 100 % fond a une précision pixel de 0,796 mais un mIoU de 0,265

Cellule JOUET instance (Mask AP) :
- Même mécanique que la détection, mais avec l'IoU calculé sur les masques
- Deux objets qui se touchent sont fusionnés en un seul masque : IoU de 0,43 avec chaque objet, donc les deux sont ratés à IoU 0,5
- Résultat : Mask AP50 = 0,51 · Mask AP50:95 = 0,33
- Les métriques sémantiques ne voient pas cette erreur, le Mask AP oui

## Brancher un vrai modèle

Les fonctions s'utilisent telles quelles avec de vraies prédictions (par exemple un détecteur YOLO) :

- Segmentation : cumuler la matrice de confusion sur tout le jeu de données, **puis** calculer IoU et Dice
  - `cm += confusion_seg(gt_map, model_predict(image), C, ignore_index=255)`
  - `seg_metrics(cm)`
- Détection ou instance : construire deux listes de dictionnaires, puis appeler `evaluate_ap`
  - `preds = [dict(img=i, cls=c, score=s, obj=boite_ou_masque), ...]`
  - `gts = [dict(img=i, cls=c, obj=boite_ou_masque), ...]`
  - `evaluate_ap(preds, gts, box_iou)` ou `evaluate_ap(preds, gts, mask_iou)`
- Garder le même format de coordonnées et les mêmes identifiants de classes pour le modèle et la vérité terrain

## Quelle métrique choisir ?

- Classification : rapporter le F1 par classe et le ROC-AUC. Attention à l'exactitude quand les classes sont déséquilibrées
- Détection : rapporter AP50, AP50:95 et mAP. Attention : un seuil d'IoU souple cache de mauvaises boîtes
- Segmentation : rapporter mIoU et Dice, pas seulement la précision pixel (dominée par le fond)



## Technologies

- Python 3, NumPy, Matplotlib, scikit-learn
- Optionnel : PyTorch et torchmetrics pour la vérification COCO
