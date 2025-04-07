# Apprentissage Fédéré Optimisé pour MNIST

Ce projet implémente une version optimisée de l'apprentissage fédéré (Federated Learning) avec approvisionnement des données par les edges, appliquée au dataset MNIST.

## Table des matières
1. [Introduction](#introduction)
2. [Prérequis](#prérequis)
3. [Installation et configuration](#installation-et-configuration)
4. [Exécution du code](#exécution-du-code)
5. [Améliorations apportées](#améliorations-apportées)
6. [Résolution des problèmes courants](#résolution-des-problèmes-courants)
7. [Analyse des résultats](#analyse-des-résultats)
8. [Perspectives](#perspectives)

## Introduction

L'apprentissage fédéré est une approche de machine learning où l'entraînement est effectué sur des dispositifs ou serveurs décentralisés (edges) qui conservent leurs données localement, sans les partager. Cette implémentation optimisée applique ce concept au dataset MNIST des chiffres manuscrits, avec plusieurs améliorations visant à augmenter les performances du modèle global.

Contrairement à une approche centralisée traditionnelle, l'apprentissage fédéré permet de respecter la confidentialité des données et de réduire les coûts de communication en ne partageant que les mises à jour des modèles plutôt que les données brutes.

## Prérequis

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Le dataset MNIST au format d'images organisées en dossiers par classe

## Installation et configuration

1. Clonez ce dépôt ou téléchargez les fichiers source.

2. Téléchargez le dataset MNIST au format JPEG depuis Kaggle:
   - Rendez-vous sur https://www.kaggle.com/datasets/scolianni/mnistasjpg
   - Téléchargez et extrayez le fichier ZIP
   - Conservez le dossier `trainingSet` qui contient les images d'entraînement

3. Installez les dépendances requises:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn
   ```
   
   Note: Si vous rencontrez des erreurs lors de l'installation de TensorFlow, essayez:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.12.0
   ```

4. Ouvrez le fichier `federated_learning_dylan.py` et modifiez le chemin vers le dataset MNIST:
   ```python
   mnist_base_path = '/chemin/vers/votre/MNIST/trainingSet/trainingSet/'
   ```

## Exécution du code

Pour exécuter l'implémentation optimisée d'apprentissage fédéré:

```bash
python federated_learning_dylan.py
```

Pendant l'exécution, vous verrez:
- L'initialisation des edges
- La distribution des classes aux différents edges
- Les résultats de chaque round d'apprentissage fédéré
- Une comparaison avec un modèle centralisé
- Des graphiques comparant les performances

Le script génère également un fichier image `federated_vs_centralized_improved.png` qui montre l'évolution de la perte et de la précision au fil des rounds.

## Améliorations apportées

Cette implémentation optimisée apporte plusieurs améliorations par rapport à l'approche de base:

### 1. Distribution stratégique des données
- Chaque edge reçoit au moins 3 classes pour assurer une diversité minimale
- Distribution alternée pair/impair pour une meilleure répartition globale
- Garantie que chaque classe est représentée dans au moins un edge

### 2. Algorithme FedProx
- Utilisation de FedProx au lieu de FedAvg standard
- Ajout d'un terme de proximité (μ) qui limite la divergence des modèles locaux
- Régularisation des modèles locaux pour éviter la spécialisation excessive

### 3. Agrégation adaptative des poids
- Détection et filtrage des contributions aberrantes
- Pondération des edges en fonction de leur historique de stabilité
- Utilisation de momentum pour stabiliser les mises à jour du modèle global

### 4. Optimisations d'entraînement
- Taux d'apprentissage adaptatif selon la taille du dataset local
- Nombre d'époques variable, augmentant vers la fin de l'apprentissage
- Support optionnel pour l'augmentation de données (désactivé par défaut pour éviter les problèmes de compatibilité)

## Résolution des problèmes courants

### Problème d'installation de TensorFlow
Si vous rencontrez des erreurs avec les chemins longs sur Windows:
1. Ouvrez PowerShell en tant qu'administrateur
2. Exécutez: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
3. Redémarrez votre ordinateur

### Erreur avec l'augmentation de données
Si vous obtenez des erreurs liées à l'augmentation de données:
1. Assurez-vous que `data_augmentation=False` est passé à la fonction `compare_with_centralized()`
2. Vérifiez que la fonction `compare_with_centralized()` n'écrase pas cette valeur en interne

### Erreur avec les callbacks
Si vous obtenez une erreur `TypeError: MyModel.fit_it() got an unexpected keyword argument 'callbacks'`:
1. Modifiez la méthode `train_model()` dans la classe `EdgeNode` pour supprimer les callbacks:
   ```python
   self.model.fit_it(trains=self.dataset, epochs=epochs, tests=None, verbose=self.verbose)
   ```

## Analyse des résultats

Après l'exécution, le script affiche une analyse comparative entre le modèle fédéré optimisé et le modèle centralisé:

- Taux d'amélioration du modèle fédéré entre le premier et le dernier round
- Écart final avec le modèle centralisé en points de pourcentage
- Estimation du nombre de rounds supplémentaires nécessaires pour atteindre la performance centralisée

Typiquement, cette implémentation optimisée réduit significativement l'écart entre le modèle fédéré et le modèle centralisé par rapport à l'implémentation de base, atteignant souvent plus de 80% de la performance centralisée.

## Perspectives

Cette implémentation peut être encore améliorée de plusieurs façons:

- Intégration d'autres algorithmes fédérés comme FedOpt ou FedMA
- Exploration de l'apprentissage fédéré asynchrone
- Implémentation de techniques d'apprentissage par transfert
- Ajout de mécanismes de confidentialité différentielle
- Extension à d'autres types de modèles (CNN, transformers)

N'hésitez pas à expérimenter avec les paramètres comme le nombre d'edges, de rounds, ou le facteur de proximité pour observer leurs effets sur les performances du modèle.