Analyse complète des fichiers de code d'apprentissage fédéré
Voici une analyse détaillée de chaque fichier de code que vous avez partagé, qui ensemble constituent une implémentation complète d'apprentissage fédéré avec différentes variantes algorithmiques.
1. main_experimentation.py (Script principal)
Ce fichier contient le script principal qui exécute et compare différentes approches d'apprentissage fédéré sur le dataset MNIST.
Fonctions principales:
* run_federated_learning(): Exécute une instance complète d'apprentissage fédéré avec FedAvg ou FedProx.
* compare_fedavg_fedprox(): Compare les performances de FedAvg et FedProx avec différentes valeurs du paramètre μ.
* experiment_non_iid_effect(): Analyse l'impact de différents niveaux de non-IID sur les performances.
* run_experiments(): Exécute une série d'expériences avec différentes configurations.
* plot_comparison(), plot_non_iid_comparison(), plot_experiment_results(): Fonctions pour visualiser les résultats.
Ce script permet d'évaluer l'impact de nombreux facteurs sur les performances de l'apprentissage fédéré:
* Nombre de clients (edges)
* Distribution des données (IID vs non-IID)
* Algorithmes d'agrégation (FedAvg vs FedProx)
* Nombre d'époques locales
* Valeur du paramètre μ pour FedProx
2. run_adaptive_fedprox (Script pour μ adaptatif)
Ce fichier implémente et teste l'approche de FedProx avec un paramètre μ adaptatif.
Fonctions principales:
* run_federated_learning_adaptive_mu(): Exécute FedProx avec μ adaptatif.
* run_federated_learning_standard(): Version standard de FedProx pour comparaison.
* compare_fedavg_fedprox_adaptive(): Compare FedAvg, FedProx standard et FedProx adaptatif.
* plot_comparison_with_adaptive(): Visualise les performances comparatives.
Cette implémentation ajuste dynamiquement le paramètre μ en fonction de la distribution des classes et de la progression de l'entraînement.
3. run_comparaison_with_fedsgv (Comparaison avec FedSGD)
Ce script étend l'analyse en incluant FedSGD dans la comparaison.
Fonctions principales:
* compare_all_methods(): Compare FedSGD, FedAvg, FedProx standard et FedProx adaptatif.
* plot_comparison_with_fedsgd(): Visualise les performances de toutes les méthodes, y compris les gains relatifs.
Ce fichier met en évidence les différences entre les quatre approches principales d'apprentissage fédéré et évalue leur efficacité relative.
4. adaptive_edge_node (Classe AdaptiveEdgeNode)
Ce fichier définit la classe AdaptiveEdgeNode qui étend le nœud edge standard pour supporter le paramètre μ adaptatif.
Fonctionnalités principales:
* Chargement et gestion des données locales
* Calcul de la distribution des classes
* Entraînement du modèle local avec adaptation du paramètre μ
* Support pour le terme de régularisation proximale
Cette classe permet aux edges de calculer et d'utiliser des valeurs de μ personnalisées en fonction de leurs caractéristiques de données spécifiques.
5. adaptive_federated_server (Classe AdaptiveFederatedServer)
Ce fichier implémente le serveur central avec support pour le calcul et la distribution des valeurs de μ adaptatives.
Fonctionnalités principales:
* Agrégation des poids selon FedAvg
* Calcul des valeurs de μ adaptatives pour chaque client
* Suivi de la distribution globale des classes
* Évaluation du modèle global
Le serveur analyse les distributions de classes des edges et calcule des valeurs de μ optimisées pour chaque client.
6. adaptive_mu_calculator (Module AdaptiveMuCalculator)
Ce module contient les algorithmes spécifiques pour calculer dynamiquement le paramètre μ.
Fonctionnalités principales:
* Calcul de la divergence entre distributions locales et globales
* Adaptation de μ en fonction de multiples facteurs:
    * Hétérogénéité des données
    * Progression de l'entraînement
    * Amélioration des performances
* Calcul de valeurs de μ individualisées pour chaque client
Ce module implémente la logique d'adaptation du paramètre μ, prenant en compte les caractéristiques des données, l'avancement de l'entraînement et les performances actuelles.
7. (Implémentation FedAvg de base)
Ce fichier contient l'implémentation de base de FedAvg avec les classes EdgeNode et FederatedServer.
Fonctionnalités principales:
* Définition de la structure fondamentale pour l'apprentissage fédéré
* Chargement et distribution des données par les edges
* Agrégation pondérée des modèles locaux
* Comparaison avec l'apprentissage centralisé
Cette implémentation représente la version de référence de l'apprentissage fédéré sur laquelle les autres approches sont construites.
8. edge_node.py (Classe EdgeNode)
Ce fichier définit la classe EdgeNode standard qui est utilisée dans l'implémentation de FedAvg et FedProx.
Fonctionnalités principales:
* Chargement et prétraitement des données locales
* Entraînement du modèle local
* Support pour FedProx avec μ fixe
* Comptage et suivi des échantillons
Cette classe représente un participant individuel dans le système d'apprentissage fédéré.
9. (Implémentation FedAvg avec fonctions auxiliaires)
Ce fichier contient une implémentation de FedAvg avec des fonctions auxiliaires pour la gestion des poids et la distribution des données.
Fonctions principales:
* make_edges(): Répartit les données entre les edges
* weight_scaling_factor(): Calcule les facteurs de pondération
* scale_model_weights(): Applique les facteurs de pondération aux poids
* sum_scaled_weights(): Agrège les poids pondérés
Cette implémentation fournit des détails supplémentaires sur la répartition des données et le calcul des pondérations.
10. (Implémentation FedAvg améliorée)
Ce fichier présente une version améliorée de l'apprentissage fédéré avec diverses optimisations.
Fonctionnalités améliorées:
* Distribution stratégique des données
* Taux d'apprentissage adaptatif par edge
* Implémentation de FedProx avec μ variable
* Techniques de régularisation
* Augmentation de données
* Détection et filtrage des contributeurs anormaux
* Utilisation du momentum pour stabiliser l'apprentissage
Cette implémentation représente une version avancée qui combine plusieurs techniques pour améliorer les performances.
Modules auxiliaires:
* FederatedServer: Gère l'agrégation des modèles et l'évaluation du modèle global.
* fl_dataquest: Charge et prétraite les données MNIST.
* fl_model: Définit l'architecture du modèle utilisé par tous les participants.
* FedProxModel: Étend le modèle de base avec le terme de régularisation proximale.
Analyse globale:
L'ensemble de ces fichiers constitue un framework complet pour expérimenter avec différentes approches d'apprentissage fédéré. Les implémentations vont du simple FedAvg à des approches plus sophistiquées comme FedProx avec μ adaptatif, permettant une analyse approfondie des différentes stratégies d'apprentissage fédéré dans divers scénarios (distribution IID/non-IID, nombre variable de clients, différentes configurations d'entraînement).
La structure modulaire permet d'isoler et d'étudier l'impact de chaque composant du système d'apprentissage fédéré, tandis que les fonctions de visualisation facilitent l'interprétation des résultats.
