# ComparaisonFedAvgvsFedProx.png

 ## Analyse comparative de FedAvg et FedProx sur MNIST
Description du graphique
Ce graphique présente une comparaison de la précision entre les algorithmes d'apprentissage fédéré FedAvg et FedProx (avec différentes valeurs du paramètre μ) sur le jeu de données MNIST.
Éléments clés du graphique

* Axe X : Nombre de rounds d'apprentissage fédéré (de 0 à 30)
Axe Y : Précision du modèle (de 0 à 1)
Courbes :

* Bleu : FedAvg (algorithme standard d'agrégation fédérée)
Orange : FedProx avec μ=0.01 (régularisation proximale faible)
Vert : FedProx avec μ=0.1 (régularisation proximale plus forte)
Rouge pointillé : Modèle centralisé (référence)



## Résultats principaux

Performance finale : FedProx avec μ=0.1 (vert) atteint la meilleure précision finale après 30 rounds (~0.7).
Convergence initiale : FedAvg (bleu) démarre avec une convergence plus rapide dans les premiers rounds.
Stabilité : FedProx avec μ=0.1 semble plus stable et continue de progresser régulièrement.
Écart avec le modèle centralisé : Un écart significatif persiste entre les approches fédérées et le modèle centralisé (~0.97), ce qui est normal en raison de la nature distribuée de l'apprentissage.

## Implications

Avantage de FedProx : Ce graphique démontre l'efficacité de la régularisation proximale avec μ=0.1 pour améliorer la convergence à long terme sur des données distribuées.
Compromis : FedProx avec μ=0.01 se montre moins performant que les deux autres approches, suggérant que ce niveau de régularisation est insuffisant pour les conditions expérimentales.
Distribution des données : Ce test a été réalisé avec une distribution non-IID des données (2 classes par client), représentant un scénario réaliste d'apprentissage fédéré.

Conclusion
FedProx avec une valeur appropriée de μ (0.1 dans cette expérience) offre un avantage significatif par rapport à FedAvg dans un contexte d'apprentissage fédéré avec données non-IID, particulièrement pour les phases avancées de l'entraînement. Ce résultat confirme l'hypothèse que l'ajout d'un terme de régularisation proximale peut améliorer la convergence des modèles dans un contexte fédéré hétérogène.

## Comparaison de la perte: FedAvg vs FedProx

Axe vertical: Mesure la valeur de la perte (loss), qui s'étend d'environ 0.85 à 2.50. Une valeur plus basse indique une meilleure performance du modèle.
Axe horizontal: Représente les 30 rounds d'entraînement.
Courbes représentées:

FedAvg (courbe bleue): Commence à environ 2.33, augmente légèrement jusqu'à un pic d'environ 2.48 aux rounds 2-4, puis diminue jusqu'au round 15 où elle se stabilise autour de 2.10 pour le reste de l'entraînement.
FedProx (μ=0.01) (courbe orange): Démarre avec la valeur la plus élevée (2.50), puis diminue constamment et rapidement tout au long des 30 rounds, atteignant la valeur la plus basse à la fin (environ 0.85).
FedProx (μ=0.1) (courbe verte): Commence avec la valeur la plus basse des trois (environ 2.15), diminue de façon constante et plus rapide que FedAvg jusqu'au round 20, puis continue à diminuer mais plus lentement, terminant à environ 0.90.


Observations clés:

Les deux variantes de FedProx surpassent significativement FedAvg en termes de réduction de la perte.
FedProx avec μ=0.01 commence moins bien mais finit par obtenir la meilleure performance finale.
FedProx avec μ=0.1 a une performance initiale meilleure mais est légèrement dépassé par la version μ=0.01 vers la fin de l'entraînement.
FedAvg atteint un plateau assez rapidement (vers le round 15) et ne montre pratiquement aucune amélioration par la suite.
Les deux variantes de FedProx continuent de s'améliorer même après 30 rounds.


Particularité intéressante: Les courbes de FedProx avec μ=0.01 et μ=0.1 se croisent au round 21-22, suggérant que différentes valeurs de μ peuvent être optimales à différentes étapes de l'entraînement.

Ce graphique démontre clairement l'avantage d'utiliser FedProx sur FedAvg pour minimiser la fonction de perte. Il suggère également qu'un paramètre μ plus faible (0.01) peut conduire à une meilleure convergence à long terme, tandis qu'un μ plus élevé (0.1) peut offrir de meilleures performances initiales.