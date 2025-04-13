Analyse comparative de FedAvg et FedProx sur MNIST
Description du graphique
Ce graphique présente une comparaison de la précision entre les algorithmes d'apprentissage fédéré FedAvg et FedProx (avec différentes valeurs du paramètre μ) sur le jeu de données MNIST.
Éléments clés du graphique

Axe X : Nombre de rounds d'apprentissage fédéré (de 0 à 30)
Axe Y : Précision du modèle (de 0 à 1)
Courbes :

Bleu : FedAvg (algorithme standard d'agrégation fédérée)
Orange : FedProx avec μ=0.01 (régularisation proximale faible)
Vert : FedProx avec μ=0.1 (régularisation proximale plus forte)
Rouge pointillé : Modèle centralisé (référence)



Résultats principaux

Performance finale : FedProx avec μ=0.1 (vert) atteint la meilleure précision finale après 30 rounds (~0.7).
Convergence initiale : FedAvg (bleu) démarre avec une convergence plus rapide dans les premiers rounds.
Stabilité : FedProx avec μ=0.1 semble plus stable et continue de progresser régulièrement.
Écart avec le modèle centralisé : Un écart significatif persiste entre les approches fédérées et le modèle centralisé (~0.97), ce qui est normal en raison de la nature distribuée de l'apprentissage.

Implications

Avantage de FedProx : Ce graphique démontre l'efficacité de la régularisation proximale avec μ=0.1 pour améliorer la convergence à long terme sur des données distribuées.
Compromis : FedProx avec μ=0.01 se montre moins performant que les deux autres approches, suggérant que ce niveau de régularisation est insuffisant pour les conditions expérimentales.
Distribution des données : Ce test a été réalisé avec une distribution non-IID des données (2 classes par client), représentant un scénario réaliste d'apprentissage fédéré.

Conclusion
FedProx avec une valeur appropriée de μ (0.1 dans cette expérience) offre un avantage significatif par rapport à FedAvg dans un contexte d'apprentissage fédéré avec données non-IID, particulièrement pour les phases avancées de l'entraînement. Ce résultat confirme l'hypothèse que l'ajout d'un terme de régularisation proximale peut améliorer la convergence des modèles dans un contexte fédéré hétérogène.