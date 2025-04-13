# Apprentissage Fédéré : Analyse Comparative de FedAvg, FedProx et FedProx Adaptatif
# Introduction
Ce projet implémente et compare trois approches d'apprentissage fédéré pour la classification d'images sur le dataset MNIST: FedAvg (Federated Averaging), FedProx (avec μ fixe), et FedProx avec μ adaptatif. L'objectif est d'analyser les performances de ces algorithmes dans un environnement où les données sont distribuées de manière non-IID (non identiquement distribuées) entre plusieurs clients, ce qui représente un scénario réaliste pour l'apprentissage fédéré.
L'apprentissage fédéré est une approche de machine learning qui permet d'entraîner un modèle global à partir de données localisées sur différents appareils (clients), sans nécessiter le transfert des données vers un serveur central. Cette approche présente des avantages en termes de confidentialité des données et de réduction de la bande passante nécessaire.
Méthodes implémentées
1. FedAvg (Federated Averaging)
L'algorithme de base où chaque client entraîne un modèle local sur ses données, puis le serveur agrège ces modèles en prenant une moyenne pondérée des poids. Cette méthode, bien que simple, peut souffrir de divergence lorsque les données sont non-IID.
2. FedProx (μ fixe = 0.1)
Une extension de FedAvg qui ajoute un terme de régularisation proximale pour limiter la divergence entre les modèles locaux et le modèle global. Un paramètre μ fixe contrôle l'intensité de cette régularisation, aidant à maintenir la stabilité de l'entraînement même avec des données non-IID.
3. FedProx avec μ adaptatif
Notre contribution principale est une version améliorée de FedProx où le paramètre μ est adapté dynamiquement pour chaque client en fonction de:
* La divergence entre sa distribution locale et la distribution globale des classes
* L'évolution des performances du modèle global
* La progression de l'entraînement (round actuel)
Configuration expérimentale
* Dataset: MNIST (chiffres manuscrits)
* Distribution: Non-IID avec 2 classes par client (distribution de type silo)
* Nombre de clients: 20
* Nombre de rounds: 30
* Époques locales par round: 2
* Fraction de clients par round: 80% (sélection aléatoire)
* Partitionnement train/test: 80%/20%
* Modèle: Réseau de neurones avec deux couches cachées (200 et 100 neurones)
Analyse des résultats

### Graphique 1: Évolution du μ adaptatif et de la précision
￼
Ce graphique montre la relation entre l'évolution du paramètre μ adaptatif (axe gauche, en bleu) et la précision du modèle (axe droit, en rouge) au fil des rounds. On observe que:
* La valeur moyenne de μ commence à environ 0.01, diminue dans les premiers rounds jusqu'à environ 0.006, puis augmente progressivement pour atteindre des valeurs plus élevées (jusqu'à 0.015) vers la fin de l'entraînement.
* La précision augmente globalement de manière constante, avec quelques fluctuations, atteignant environ 0.75 (75%) après 30 rounds.
* On note une corrélation intéressante: quand μ augmente, la précision a tendance à s'améliorer également, ce qui suggère que l'adaptation du paramètre μ contribue positivement à l'apprentissage.
Graphique 2: Comparaison de la perte
￼
Ce graphique compare l'évolution de la fonction de perte pour les trois algorithmes:
* FedAvg (bleu) montre la diminution de perte la plus rapide et atteint la valeur la plus basse (environ 0.47).
* FedProx standard (orange) diminue plus lentement mais de manière constante, terminant à environ 0.58.
* FedProx adaptatif (vert) présente un comportement plus irrégulier: il commence à diminuer rapidement, puis se stabilise autour d'une valeur de 0.9, révélant un compromis différent entre minimisation de la perte et généralisation.
Graphique 3: Comparaison de l'écart avec le modèle centralisé
￼
### Ce graphique montre l'écart relatif (en pourcentage) entre les performances des algorithmes fédérés et celles du modèle centralisé entraîné sur toutes les données:
* FedAvg termine avec le plus petit écart (12.2%), montrant qu'il peut s'approcher le plus des performances du modèle centralisé malgré la distribution non-IID.
* FedProx standard affiche un écart de 18.6%.
* FedProx adaptatif montre un écart de 23.7%, mais présente un comportement intéressant en début d'entraînement où il réduit l'écart plus rapidement que FedProx standard.
Graphique 4: Évolution du paramètre μ adaptatif
￼
### Ce graphique détaille l'évolution du paramètre μ adaptatif:
* La valeur moyenne de μ (ligne bleue) présente des fluctuations importantes, suggérant que l'algorithme s'adapte aux conditions changeantes de l'entraînement.
* La zone ombrée bleue représente la plage des valeurs de μ pour différents clients, montrant que le paramètre est personnalisé pour chaque client.
* On observe une tendance générale à l'augmentation de μ à mesure que l'entraînement progresse, avec une croissance plus marquée après le round 20.
Graphique 5: Comparaison de la précision
￼
### Ce graphique compare directement la précision des trois algorithmes par rapport au modèle centralisé:
* FedAvg (bleu) surpasse les autres méthodes, atteignant une précision d'environ 0.85 après 30 rounds.
* FedProx standard (orange) atteint environ 0.79.
* FedProx adaptatif (vert) montre des performances plus variables, avec une précision finale d'environ 0.74.
* Le modèle centralisé (ligne rouge pointillée) maintient une précision proche de 0.97, représentant la limite supérieure théorique.
Conclusion
Cette étude comparative révèle plusieurs enseignements importants sur l'apprentissage fédéré dans un contexte non-IID:
# Conclusion
1. Performances relatives: Contre-intuitivement, FedAvg surpasse FedProx (standard et adaptatif) en termes de précision finale et de minimisation de la perte dans notre configuration. Cela peut s'expliquer par le nombre relativement faible de clients (5) et la distribution spécifique des données.
2. Dynamique d'apprentissage: FedProx adaptatif montre une dynamique d'apprentissage distincte, avec des performances initiales souvent supérieures à FedProx standard, mais une stabilisation à un niveau inférieur. Cette caractéristique pourrait être avantageuse dans des scénarios où une convergence rapide est prioritaire.
3. Comportement du paramètre μ: L'évolution du paramètre μ adaptatif montre une tendance intéressante - il commence bas, permettant plus d'exploration, puis augmente progressivement pour favoriser la stabilité. Cette adaptation semble correspondre aux besoins changeants du processus d'apprentissage.
4. Écart avec le modèle centralisé: Tous les algorithmes fédérés maintiennent un écart significatif avec le modèle centralisé, illustrant le défi fondamental de l'apprentissage fédéré: obtenir des performances comparables sans centraliser les données.
5. Implications pratiques: Le choix entre FedAvg et FedProx dépend du contexte spécifique. FedAvg semble plus performant pour un petit nombre de clients avec des époques locales limitées, tandis que FedProx pourrait être plus adapté à des scénarios avec plus de clients et/ou des données plus hétérogènes.
Pour les travaux futurs, il serait intéressant d'explorer:
* L'impact d'un nombre plus élevé de clients (10-100)
* Des distributions non-IID plus extrêmes
* Des stratégies alternatives d'adaptation du paramètre μ
* L'application à des datasets plus complexes (CIFAR-10, ImageNet)
* L'intégration de mécanismes de confidentialité différentielle
Cette étude démontre que l'apprentissage fédéré peut atteindre des performances raisonnables même avec des données distribuées de manière non uniforme, tout en préservant la confidentialité des données locales.
