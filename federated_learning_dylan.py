"""
Federated Learning amélioré avec approvisionnement des données par les edges.

Cette implémentation optimisée vise à améliorer les performances du modèle fédéré
en introduisant plusieurs techniques avancées:
1. Distribution stratégique des données entre les edges
2. Personnalisation du taux d'apprentissage par edge
3. Implémentation de FedProx au lieu de FedAvg basique
4. Techniques de régularisation pour réduire l'overfitting sur les données locales
5. Augmentation du nombre de rounds et d'époques locales
"""

import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import fl_dataquest
import fl_model

class EdgeNode:
    """
    Représente un nœud edge optimisé dans le système d'apprentissage fédéré.
    """
    
    def __init__(self, edge_id, base_path, data_folders=None, verbose=0, mu=0.01):
        """
        Initialise un nœud edge avec des paramètres améliorés.
        
        Args:
            edge_id: Identifiant unique de l'edge
            base_path: Chemin de base vers les données
            data_folders: Liste des dossiers à charger (si None, utilise un sous-ensemble stratégique)
            verbose: Niveau de verbosité
            mu: Paramètre de régularisation pour FedProx
        """
        self.edge_id = edge_id
        self.base_path = base_path
        self.data_folders = data_folders
        self.verbose = verbose
        self.model = None
        self.dataset = None
        self.sample_count = 0
        self.mu = mu  # Paramètre de proximité pour FedProx
        self.learning_rate = 0.01  # Taux d'apprentissage initial personnalisable
        
    def load_data(self, num_classes=10, folders_per_edge=None, augment=False):
        """
        Charge les données pour cet edge avec une stratégie optimisée.
        
        Args:
            num_classes: Nombre total de classes
            folders_per_edge: Nombre de dossiers (classes) que cet edge doit charger
            augment: Activer l'augmentation de données pour améliorer la généralisation
        """
        if self.verbose > 0:
            print(f"Edge {self.edge_id} commence à charger ses données...")
        
        # Stratégie de sélection de données plus élaborée
        if self.data_folders is None:
            if folders_per_edge is None:
                # Garantir un minimum de diversité tout en permettant la spécialisation
                folders_per_edge = max(3, random.randint(3, num_classes))
                
            # Distribution stratégique pour assurer une meilleure couverture globale
            all_folders = [str(i) for i in range(num_classes)]
            
            # Stratégie basée sur l'ID de l'edge pour une distribution plus équilibrée
            # Les edges avec ID pair prennent plutôt des chiffres pairs, impairs pour impairs
            if int(self.edge_id.split('_')[-1]) % 2 == 0:
                priority_folders = [str(i) for i in range(num_classes) if i % 2 == 0]
                secondary_folders = [str(i) for i in range(num_classes) if i % 2 != 0]
            else:
                priority_folders = [str(i) for i in range(num_classes) if i % 2 != 0]
                secondary_folders = [str(i) for i in range(num_classes) if i % 2 == 0]
            
            # Combiner les listes avec priorité
            combined_folders = priority_folders + secondary_folders
            self.data_folders = combined_folders[:folders_per_edge]
            
        if self.verbose > 0:
            print(f"Edge {self.edge_id} va charger les classes: {self.data_folders}")
        
        # Collecter tous les chemins d'images et les labels correspondants
        all_image_paths = []
        all_labels = []
        
        for folder in self.data_folders:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                # Obtenir tous les fichiers d'images dans ce dossier
                image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                               if os.path.isfile(os.path.join(folder_path, f))]
                all_image_paths.extend(image_files)
                # Stocker directement les labels (folder contient déjà le chiffre)
                all_labels.extend([int(folder)] * len(image_files))
        
        # Mélanger les chemins d'images et les labels ensemble
        combined = list(zip(all_image_paths, all_labels))
        random.shuffle(combined)
        all_image_paths, all_labels = zip(*combined) if combined else ([], [])
        
        # Charger et prétraiter les images
        data, _ = fl_dataquest.load_and_preprocess(all_image_paths, verbose=self.verbose if self.verbose > 1 else None)
        
        # Convertir les labels en format one-hot
        lb = LabelBinarizer()
        lb.fit(range(num_classes))  # S'assurer que toutes les classes sont représentées
        labels_encoded = lb.transform(all_labels)
        
        # Créer un dataset TensorFlow avec prefetch pour optimiser les performances
        self.dataset = tf.data.Dataset.from_tensor_slices((data, labels_encoded))
        self.dataset = self.dataset.shuffle(buffer_size=len(all_labels))
        
        # Augmentation de données si demandée
        if augment:
            self.dataset = self.dataset.map(self._augment_data, 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Optimisation des performances
        self.dataset = self.dataset.batch(32)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        self.sample_count = len(all_labels)
        
        if self.verbose > 0:
            print(f"Edge {self.edge_id} a chargé {self.sample_count} échantillons")
            
        return self.dataset, self.sample_count
    
    def _augment_data(self, image, label):
        angle = tf.random.uniform([], -0.1, 0.1)
        image = tf.cast(image, tf.float32)
        
        image = tf.image.stateless_random_crop(
            tf.pad(image, [[2, 2], [2, 2], [0, 0]]),
            size=[28, 28, 1],
            seed=[42, 42]
        )
        
        # Variation de contraste
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Reconvertir au type original si nécessaire
        image = tf.cast(image, tf.float64)
        
        return image, label
    
    def train_model(self, global_model_weights, input_shape, num_classes=10, epochs=1, adaptive_lr=True):
        """
        Entraîne un modèle local en utilisant FedProx pour limiter la divergence.
        
        Args:
            global_model_weights: Poids du modèle global à utiliser pour initialiser le modèle local
            input_shape: Forme des données d'entrée
            num_classes: Nombre de classes pour la classification
            epochs: Nombre d'époques d'entraînement
            adaptive_lr: Utiliser un taux d'apprentissage adaptatif
            
        Returns:
            Les poids du modèle local entraîné
        """
        if self.dataset is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
        
        # Initialiser le modèle local avec les poids du modèle global
        self.model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        
        # Sauvegarder les poids globaux avant l'entraînement pour FedProx
        global_weights = global_model_weights.copy()
        self.model.set_weights(global_weights)
        
        # Définir un taux d'apprentissage adaptatif pour chaque edge en fonction de la quantité de données
        if adaptive_lr:
            if self.sample_count < 1000:
                self.learning_rate = 0.005  # Taux plus bas pour les petits datasets (moins stable)
            elif self.sample_count > 5000:
                self.learning_rate = 0.015  # Taux plus élevé pour les grands datasets (plus stable)
        
        # Compiler le modèle avec le taux d'apprentissage optimisé
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.model.model.compile(optimizer=optimizer, 
                               loss='categorical_crossentropy', 
                               metrics=['accuracy'])
        
        # Créer des callbacks pour améliorer l'entraînement
        callbacks = []
        if epochs > 1:
            # Early stopping pour éviter l'overfitting
            early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            # Réduction du taux d'apprentissage sur plateau
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=0.001)
            callbacks = [early_stopping, reduce_lr]
        
        # Entraîner le modèle local avec FedProx (proximal term)
        # Implémentation personnalisée de FedProx
        
        # D'abord, entraînement régulier
        self.model.fit_it(trains=self.dataset, epochs=epochs, tests=None, 
                 verbose=self.verbose)
        
        # Puis, ajustement des poids selon FedProx pour limiter la divergence
        if self.mu > 0:
            current_weights = self.model.get_weights()
            proximal_weights = []
            
            # Appliquer régularisation FedProx: pénaliser les poids qui s'éloignent trop du modèle global
            for i in range(len(current_weights)):
                # Terme proximal: tiré les poids locaux vers les poids globaux
                w_proximal = current_weights[i] - self.mu * (current_weights[i] - global_weights[i])
                proximal_weights.append(w_proximal)
            
            # Mettre à jour les poids avec le terme proximal
            self.model.set_weights(proximal_weights)
        
        # Retourner les poids du modèle local
        return self.model.get_weights()
    
    def get_sample_count(self):
        """
        Retourne le nombre d'échantillons dans le dataset de cet edge.
        """
        return self.sample_count


class FederatedServer:
    """
    Représente le serveur central amélioré dans le système d'apprentissage fédéré.
    """
    
    def __init__(self, input_shape, num_classes=10, verbose=0):
        """
        Initialise le serveur fédéré avec des fonctionnalités améliorées.
        
        Args:
            input_shape: Forme des données d'entrée
            num_classes: Nombre de classes pour la classification
            verbose: Niveau de verbosité
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.verbose = verbose
        self.global_model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        
        # Pour les évaluations
        self.test_dataset = None
        
        # Historique complet des poids pour détecter les anomalies
        self.weights_history = []
        
        # Pour détecter et gérer les clients malveillants ou défaillants
        self.edge_performance_history = {}
        
    def set_test_data(self, test_dataset):
        """
        Définit les données de test pour l'évaluation du modèle global.
        
        Args:
            test_dataset: Dataset de test
        """
        self.test_dataset = test_dataset
        
    def initialize_model(self):
        """
        Initialise le modèle global et retourne ses poids.
        
        Returns:
            Les poids du modèle global initialisé
        """
        return self.global_model.get_weights()
    
    def aggregate_weights(self, edge_weights_list, edge_sample_counts, round_num=0, edge_ids=None):
        """
        Agrège les poids des modèles locaux avec une stratégie améliorée.
        
        Args:
            edge_weights_list: Liste des poids des modèles locaux
            edge_sample_counts: Liste du nombre d'échantillons par edge
            round_num: Numéro du round actuel pour pondération adaptative
            edge_ids: Liste des identifiants des edges pour le suivi des performances
            
        Returns:
            Les poids agrégés du modèle global
        """
        if len(edge_weights_list) != len(edge_sample_counts):
            raise ValueError("La liste des poids et la liste des nombres d'échantillons doivent avoir la même longueur")
        
        # Sauvegarder l'état actuel du modèle global
        prev_weights = self.global_model.get_weights()
        
        # Vérifier les contributeurs anormaux en utilisant une médiane ou autre métrique
        # Peut aider à détecter des clients potentiellement malveillants
        if edge_ids and round_num > 0:
            for i, edge_id in enumerate(edge_ids):
                if edge_id not in self.edge_performance_history:
                    self.edge_performance_history[edge_id] = []
                
                # Calculer un score de divergence (différence moyenne avec les poids précédents)
                divergence_score = np.mean([np.mean(np.abs(edge_weights_list[i][j] - prev_weights[j])) 
                                          for j in range(len(prev_weights))])
                
                self.edge_performance_history[edge_id].append(divergence_score)
        
        # Détecter et filtrer les contributions aberrantes
        if round_num > 1 and edge_ids:
            filtered_weights = []
            filtered_counts = []
            filtered_ids = []
            
            for i, edge_id in enumerate(edge_ids):
                # Calculer la moyenne mobile des divergences passées
                history = self.edge_performance_history.get(edge_id, [0])
                avg_divergence = np.mean(history)
                
                # Si la divergence est supérieure à un seuil (adaptatif), filtrer cet edge
                if len(history) > 1 and history[-1] > 3 * avg_divergence:
                    if self.verbose > 0:
                        print(f"Edge {edge_id} filtré pour contribution aberrante.")
                    continue
                
                filtered_weights.append(edge_weights_list[i])
                filtered_counts.append(edge_sample_counts[i])
                filtered_ids.append(edge_id)
            
            # Utiliser les listes filtrées si elles ne sont pas vides
            if filtered_weights:
                edge_weights_list = filtered_weights
                edge_sample_counts = filtered_counts
                edge_ids = filtered_ids
        
        # Calculer le nombre total d'échantillons
        total_samples = sum(edge_sample_counts)
        
        # Ajustement de la pondération des contributions en fonction du round
        # Au fur et à mesure des rounds, nous pouvons donner plus de poids aux clients ayant des performances
        # stables ou consistantes
        weight_factors = []
        for i, count in enumerate(edge_sample_counts):
            base_factor = count / total_samples
            
            # Après quelques rounds, ajuster la pondération en fonction des performances antérieures
            if round_num > 2 and edge_ids:
                edge_id = edge_ids[i]
                if edge_id in self.edge_performance_history and len(self.edge_performance_history[edge_id]) > 1:
                    # Moins de divergence = plus de confiance = plus de poids
                    stability_score = 1.0 / (1.0 + np.mean(self.edge_performance_history[edge_id][-2:]))
                    # Ajuster légèrement le facteur de base (max +/- 20%)
                    base_factor *= (0.8 + 0.4 * stability_score)
            
            weight_factors.append(base_factor)
        
        # Normaliser les facteurs
        sum_factors = sum(weight_factors)
        weight_factors = [f / sum_factors for f in weight_factors]
        
        # Calculer les poids agrégés pondérés par le nombre d'échantillons ajusté
        aggregated_weights = []
        
        # Pour chaque couche du modèle
        for layer_index in range(len(edge_weights_list[0])):
            layer_weights = []
            
            # Pour chaque edge
            for edge_index in range(len(edge_weights_list)):
                # Obtenir les poids de cette couche pour cet edge
                edge_layer_weights = edge_weights_list[edge_index][layer_index]
                
                # Pondérer ces poids par la proportion d'échantillons
                weight_factor = weight_factors[edge_index]
                weighted_weights = edge_layer_weights * weight_factor
                
                layer_weights.append(weighted_weights)
            
            # Sommer tous les poids pondérés pour cette couche
            aggregated_layer_weights = sum(layer_weights)
            aggregated_weights.append(aggregated_layer_weights)
        
        # Ajouter l'agrégation à l'historique des poids pour analyse ultérieure
        self.weights_history.append(aggregated_weights)
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights, momentum=0.1, round_num=0):
        """
        Met à jour le modèle global avec les poids agrégés et utilise un momentum.
        
        Args:
            aggregated_weights: Poids agrégés à utiliser pour mettre à jour le modèle global
            momentum: Facteur de momentum pour stabiliser l'apprentissage
            round_num: Numéro du round actuel
        """
        # Si c'est le premier round, pas de momentum
        if round_num == 0 or not self.weights_history or len(self.weights_history) <= 1:
            self.global_model.set_weights(aggregated_weights)
            return
        
        # Sinon, appliquer un momentum pour stabiliser l'apprentissage
        # Récupérer les anciens poids
        old_weights = self.global_model.get_weights()
        
        # Appliquer le momentum
        new_weights = []
        for i in range(len(aggregated_weights)):
            # Mélanger les nouveaux poids avec les anciens selon le facteur de momentum
            new_layer_weights = (1 - momentum) * aggregated_weights[i] + momentum * old_weights[i]
            new_weights.append(new_layer_weights)
        
        # Mettre à jour le modèle global
        self.global_model.set_weights(new_weights)
        
    def evaluate_global_model(self):
        """
        Évalue le modèle global sur le dataset de test.
        
        Returns:
            La perte et la précision du modèle global
        """
        if self.test_dataset is None:
            raise ValueError("Aucun dataset de test n'a été défini. Appelez set_test_data() d'abord.")
        
        return self.global_model.evaluate(self.test_dataset, verbose=self.verbose)


def run_federated_learning(mnist_base_path, test_dataset, num_edges=10, num_rounds=10, 
                          edge_epochs=3, input_shape=(28, 28), num_classes=10, verbose=1, 
                          use_fedprox=True, adaptive_aggregation=True, data_augmentation=True):
    """
    Exécute le processus d'apprentissage fédéré optimisé.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        test_dataset: Dataset de test pour l'évaluation
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes pour la classification
        verbose: Niveau de verbosité
        use_fedprox: Utiliser FedProx au lieu de FedAvg
        adaptive_aggregation: Utiliser une agrégation adaptative
        data_augmentation: Utiliser l'augmentation de données
        
    Returns:
        L'historique des évaluations du modèle global
    """
    # Initialiser le serveur
    server = FederatedServer(input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges avec distribution stratégique
    edges = []
    for i in range(num_edges):
        # Paramètre mu pour FedProx, 0 équivaut à FedAvg standard
        mu = 0.01 if use_fedprox else 0.0
        edge = EdgeNode(f"edge_{i}", mnist_base_path, verbose=verbose, mu=mu)
        edges.append(edge)
    
    # Charger les données pour chaque edge
    # Stratégie de distribution améliorée
    min_classes = 3  # Au moins 3 classes par edge pour assurer diversité
    for edge in edges:
        # Pour les premiers edges, garantir une bonne couverture des classes
        if int(edge.edge_id.split('_')[-1]) < num_classes:
            # S'assurer que chaque classe apparaît au moins dans un edge
            guaranteed_class = int(edge.edge_id.split('_')[-1]) % num_classes
            folders_to_load = [str(guaranteed_class)]
            
            # Compléter avec des classes aléatoires
            remaining_classes = [str(i) for i in range(num_classes) if i != guaranteed_class]
            folders_to_load.extend(random.sample(remaining_classes, min_classes-1))
            
            edge.data_folders = folders_to_load
        
        # Charger avec ou sans augmentation de données
        edge.load_data(num_classes, augment=data_augmentation)
    
    # Vérifier la distribution des classes sur les edges
    if verbose > 0:
        class_coverage = {str(i): 0 for i in range(num_classes)}
        for edge in edges:
            for folder in edge.data_folders:
                class_coverage[folder] += 1
        print("Distribution des classes sur les edges:")
        for cls, count in class_coverage.items():
            print(f"  Classe {cls}: {count} edges")
    
    # Historique des évaluations
    evaluation_history = []
    
    # Facteur de momentum pour stabiliser l'apprentissage
    momentum = 0.1
    
    # Exécuter les rounds de fédération
    for round_num in range(num_rounds):
        if verbose > 0:
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Obtenir les poids du modèle global actuel
        global_weights = server.initialize_model() if round_num == 0 else server.global_model.get_weights()
        
        # Entraîner chaque edge avec les poids globaux actuels
        edge_weights_list = []
        edge_sample_counts = []
        edge_ids = []
        
        for edge in edges:
            # Ajuster le nombre d'époques en fonction du round
            # Plus de rounds = plus d'époques pour peaufiner
            current_epochs = edge_epochs
            if round_num > num_rounds // 2:
                current_epochs += 1  # Augmenter le nombre d'époques dans les derniers rounds
            
            # Entraîner le modèle local de cet edge
            edge_weights = edge.train_model(global_weights, input_shape, num_classes, 
                                           current_epochs, adaptive_lr=True)
            edge_weights_list.append(edge_weights)
            edge_sample_counts.append(edge.get_sample_count())
            edge_ids.append(edge.edge_id)
        
        # Agréger les poids des modèles locaux avec stratégie adaptative
        if adaptive_aggregation:
            aggregated_weights = server.aggregate_weights(edge_weights_list, edge_sample_counts, 
                                                        round_num, edge_ids)
        else:
            aggregated_weights = server.aggregate_weights(edge_weights_list, edge_sample_counts)
        
        # Mettre à jour le modèle global avec momentum
        server.update_global_model(aggregated_weights, momentum, round_num)
        
        # Évaluer le modèle global
        loss, accuracy = server.evaluate_global_model()
        evaluation_history.append((loss, accuracy))
        
        if verbose > 0:
            print(f"Round {round_num + 1} - Perte: {loss:.4f}, Précision: {accuracy:.4f}")
        
        # Ajuster le momentum en fonction des performances
        # Réduire le momentum si l'apprentissage est stable
        if round_num > 0:
            prev_loss = evaluation_history[-2][0]
            if loss < prev_loss:
                # Apprentissage stable, réduire le momentum
                momentum = max(0.05, momentum * 0.9)
            else:
                # Apprentissage instable, augmenter le momentum
                momentum = min(0.3, momentum * 1.1)
    
    return evaluation_history


def compare_with_centralized(mnist_base_path, num_edges=10, num_rounds=10, edge_epochs=3, verbose=1, data_augmentation=False):
    """
    Compare l'apprentissage fédéré optimisé avec l'approche centralisée traditionnelle.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        verbose: Niveau de verbosité
    """
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer seulement le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Configuration optimisée
    use_fedprox = True          # Utiliser FedProx pour une meilleure convergence
    adaptive_aggregation = True  # Agrégation adaptative des poids
    # data_augmentation = True     # Utiliser l'augmentation de données
    
    # Exécuter l'apprentissage fédéré optimisé
    fed_history = run_federated_learning(
        mnist_base_path, 
        test_dataset, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        input_shape=input_shape, 
        verbose=verbose,
        use_fedprox=use_fedprox,
        adaptive_aggregation=adaptive_aggregation,
        data_augmentation=data_augmentation
    )
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    X_train, X_test, y_train, y_test, _ = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    train_dataset, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Créer des callbacks pour le modèle centralisé
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    
    central_model = fl_model.MyModel(input_shape, nbclasses=10)
    # Utiliser le même nombre total d'époques
    central_model.fit_it(trains=train_dataset, epochs=num_rounds * edge_epochs, 
                        tests=test_dataset, verbose=verbose)
    
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Extraire les résultats finaux d'apprentissage fédéré
    final_fed_loss, final_fed_accuracy = fed_history[-1]
    print(f"Modèle fédéré optimisé - Perte: {final_fed_loss:.4f}, Précision: {final_fed_accuracy:.4f}")
    
    # Tracer les résultats avec amélioration de la visualisation
    plt.figure(figsize=(14, 6))
    
    # Sous-graphique pour la perte
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in fed_history], 'bo-', linewidth=2, label='Federated Loss')
    plt.axhline(y=central_loss, color='r', linestyle='-', linewidth=2, label='Centralized Loss')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Sous-graphique pour la précision
    plt.subplot(1, 2, 2)
    plt.plot([x[1] for x in fed_history], 'go-', linewidth=2, label='Federated Accuracy')
    plt.axhline(y=central_accuracy, color='r', linestyle='-', linewidth=2, label='Centralized Accuracy')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./ressource_image/federated_vs_centralized_improved.png', dpi=300)
    plt.show()
    
    # Analyse supplémentaire des résultats
    print("\n--- Analyse des résultats ---")
    
    # Calcul du taux d'amélioration
    initial_accuracy = fed_history[0][1]
    final_accuracy = fed_history[-1][1]
    improvement_rate = (final_accuracy - initial_accuracy) / initial_accuracy * 100
    
    print(f"Taux d'amélioration du modèle fédéré: {improvement_rate:.2f}%")
    print(f"Écart final avec le modèle centralisé: {(central_accuracy - final_accuracy) * 100:.2f} points de pourcentage")
    
    # Estimation du nombre de rounds nécessaires pour atteindre la performance centralisée
    if final_accuracy < central_accuracy and len(fed_history) > 2:
        # Calculer la tendance des derniers rounds
        recent_improvements = [fed_history[i][1] - fed_history[i-1][1] for i in range(-3, 0)]
        avg_improvement_per_round = sum(recent_improvements) / len(recent_improvements)
        
        if avg_improvement_per_round > 0:
            rounds_needed = (central_accuracy - final_accuracy) / avg_improvement_per_round
            print(f"Estimation de rounds supplémentaires pour atteindre la performance centralisée: {int(rounds_needed) + 1}")
    
    return fed_history, (central_loss, central_accuracy)


if __name__ == '__main__':
    # Définir le chemin de base vers les données MNIST
    mnist_base_path = 'C:/Users/dylan/Downloads/archive/trainingSet/trainingSet'
    
    # Exécuter la comparaison avec paramètres optimisés
    compare_with_centralized(
        mnist_base_path, 
        num_edges=10,       # Nombre d'edges
        num_rounds=10,      # Plus de rounds pour une meilleure convergence
        edge_epochs=3,      # Plus d'époques par edge pour améliorer l'apprentissage local
        verbose=1,
        data_augmentation=False
    )