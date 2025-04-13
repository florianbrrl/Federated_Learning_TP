"""
Federated Learning avec approvisionnement des données par les edges.

Dans cette implémentation, chaque edge est responsable de charger ses propres données,
au lieu d'avoir une distribution centralisée des données.

Cette approche est plus fidèle à un scénario FL réel où les données restent sur les appareils
et ne sont jamais centralisées.
"""

import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import fl_dataquest
import fl_model

class EdgeNode:
    """
    Représente un nœud edge dans le système d'apprentissage fédéré.
    Chaque edge est responsable de charger ses propres données.
    """
    
    def __init__(self, edge_id, base_path, data_folders=None, verbose=0):
        """
        Initialise un nœud edge.
        
        Args:
            edge_id: Identifiant unique de l'edge
            base_path: Chemin de base vers les données
            data_folders: Liste des dossiers à charger (si None, utilise un sous-ensemble aléatoire)
            verbose: Niveau de verbosité
        """
        self.edge_id = edge_id
        self.base_path = base_path
        self.data_folders = data_folders
        self.verbose = verbose
        self.model = None
        self.dataset = None
        self.sample_count = 0
        
    def load_data(self, num_classes=10, folders_per_edge=None):
        """
        Charge les données pour cet edge.
        Chaque edge ne charge qu'un sous-ensemble de dossiers MNIST.
        
        Args:
            num_classes: Nombre total de classes
            folders_per_edge: Nombre de dossiers (classes) que cet edge doit charger
        """
        if self.verbose > 0:
            print(f"Edge {self.edge_id} commence à charger ses données...")
        
        # Si aucune liste de dossiers n'est fournie, choisissez aléatoirement
        if self.data_folders is None:
            if folders_per_edge is None:
                folders_per_edge = random.randint(1, num_classes)
                
            # Choisir aléatoirement des dossiers (classes)
            all_folders = [str(i) for i in range(num_classes)]
            self.data_folders = random.sample(all_folders, folders_per_edge)
            
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
        
        # Créer un dataset TensorFlow
        self.dataset = tf.data.Dataset.from_tensor_slices((data, labels_encoded))
        self.dataset = self.dataset.shuffle(buffer_size=len(all_labels))
        self.dataset = self.dataset.batch(32)
        
        self.sample_count = len(all_labels)
        
        if self.verbose > 0:
            print(f"Edge {self.edge_id} a chargé {self.sample_count} échantillons")
            
        return self.dataset, self.sample_count
    
    def train_model(self, global_model_weights, input_shape, num_classes=10, epochs=1):
        """
        Entraîne un modèle local en utilisant les données de cet edge.
        
        Args:
            global_model_weights: Poids du modèle global à utiliser pour initialiser le modèle local
            input_shape: Forme des données d'entrée
            num_classes: Nombre de classes pour la classification
            epochs: Nombre d'époques d'entraînement
            
        Returns:
            Les poids du modèle local entraîné
        """
        if self.dataset is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
        
        # Initialiser le modèle local avec les poids du modèle global
        self.model = fl_model.MyModel(input_shape, nbclasses=num_classes)
        self.model.set_weights(global_model_weights)
        
        # Entraîner le modèle local
        self.model.fit_it(trains=self.dataset, epochs=epochs, tests=None, verbose=self.verbose)
        
        # Retourner les poids du modèle local
        return self.model.get_weights()
    
    def get_sample_count(self):
        """
        Retourne le nombre d'échantillons dans le dataset de cet edge.
        """
        return self.sample_count


class FederatedServer:
    """
    Représente le serveur central dans le système d'apprentissage fédéré.
    """
    
    def __init__(self, input_shape, num_classes=10, verbose=0):
        """
        Initialise le serveur fédéré.
        
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
    
    def aggregate_weights(self, edge_weights_list, edge_sample_counts):
        """
        Agrège les poids des modèles locaux en utilisant l'algorithme FedAvg.
        
        Args:
            edge_weights_list: Liste des poids des modèles locaux
            edge_sample_counts: Liste du nombre d'échantillons par edge
            
        Returns:
            Les poids agrégés du modèle global
        """
        if len(edge_weights_list) != len(edge_sample_counts):
            raise ValueError("La liste des poids et la liste des nombres d'échantillons doivent avoir la même longueur")
        
        # Calculer le nombre total d'échantillons
        total_samples = sum(edge_sample_counts)
        
        # Calculer les poids agrégés pondérés par le nombre d'échantillons
        aggregated_weights = []
        
        # Pour chaque couche du modèle
        for layer_index in range(len(edge_weights_list[0])):
            layer_weights = []
            
            # Pour chaque edge
            for edge_index in range(len(edge_weights_list)):
                # Obtenir les poids de cette couche pour cet edge
                edge_layer_weights = edge_weights_list[edge_index][layer_index]
                
                # Pondérer ces poids par la proportion d'échantillons
                weight_factor = edge_sample_counts[edge_index] / total_samples
                weighted_weights = edge_layer_weights * weight_factor
                
                layer_weights.append(weighted_weights)
            
            # Sommer tous les poids pondérés pour cette couche
            aggregated_layer_weights = sum(layer_weights)
            aggregated_weights.append(aggregated_layer_weights)
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights):
        """
        Met à jour le modèle global avec les poids agrégés.
        
        Args:
            aggregated_weights: Poids agrégés à utiliser pour mettre à jour le modèle global
        """
        self.global_model.set_weights(aggregated_weights)
        
    def evaluate_global_model(self):
        """
        Évalue le modèle global sur le dataset de test.
        
        Returns:
            La perte et la précision du modèle global
        """
        if self.test_dataset is None:
            raise ValueError("Aucun dataset de test n'a été défini. Appelez set_test_data() d'abord.")
        
        return self.global_model.evaluate(self.test_dataset, verbose=self.verbose)


def run_federated_learning(mnist_base_path, test_dataset, num_edges=10, num_rounds=5, 
                          edge_epochs=1, input_shape=(28, 28), num_classes=10, verbose=1):
    """
    Exécute le processus d'apprentissage fédéré complet.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        test_dataset: Dataset de test pour l'évaluation
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes pour la classification
        verbose: Niveau de verbosité
        
    Returns:
        L'historique des évaluations du modèle global
    """
    # Initialiser le serveur
    server = FederatedServer(input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges
    edges = []
    for i in range(num_edges):
        # Assurer que chaque edge a au moins 2 classes pour avoir suffisamment de données
        edge = EdgeNode(f"edge_{i}", mnist_base_path, verbose=verbose)
        edges.append(edge)
    
    # Charger les données pour chaque edge
    # Chaque edge charge ses propres données
    for edge in edges:
        edge.load_data(num_classes, folders_per_edge=max(2, random.randint(1, num_classes // 2)))
    
    # Historique des évaluations
    evaluation_history = []
    
    # Exécuter les rounds de fédération
    for round_num in range(num_rounds):
        if verbose > 0:
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Obtenir les poids du modèle global actuel
        global_weights = server.initialize_model() if round_num == 0 else server.global_model.get_weights()
        
        # Entraîner chaque edge avec les poids globaux actuels
        edge_weights_list = []
        edge_sample_counts = []
        
        for edge in edges:
            # Entraîner le modèle local de cet edge
            edge_weights = edge.train_model(global_weights, input_shape, num_classes, edge_epochs)
            edge_weights_list.append(edge_weights)
            edge_sample_counts.append(edge.get_sample_count())
        
        # Agréger les poids des modèles locaux
        aggregated_weights = server.aggregate_weights(edge_weights_list, edge_sample_counts)
        
        # Mettre à jour le modèle global
        server.update_global_model(aggregated_weights)
        
        # Évaluer le modèle global
        loss, accuracy = server.evaluate_global_model()
        evaluation_history.append((loss, accuracy))
        
        if verbose > 0:
            print(f"Round {round_num + 1} - Perte: {loss:.4f}, Précision: {accuracy:.4f}")
    
    return evaluation_history


def compare_with_centralized(mnist_base_path, num_edges=10, num_rounds=5, edge_epochs=1, verbose=1):
    """
    Compare l'apprentissage fédéré avec l'approvisionnement des données par les edges
    à l'approche centralisée traditionnelle.
    
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
    
    # Exécuter l'apprentissage fédéré
    fed_history = run_federated_learning(
        mnist_base_path, 
        test_dataset, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        input_shape=input_shape, 
        verbose=verbose
    )
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    X_train, X_test, y_train, y_test, _ = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    train_dataset, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    central_model = fl_model.MyModel(input_shape, nbclasses=10)
    central_model.fit_it(trains=train_dataset, epochs=num_rounds * edge_epochs, tests=test_dataset, verbose=verbose)
    
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Extraire les résultats finaux d'apprentissage fédéré
    final_fed_loss, final_fed_accuracy = fed_history[-1]
    print(f"Modèle fédéré - Perte: {final_fed_loss:.4f}, Précision: {final_fed_accuracy:.4f}")
    
    # Tracer les résultats
    plt.figure(figsize=(12, 5))
    
    # Sous-graphique pour la perte
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in fed_history], 'bo-', label='Federated Loss')
    plt.axhline(y=central_loss, color='r', linestyle='-', label='Centralized Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    
    # Sous-graphique pour la précision
    plt.subplot(1, 2, 2)
    plt.plot([x[1] for x in fed_history], 'go-', label='Federated Accuracy')
    plt.axhline(y=central_accuracy, color='r', linestyle='-', label='Centralized Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Définir le chemin de base vers les données MNIST
    # Utilisez le chemin qui a été détecté dans les logs d'erreur
    mnist_base_path = 'C:/Users/barra/Documents/M2/Menez/TP4/Mnist/Mnist/trainingSet/trainingSet/'
    
    # Comparer l'apprentissage fédéré avec l'approvisionnement des données par les edges
    # à l'approche centralisée traditionnelle
    compare_with_centralized(
        mnist_base_path, 
        num_edges=10, 
        num_rounds=5, 
        edge_epochs=1, 
        verbose=1
    )