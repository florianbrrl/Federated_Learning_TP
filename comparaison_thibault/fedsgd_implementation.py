"""
Ce module implémente FedSGD pour comparaison avec FedAvg, FedProx et FedProx adaptatif.
FedSGD est une variante de l'apprentissage fédéré qui n'effectue qu'une seule étape
de descente de gradient sur chaque client avant l'agrégation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, legacy
import os

# Importer les modules personnalisés
from fedprox_model import FedProxModel
from edge_node import EdgeNode
from federated_server import FederatedServer
import fl_dataquest


class FedSGDEdgeNode(EdgeNode):
    """
    Nœud edge spécifique pour FedSGD, qui n'effectue qu'une étape de SGD par round.
    """
    
    def __init__(self, edge_id, base_path, model_class, data_folders=None, verbose=0):
        """
        Initialise un edge node pour FedSGD.
        """
        super().__init__(edge_id, base_path, model_class, data_folders=data_folders, mu=0.0, verbose=verbose)
        self.learning_rate = 0.01  # Taux d'apprentissage pour SGD
        
    def train_model(self, global_model_weights, input_shape, num_classes=10, epochs=1):
        """
        Effectue une seule étape de SGD sur le modèle local.
        
        Args:
            global_model_weights: Poids du modèle global
            input_shape: Forme des données d'entrée
            num_classes: Nombre de classes
            epochs: Non utilisé dans FedSGD
            
        Returns:
            Poids mis à jour après une étape de SGD
        """
        if self.dataset is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
        
        # Initialiser le modèle local avec les poids du modèle global
        self.model = self.model_class(input_shape, nbclasses=num_classes)
        self.model.set_weights(global_model_weights)
        
        # Obtenir un seul batch de données pour une étape de SGD
        for batch in self.dataset.take(1):
            x_batch, y_batch = batch
            
            # Effectuer une étape de SGD manuellement
            with tf.GradientTape() as tape:
                # Passer les données dans le modèle
                predictions = self.model.model(x_batch, training=True)
                # Calculer la perte
                loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)
                
            # Calculer les gradients
            grads = tape.gradient(loss, self.model.model.trainable_variables)
            
            # Appliquer les gradients manuellement
            for i, var in enumerate(self.model.model.trainable_variables):
                var.assign_sub(self.learning_rate * grads[i])
                
            if self.verbose > 1:
                print(f"Edge {self.edge_id} - Perte après une étape SGD: {loss.numpy():.4f}")
        
        # Retourner les poids mis à jour
        return self.model.get_weights()


def run_federated_learning_sgd(mnist_base_path, test_dataset, 
                              num_edges=10, num_rounds=10, 
                              input_shape=(28, 28), num_classes=10, 
                              folders_per_edge=2, verbose=1):
    """
    Exécute le processus d'apprentissage fédéré avec FedSGD.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        test_dataset: Dataset de test pour l'évaluation
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes pour la classification
        folders_per_edge: Nombre de dossiers (classes) par edge
        verbose: Niveau de verbosité
        
    Returns:
        L'historique des évaluations du modèle global
    """
    # Initialiser le serveur
    server = FederatedServer(FedProxModel, input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges pour FedSGD
    edges = []
    for i in range(num_edges):
        edge = FedSGDEdgeNode(f"edge_{i}", mnist_base_path, FedProxModel, verbose=verbose)
        edges.append(edge)
    
    # Charger les données pour chaque edge
    for edge in edges:
        edge.load_data(num_classes, folders_per_edge=folders_per_edge)
    
    # Historique des évaluations
    evaluation_history = []
    
    # Exécuter les rounds de fédération
    for round_num in range(num_rounds):
        if verbose > 0:
            print(f"\n--- Round {round_num + 1}/{num_rounds} - FedSGD ---")
        
        # Obtenir les poids du modèle global actuel
        global_weights = server.initialize_model() if round_num == 0 else server.global_model.get_weights()
        
        # Entraîner chaque edge avec une seule étape de SGD
        edge_weights_list = []
        edge_sample_counts = []
        
        for edge in edges:
            # Entraîner le modèle local de cet edge avec une étape de SGD
            edge_weights = edge.train_model(global_weights, input_shape, num_classes)
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