"""
Ce module implémente le serveur central dans le système d'apprentissage fédéré.
Il gère l'agrégation des poids et l'évaluation du modèle global.
"""
import numpy as np
import tensorflow as tf

class FederatedServer:
    """
    Représente le serveur central dans le système d'apprentissage fédéré.
    Gère l'initialisation, l'agrégation et l'évaluation du modèle global.
    """
    
    def __init__(self, model_class, input_shape, num_classes=10, verbose=0):
        """
        Initialise le serveur fédéré.
        
        Args:
            model_class: Classe du modèle à utiliser (FedProxModel ou autre)
            input_shape: Forme des données d'entrée
            num_classes: Nombre de classes pour la classification
            verbose: Niveau de verbosité
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.verbose = verbose
        self.model_class = model_class
        
        # Initialisation du modèle global avec mu=0 (pas de régularisation proximale au niveau global)
        self.global_model = model_class(input_shape, nbclasses=num_classes, mu=0.0)
        
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