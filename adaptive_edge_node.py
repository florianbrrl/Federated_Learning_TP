"""
Ce module étend la classe EdgeNode pour supporter le paramètre μ adaptatif.
Il permet aux edges d'ajuster leur régularisation proximale dynamiquement.
"""

import numpy as np
import random
import os
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from collections import Counter

from adaptive_mu_calculator import extract_class_distribution

class AdaptiveEdgeNode:
    """
    Représente un nœud edge dans le système d'apprentissage fédéré.
    Chaque edge est responsable de charger ses propres données et 
    supporte un paramètre μ adaptatif pour FedProx.
    """
    
    def __init__(self, edge_id, base_path, model_class, data_folders=None, mu=0.01, verbose=0):
        """
        Initialise un nœud edge avec support pour μ adaptatif.
        
        Args:
            edge_id: Identifiant unique de l'edge
            base_path: Chemin de base vers les données
            model_class: Classe du modèle à utiliser
            data_folders: Liste des dossiers à charger (si None, utilise un sous-ensemble aléatoire)
            mu: Paramètre initial de régularisation proximale pour FedProx (0.0 = FedAvg)
            verbose: Niveau de verbosité
        """
        self.edge_id = edge_id
        self.base_path = base_path
        self.data_folders = data_folders
        self.verbose = verbose
        self.model_class = model_class
        self.mu = mu
        self.model = None
        self.dataset = None
        self.sample_count = 0
        self.class_distribution = None
        
    def load_data(self, num_classes=10, folders_per_edge=None):
        """
        Charge les données pour cet edge et calcule la distribution des classes.
        
        Args:
            num_classes: Nombre total de classes
            folders_per_edge: Nombre de dossiers (classes) que cet edge doit charger
        """
        from fl_dataquest import load_and_preprocess
        
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
        data, _ = load_and_preprocess(all_image_paths, verbose=self.verbose if self.verbose > 1 else 0)
        
        # Convertir les labels en format one-hot
        lb = LabelBinarizer()
        lb.fit(range(num_classes))  # S'assurer que toutes les classes sont représentées
        labels_encoded = lb.transform(all_labels)
        
        # Créer un dataset TensorFlow
        self.dataset = tf.data.Dataset.from_tensor_slices((data, labels_encoded))
        self.dataset = self.dataset.shuffle(buffer_size=len(all_labels))
        self.dataset = self.dataset.batch(32)
        
        self.sample_count = len(all_labels)
        
        # Calculer la distribution des classes
        self.class_distribution = Counter(all_labels)
        self.class_distribution = dict(self.class_distribution)
        
        if self.verbose > 0:
            print(f"Edge {self.edge_id} a chargé {self.sample_count} échantillons")
            print(f"Distribution des classes: {self.class_distribution}")
            
        return self.dataset, self.sample_count
    
    def train_model(self, global_model_weights, input_shape, num_classes=10, epochs=1):
        """
        Entraîne un modèle local en utilisant les données de cet edge.
        Implémente FedProx en ajustant les poids après l'entraînement si mu > 0.
        
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
        self.model = self.model_class(input_shape, nbclasses=num_classes, mu=self.mu)
        self.model.set_weights(global_model_weights)
        
        # Si on utilise FedProx, on stocke les poids globaux pour la régularisation
        if self.mu > 0:
            self.model.set_global_weights(global_model_weights)
            # Recompiler le modèle avec la perte proximale
            self.model.compile_proximal()
            
            if self.verbose > 1:
                print(f"Edge {self.edge_id} utilise FedProx avec μ = {self.mu}")
        
        # Entraîner le modèle local
        self.model.fit_it(trains=self.dataset, epochs=epochs, tests=None, verbose=self.verbose)
        
        # Retourner les poids du modèle local
        return self.model.get_weights()
    
    def get_sample_count(self):
        """
        Retourne le nombre d'échantillons dans le dataset de cet edge.
        """
        return self.sample_count
    
    def get_class_distribution(self):
        """
        Retourne la distribution des classes pour cet edge.
        """
        if self.class_distribution is None:
            raise ValueError("La distribution des classes n'a pas été calculée. Appelez load_data() d'abord.")
        
        return self.class_distribution
    
    def update_mu(self, new_mu):
        """
        Met à jour le paramètre μ pour cet edge.
        """
        self.mu = new_mu
        if self.verbose > 0:
            print(f"Edge {self.edge_id}: μ mis à jour à {new_mu:.4f}")