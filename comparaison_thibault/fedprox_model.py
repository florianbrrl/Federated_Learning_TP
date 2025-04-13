"""
Ce module implémente un modèle pour FedProx avec support pour la régularisation proximale.
Il étend le modèle de base en ajoutant un terme de régularisation pour limiter la divergence
entre les modèles locaux et le modèle global.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD, legacy
import tensorflow.keras.backend as K

class FedProxModel:
    """
    Modèle pour l'apprentissage fédéré avec support pour FedProx.
    Cette classe étend les fonctionnalités du modèle de base avec
    la régularisation proximale qui caractérise FedProx.
    """
    
    def __init__(self, input_shape, nbclasses=10, mu=0.0):
        """
        Initialise le modèle avec un paramètre de régularisation proximale.
        
        Args:
            input_shape: Forme des données d'entrée (ex: (28, 28) pour MNIST)
            nbclasses: Nombre de classes pour la classification
            mu: Paramètre de régularisation proximale (0 = équivalent à FedAvg)
        """
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(200))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))  # Ajout d'un dropout pour améliorer la généralisation
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.1))  # Ajout d'un dropout pour améliorer la généralisation
        self.model.add(Dense(nbclasses))
        self.model.add(Activation("softmax"))

        self.input_shape = input_shape
        self.classes = nbclasses
        self.mu = mu
        self.global_weights = None
        
        # Utiliser l'optimiseur legacy pour M1/M2 Macs
        try:
            # Tenter d'utiliser l'optimiseur legacy avec un taux d'apprentissage adapté
            self.optimizer = legacy.SGD(learning_rate=0.01)
        except:
            # Retomber sur l'optimiseur standard si legacy n'est pas disponible
            self.optimizer = SGD(learning_rate=0.01)
        
        # Initialisation sans le terme proximal
        self.loss_fn = 'categorical_crossentropy'
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"])
    
    def get_weights(self):
        """
        Récupère les poids du modèle.
        
        Returns:
            Les poids du modèle
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        Définit les poids du modèle.
        
        Args:
            weights: Les poids à définir
        """
        self.model.set_weights(weights)
        
    def set_global_weights(self, weights):
        """
        Stocke les poids globaux pour la régularisation proximale.
        Ces poids servent de référence pour limiter la divergence.
        
        Args:
            weights: Les poids globaux à stocker
        """
        # Stocker une copie profonde des poids globaux (pas de référence)
        self.global_weights = [np.array(w, dtype=np.float32) for w in weights]
        
    def manual_proximal_step(self):
        """
        Applique manuellement le terme proximal après l'entraînement standard.
        Cette approche est plus robuste que l'intégration du terme dans la fonction de perte.
        """
        if self.mu <= 0 or self.global_weights is None:
            return
            
        current_weights = self.get_weights()
        proximal_weights = []
        
        for i, (w_current, w_global) in enumerate(zip(current_weights, self.global_weights)):
            # Vérifier que les formes correspondent
            if w_current.shape != w_global.shape:
                print(f"Warning: Shape mismatch at layer {i}: {w_current.shape} vs {w_global.shape}")
                proximal_weights.append(w_current)  # conserver les poids actuels
                continue
                
            # Formule FedProx: w_new = w_current - mu * (w_current - w_global)
            w_proximal = w_current - self.mu * (w_current - w_global)
            proximal_weights.append(w_proximal)
        
        # Mettre à jour les poids avec l'ajustement proximal
        self.set_weights(proximal_weights)
    
    def compile_proximal(self):
        """
        Compile le modèle pour l'entraînement.
        """
        # Utiliser l'optimiseur standard - n'incluez pas de terme proximal dans la perte
        # Nous appliquerons manuellement le terme proximal après l'entraînement
        try:
            self.optimizer = legacy.SGD(learning_rate=0.01)
        except:
            self.optimizer = SGD(learning_rate=0.01)
            
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"]
        )
    
    def fit_it(self, trains, epochs, tests=None, verbose=0):
        """
        Entraîne le modèle avec régularisation proximale manuelle.
        
        Args:
            trains: Dataset d'entraînement
            epochs: Nombre d'époques d'entraînement
            tests: Dataset de test (optionnel)
            verbose: Niveau de verbosité (0=silencieux, 1=progression, 2=détaillé)
        """
        # Entraîner le modèle normalement (sans terme proximal dans la perte)
        self.history = self.model.fit(
            trains,
            epochs=epochs,
            validation_data=tests,
            verbose=verbose
        )
        
        # Appliquer manuellement le terme proximal après l'entraînement
        if self.mu > 0 and self.global_weights is not None:
            self.manual_proximal_step()

    def evaluate(self, tests, verbose=0):
        """
        Évalue le modèle sur un dataset de test.
        
        Args:
            tests: Dataset de test
            verbose: Niveau de verbosité
            
        Returns:
            loss, accuracy: La perte et la précision du modèle
        """
        return self.model.evaluate(tests, verbose=verbose)

    def summary(self):
        """
        Retourne un résumé du modèle.
        
        Returns:
            Le résumé du modèle
        """
        return self.model.summary()
    
    def pretty_print_layers(self):
        """
        Affiche les informations sur les poids des couches du modèle.
        """
        for layer_i in range(len(self.model.layers)):
            l = self.model.layers[layer_i].get_weights()
            if len(l) != 0:
                w = self.model.layers[layer_i].get_weights()[0]  # weight 
                b = self.model.layers[layer_i].get_weights()[1]  # bias
                print(f'Layer {layer_i} has weights of shape {np.shape(w)} and biases of shape {np.shape(b)}')