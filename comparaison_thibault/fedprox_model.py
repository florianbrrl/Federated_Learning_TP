"""
Ce module implémente un modèle pour FedProx avec support pour la régularisation proximale.
Il étend le modèle de base en ajoutant un terme de régularisation pour limiter la divergence
entre les modèles locaux et le modèle global.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Input
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
        self.model.add(Dense(128))
        self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        self.model.add(Dense(nbclasses))
        self.model.add(Activation("softmax"))

        self.input_shape = input_shape
        self.classes = nbclasses
        self.mu = mu
        self.global_weights = None
        self.proximal_layers = []  # Pour stocker les variables du terme proximal
        
        # Utiliser l'optimiseur legacy pour M1/M2 Macs
        try:
            # Tenter d'utiliser l'optimiseur legacy
            optimizer = legacy.SGD(learning_rate=0.01)
        except:
            # Retomber sur l'optimiseur standard si legacy n'est pas disponible
            optimizer = SGD(learning_rate=0.01)
        
        # Initialisation sans le terme proximal
        self.loss_fn = 'categorical_crossentropy'
        self.model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=["accuracy"])
    
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
        # Stocker les poids globaux comme des tenseurs constants
        self.global_weights = []
        # Convertir les poids numpy en tenseurs constants TensorFlow
        for w in weights:
            self.global_weights.append(tf.constant(w, dtype=tf.float32))
        
        # Préparer les couches du modèle pour le terme proximal
        self.proximal_layers = []
        model_layers = [layer for layer in self.model.layers if len(layer.weights) > 0]
        
        # S'assurer que les longueurs correspondent
        if len(model_layers) * 2 != len(self.global_weights):
            raise ValueError(f"Mismatch in weights: model has {len(model_layers)} layers with weights, but global_weights has {len(self.global_weights)} elements")
    
    def proximal_loss_wrapper(self):
        """
        Crée une fonction de perte personnalisée qui inclut le terme proximal.
        Cette approche évite l'appel à get_weights() dans le graphe TensorFlow.
        
        Returns:
            La fonction de perte avec terme proximal
        """
        mu = self.mu
        global_weights = self.global_weights
        
        def proximal_loss(y_true, y_pred):
            # Perte originale
            original_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # Si pas de poids globaux, retourner juste la perte originale
            if global_weights is None or mu == 0.0:
                return original_loss
            
            # Ajouter le terme proximal
            proximal_term = 0.0
            
            # Récupérer les couches avec des poids
            model_layers = [layer for layer in self.model.layers if len(layer.weights) > 0]
            
            # Calculer la régularisation proximale
            i = 0
            for layer in model_layers:
                # Pour chaque poids et biais dans la couche
                for w, w_t in zip(layer.weights, global_weights[i:i+len(layer.weights)]):
                    # Calculer la norme L2 carrée de la différence
                    proximal_term += (1.0/2.0) * tf.reduce_sum(tf.square(w - w_t))
                i += len(layer.weights)
            
            # Retourner la perte combinée
            return original_loss + mu * proximal_term
        
        return proximal_loss
    
    def compile_proximal(self):
        """
        Compile le modèle avec la perte proximale si mu > 0.
        La régularisation proximale est ajoutée si mu > 0, sinon
        on utilise simplement la perte standard (FedAvg).
        """
        try:
            # Tenter d'utiliser l'optimiseur legacy
            optimizer = legacy.SGD(learning_rate=0.01)
        except:
            # Retomber sur l'optimiseur standard si legacy n'est pas disponible
            optimizer = SGD(learning_rate=0.01)
            
        if self.mu > 0 and self.global_weights is not None:
            # Utiliser la fonction wrapper qui ne dépend pas de get_weights()
            self.model.compile(
                optimizer=optimizer,
                loss=self.proximal_loss_wrapper(),
                metrics=["accuracy"]
            )
        else:
            # Sans régularisation proximale (FedAvg standard)
            self.model.compile(
                optimizer=optimizer,
                loss=self.loss_fn,
                metrics=["accuracy"]
            )
    
    def fit_it(self, trains, epochs, tests=None, verbose=0):
        """
        Entraîne le modèle avec ou sans régularisation proximale.
        
        Args:
            trains: Dataset d'entraînement
            epochs: Nombre d'époques d'entraînement
            tests: Dataset de test (optionnel)
            verbose: Niveau de verbosité (0=silencieux, 1=progression, 2=détaillé)
        """
        # Si FedProx est activé (mu > 0), recompiler avec la perte proximale
        if self.mu > 0 and self.global_weights is not None:
            self.compile_proximal()
        
        # Entraîner le modèle
        self.history = self.model.fit(
            trains,
            epochs=epochs,
            validation_data=tests,
            verbose=verbose
        )

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