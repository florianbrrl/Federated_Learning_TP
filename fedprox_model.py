"""
Ce module étend le modèle de base pour implémenter FedProx.
Il ajoute la régularisation proximale au modèle d'apprentissage fédéré.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD
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
        self.model.add(Input(input_shape))
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
        
        # Initialisation sans le terme proximal
        self.loss_fn = 'categorical_crossentropy'
        self.model.compile(optimizer="SGD", loss=self.loss_fn, metrics=["accuracy"])
    
    def get_weights(self):
        """Récupère les poids du modèle."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Définit les poids du modèle."""
        self.model.set_weights(weights)
        
    def set_global_weights(self, weights):
        """
        Stocke les poids globaux pour la régularisation proximale.
        Ces poids servent de référence pour limiter la divergence.
        """
        self.global_weights = weights.copy()
    
    def _proximal_term(self):
        """
        Calcule le terme de régularisation proximale entre les poids
        actuels et les poids globaux.
        """
        if self.global_weights is None:
            return 0.0
            
        proximal_term = 0.0
        current_weights = self.model.get_weights()
        
        for w_curr, w_glob in zip(current_weights, self.global_weights):
            proximal_term += (1/2) * tf.reduce_sum(tf.square(w_curr - w_glob))
            
        return self.mu * proximal_term
    
    def compile_proximal(self):
        """
        Compile le modèle avec la perte proximale si mu > 0.
        La régularisation proximale est ajoutée si mu > 0, sinon
        on utilise simplement la perte standard (FedAvg).
        """
        if self.mu > 0 and self.global_weights is not None:
            # Définir une fonction de perte personnalisée avec le terme proximal
            def proximal_loss(y_true, y_pred):
                # Perte originale
                original_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
                # Ajouter le terme proximal
                return original_loss + self._proximal_term()
                
            # Compiler avec la perte proximale
            self.model.compile(
                optimizer=SGD(learning_rate=0.01),
                loss=proximal_loss,
                metrics=["accuracy"]
            )
        else:
            # Sans régularisation proximale (FedAvg standard)
            self.model.compile(
                optimizer=SGD(learning_rate=0.01),
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
        # Recompiler pour s'assurer que la perte est à jour
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
        """Retourne un résumé du modèle."""
        return self.model.summary()
    
    def pretty_print_layers(self):
        """Affiche les informations sur les poids des couches du modèle."""
        for layer_i in range(len(self.model.layers)):
            l = self.model.layers[layer_i].get_weights()
            if len(l) != 0:
                w = self.model.layers[layer_i].get_weights()[0]  # weight 
                b = self.model.layers[layer_i].get_weights()[1]  # bias
                print(f'Layer {layer_i} has weights of shape {np.shape(w)} and biases of shape {np.shape(b)}')