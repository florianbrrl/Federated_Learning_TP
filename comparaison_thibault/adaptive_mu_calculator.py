"""
Module d'implémentation du paramètre μ adaptatif pour FedProx.
Ce module calcule dynamiquement le paramètre de régularisation proximale
en fonction de l'hétérogénéité des données et de la progression de l'entraînement.
"""

import numpy as np
from collections import Counter
import scipy.stats as stats

class AdaptiveMuCalculator:
    """
    Classe qui calcule dynamiquement la valeur du paramètre μ pour FedProx
    en fonction de différents facteurs comme l'hétérogénéité des données
    et le stade de l'entraînement.
    """
    
    def __init__(self, initial_mu=0.01, min_mu=0.001, max_mu=0.2):
        """
        Initialise le calculateur de μ adaptatif.
        
        Args:
            initial_mu: Valeur initiale de μ
            min_mu: Valeur minimale autorisée pour μ
            max_mu: Valeur maximale autorisée pour μ
        """
        self.initial_mu = initial_mu
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.current_mu = initial_mu
        self.previous_performances = []
        self.class_distribution_history = []
        
    def update_class_distribution(self, class_distribution):
        """
        Met à jour l'historique des distributions de classes.
        
        Args:
            class_distribution: Dictionnaire avec le nombre d'exemples par classe
        """
        self.class_distribution_history.append(class_distribution)
        
    def update_performance(self, accuracy):
        """
        Met à jour l'historique des performances.
        
        Args:
            accuracy: Précision du modèle au round actuel
        """
        self.previous_performances.append(accuracy)
        
    def compute_distribution_divergence(self, local_distribution, global_distribution):
        """
        Calcule la divergence entre la distribution locale et globale des classes.
        
        Args:
            local_distribution: Distribution des classes pour un client
            global_distribution: Distribution globale des classes
            
        Returns:
            Un score de divergence entre 0 (identique) et 1 (complètement différent)
        """
        # Normaliser les distributions
        total_local = sum(local_distribution.values())
        total_global = sum(global_distribution.values())
        
        # Calculer la divergence de Jensen-Shannon
        p = np.zeros(10)  # Assumant 10 classes pour MNIST
        q = np.zeros(10)
        
        for i in range(10):
            p[i] = local_distribution.get(i, 0) / total_local if total_local > 0 else 0
            q[i] = global_distribution.get(i, 0) / total_global if total_global > 0 else 0
        
        # Éviter les divisions par zéro
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        # Calculer la divergence JS
        m = 0.5 * (p + q)
        js_divergence = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
        
        # Normaliser entre 0 et 1
        return min(1.0, js_divergence / np.log(2))
    
    def calculate_adaptive_mu(self, local_distributions, global_distribution, current_round, total_rounds, 
                              current_accuracy, previous_accuracy=None):
        """
        Calcule le paramètre μ adaptatif en fonction de multiples facteurs.
        
        Args:
            local_distributions: Liste des distributions de classes pour chaque client
            global_distribution: Distribution globale des classes
            current_round: Numéro du round actuel
            total_rounds: Nombre total de rounds prévus
            current_accuracy: Précision actuelle du modèle
            previous_accuracy: Précision du modèle au round précédent
            
        Returns:
            Valeur adaptative de μ
        """
        # 1. Calculer le score d'hétérogénéité moyen
        divergence_scores = []
        for local_dist in local_distributions:
            score = self.compute_distribution_divergence(local_dist, global_distribution)
            divergence_scores.append(score)
        
        mean_divergence = np.mean(divergence_scores)
        
        # 2. Facteur basé sur l'hétérogénéité (plus c'est hétérogène, plus μ est grand)
        heterogeneity_factor = 0.5 + 0.5 * mean_divergence
        
        # 3. Facteur basé sur la progression de l'entraînement
        # Au début, on permet plus d'exploration avec un μ plus faible
        # Vers la fin, on augmente μ pour favoriser la convergence
        round_factor = current_round / total_rounds
        
        # 4. Facteur basé sur l'amélioration de la performance
        performance_factor = 1.0
        if previous_accuracy is not None and previous_accuracy > 0:
            relative_improvement = (current_accuracy - previous_accuracy) / previous_accuracy
            # Si la performance stagne ou diminue, augmenter μ pour stabiliser
            if relative_improvement <= 0:
                performance_factor = 1.2
            # Si la performance s'améliore rapidement, réduire μ pour permettre plus d'adaptation
            elif relative_improvement > 0.05:
                performance_factor = 0.8
        
        # Combiner tous les facteurs pour calculer μ adaptatif
        adaptive_mu = self.initial_mu * heterogeneity_factor * (0.7 + 0.6 * round_factor) * performance_factor
        
        # Limiter à la plage [min_mu, max_mu]
        adaptive_mu = max(self.min_mu, min(self.max_mu, adaptive_mu))
        
        # Mettre à jour la valeur courante
        self.current_mu = adaptive_mu
        
        return adaptive_mu
    
    def get_client_mus(self, client_distributions, global_distribution, current_round, total_rounds, 
                       current_accuracy=None, previous_accuracy=None):
        """
        Calcule des valeurs de μ individualisées pour chaque client en fonction
        de leur divergence par rapport à la distribution globale.
        
        Args:
            client_distributions: Liste des distributions de classes pour chaque client
            global_distribution: Distribution globale des classes
            current_round: Numéro du round actuel
            total_rounds: Nombre total de rounds prévus
            current_accuracy: Précision actuelle du modèle global
            previous_accuracy: Précision du modèle au round précédent
            
        Returns:
            Liste des valeurs de μ pour chaque client
        """
        client_mus = []
        
        # Calculer le μ de base
        base_mu = self.calculate_adaptive_mu(
            client_distributions, global_distribution, 
            current_round, total_rounds, current_accuracy, previous_accuracy
        )
        
        # Ajuster individuellement pour chaque client
        for client_idx, client_dist in enumerate(client_distributions):
            # Calculer la divergence de ce client
            divergence = self.compute_distribution_divergence(client_dist, global_distribution)
            
            # Plus la divergence est grande, plus μ est grand pour limiter la divergence
            client_mu = base_mu * (1.0 + 0.5 * divergence)
            
            # Limiter à la plage [min_mu, max_mu]
            client_mu = max(self.min_mu, min(self.max_mu, client_mu))
            client_mus.append(client_mu)
        
        return client_mus


def extract_class_distribution(dataset):
    """
    Extrait la distribution des classes à partir d'un dataset TensorFlow.
    
    Args:
        dataset: Dataset TensorFlow contenant des tuples (données, labels)
        
    Returns:
        Un dictionnaire avec le nombre d'exemples par classe
    """
    import tensorflow as tf
    
    class_counts = Counter()
    
    # Parcourir le dataset pour compter les classes
    for data_batch, label_batch in dataset:
        # Pour les labels one-hot, convertir en indices de classe
        if len(label_batch.shape) > 1 and label_batch.shape[1] > 1:
            class_indices = tf.argmax(label_batch, axis=1).numpy()
        else:
            class_indices = label_batch.numpy()
        
        batch_counts = Counter(class_indices)
        class_counts.update(batch_counts)
    
    return dict(class_counts)