"""
Script principal pour exécuter l'apprentissage fédéré avec μ adaptatif.
Compare les performances de FedAvg, FedProx standard et FedProx avec μ adaptatif.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random
import concurrent.futures
from collections import Counter

# Importer les modules personnalisés
from fedprox_model import FedProxModel
from adaptive_edge_node import AdaptiveEdgeNode
from adaptive_federated_server import AdaptiveFederatedServer
from adaptive_mu_calculator import extract_class_distribution

# Importation des modules de base pour les données et l'évaluation
import fl_dataquest


def run_federated_learning_adaptive_mu(mnist_base_path, test_dataset, 
                                      num_edges=20, num_rounds=10, edge_epochs=1, 
                                      input_shape=(28, 28), num_classes=10, 
                                      folders_per_edge=2, verbose=1, 
                                      client_fraction=1.0, # Facultatif: fraction des clients à sélectionner par round
                                      use_momentum=True, # Utiliser momentum pour stabiliser l'apprentissage
                                      momentum=0.1, # Paramètre de momentum
                                      dirichlet_alpha=0.5): # Pour distribution non-IID plus naturelle
    """
    Exécute le processus d'apprentissage fédéré avec μ adaptatif.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        test_dataset: Dataset de test pour l'évaluation
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes pour la classification
        folders_per_edge: Nombre de dossiers (classes) par edge
        verbose: Niveau de verbosité
        client_fraction: Fraction des clients à sélectionner par round
        use_momentum: Utiliser momentum pour stabiliser l'apprentissage
        momentum: Paramètre de momentum
        dirichlet_alpha: Paramètre alpha pour distribution de Dirichlet (plus petit = plus non-IID)
        
    Returns:
        L'historique des évaluations du modèle global et des valeurs de μ
    """
    # Initialiser le serveur avec support pour μ adaptatif
    server = AdaptiveFederatedServer(FedProxModel, input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges avec support pour μ adaptatif
    edges = []
    for i in range(num_edges):
        # Initialiser avec une valeur μ de départ
        edge = AdaptiveEdgeNode(f"edge_{i}", mnist_base_path, FedProxModel, mu=0.01, verbose=verbose)
        edges.append(edge)
    
    # Charger les données pour chaque edge avec distribution de Dirichlet
    for edge in edges:
        edge.load_data(num_classes, folders_per_edge=folders_per_edge, dirichlet_alpha=dirichlet_alpha)
    
    # Historique des évaluations et des valeurs de μ
    evaluation_history = []
    mu_history = []
    
    # Pour le momentum
    previous_weights = None
    
    # Pour adaptative learning rate
    initial_lr = 0.01
    
    # Exécuter les rounds de fédération
    for round_num in range(num_rounds):
        if verbose > 0:
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Obtenir les poids du modèle global actuel
        global_weights = server.initialize_model() if round_num == 0 else server.global_model.get_weights()
        
        # Sélectionner un sous-ensemble de clients si client_fraction < 1
        selected_edges = edges
        if client_fraction < 1.0:
            num_selected = max(1, int(client_fraction * num_edges))
            selected_edges = random.sample(edges, num_selected)
            if verbose > 0:
                print(f"Sélection de {num_selected} clients pour ce round")
        
        # Collecter les distributions de classes de tous les edges sélectionnés
        edge_class_distributions = [edge.get_class_distribution() for edge in selected_edges]
        edge_sample_counts = [edge.get_sample_count() for edge in selected_edges]
        
        # Mettre à jour la distribution globale des classes
        server.update_global_class_distribution(edge_class_distributions, edge_sample_counts)
        
        # Si ce n'est pas le premier round, calculer les valeurs de μ adaptatives
        if round_num > 0:
            current_accuracy = evaluation_history[-1][1]
            edge_mus = server.calculate_adaptive_mus(
                edge_class_distributions, 
                round_num, 
                num_rounds, 
                current_accuracy
            )
            
            # Mettre à jour les valeurs de μ pour chaque edge
            for i, edge in enumerate(selected_edges):
                edge.update_mu(edge_mus[i])
            
            # Enregistrer les valeurs de μ
            mu_history.append(edge_mus)
        else:
            # Premier round, utiliser les valeurs initiales
            initial_mus = [edge.mu for edge in selected_edges]
            mu_history.append(initial_mus)
        
        # Ajuster le learning rate en fonction du round
        current_lr = initial_lr * (1.0 / (1.0 + 0.1 * round_num))
        if verbose > 0:
            print(f"Learning rate pour ce round: {current_lr:.6f}")
        
        # Fonction pour entraîner un edge en parallèle
        def train_edge(edge):
            # Ajuster le nombre d'époques en fonction de la taille des données locales
            adaptive_epochs = edge_epochs
            if edge.get_sample_count() < 500:
                adaptive_epochs = min(5, edge_epochs + 2)  # Plus d'époques pour les petits datasets
            elif edge.get_sample_count() > 2000:
                adaptive_epochs = max(1, edge_epochs - 1)  # Moins d'époques pour les grands datasets
                
            # Entraîner le modèle local de cet edge
            edge_weights = edge.train_model(
                global_weights, 
                input_shape, 
                num_classes, 
                adaptive_epochs,
                learning_rate=current_lr
            )
            return edge_weights
        
        # Entraîner les edges en parallèle si possible
        edge_weights_list = []
        try:
            # Utiliser la parallélisation si le nombre d'edges est suffisant
            if len(selected_edges) >= 4:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_edges), os.cpu_count())) as executor:
                    edge_weights_list = list(executor.map(train_edge, selected_edges))
            else:
                # Sinon, entraîner séquentiellement
                for edge in selected_edges:
                    edge_weights_list.append(train_edge(edge))
        except Exception as e:
            print(f"Erreur lors de l'entraînement parallèle: {e}")
            # Fallback à l'entraînement séquentiel en cas d'erreur
            edge_weights_list = []
            for edge in selected_edges:
                edge_weights = edge.train_model(global_weights, input_shape, num_classes, edge_epochs)
                edge_weights_list.append(edge_weights)
        
        # Agréger les poids des modèles locaux
        aggregated_weights = server.aggregate_weights(edge_weights_list, edge_sample_counts)
        
        # Appliquer le momentum si activé et après le premier round
        if use_momentum and previous_weights is not None:
            new_weights = []
            for i in range(len(aggregated_weights)):
                # Mélanger les nouveaux poids avec les anciens selon le facteur de momentum
                new_layer_weights = (1 - momentum) * aggregated_weights[i] + momentum * previous_weights[i]
                new_weights.append(new_layer_weights)
            aggregated_weights = new_weights
        
        # Stocker les poids pour le momentum du prochain round
        previous_weights = aggregated_weights.copy() if use_momentum else None
        
        # Mettre à jour le modèle global
        server.update_global_model(aggregated_weights)
        
        # Évaluer le modèle global
        loss, accuracy = server.evaluate_global_model()
        evaluation_history.append((loss, accuracy))
        
        if verbose > 0:
            avg_mu = np.mean(mu_history[-1]) if mu_history else 0
            print(f"Round {round_num + 1} - Perte: {loss:.4f}, Précision: {accuracy:.4f}, μ moyen: {avg_mu:.4f}")
    
    return evaluation_history, mu_history


def run_federated_learning_standard(mnist_base_path, test_dataset, model_class, mu=0.0, 
                                   num_edges=20, num_rounds=5, edge_epochs=1, 
                                   input_shape=(28, 28), num_classes=10, 
                                   folders_per_edge=2, verbose=1):
    """
    Exécute le processus d'apprentissage fédéré standard (FedAvg ou FedProx avec μ fixe).
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        test_dataset: Dataset de test pour l'évaluation
        model_class: Classe du modèle à utiliser (FedProxModel)
        mu: Paramètre de régularisation proximale (0 = FedAvg, >0 = FedProx)
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes pour la classification
        folders_per_edge: Nombre de dossiers (classes) par edge
        verbose: Niveau de verbosité
        
    Returns:
        L'historique des évaluations du modèle global
    """
    from edge_node import EdgeNode
    from federated_server import FederatedServer
    
    # Initialiser le serveur
    server = FederatedServer(model_class, input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges
    edges = []
    for i in range(num_edges):
        edge = EdgeNode(f"edge_{i}", mnist_base_path, model_class, mu=mu, verbose=verbose)
        edges.append(edge)
    
    # Charger les données pour chaque edge
    for edge in edges:
        edge.load_data(num_classes, folders_per_edge=folders_per_edge)
    
    # Historique des évaluations
    evaluation_history = []
    
    # Pour le momentum
    previous_weights = None
    momentum = 0.1
    use_momentum = True
    
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
        
        # Appliquer le momentum si activé et après le premier round
        if use_momentum and previous_weights is not None:
            new_weights = []
            for i in range(len(aggregated_weights)):
                # Mélanger les nouveaux poids avec les anciens selon le facteur de momentum
                new_layer_weights = (1 - momentum) * aggregated_weights[i] + momentum * previous_weights[i]
                new_weights.append(new_layer_weights)
            aggregated_weights = new_weights
        
        # Stocker les poids pour le momentum du prochain round
        previous_weights = aggregated_weights.copy() if use_momentum else None
        
        # Mettre à jour le modèle global
        server.update_global_model(aggregated_weights)
        
        # Évaluer le modèle global
        loss, accuracy = server.evaluate_global_model()
        evaluation_history.append((loss, accuracy))
        
        if verbose > 0:
            print(f"Round {round_num + 1} - Perte: {loss:.4f}, Précision: {accuracy:.4f}")
    
    return evaluation_history


def compare_fedavg_fedprox_adaptive(mnist_base_path, num_edges=20, num_rounds=10, edge_epochs=1, 
                                  folders_per_edge=2, verbose=1, use_adaptive_params=True):
    """
    Compare les performances de FedAvg, FedProx standard et FedProx avec μ adaptatif.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        folders_per_edge: Nombre de dossiers (classes) par edge 
        verbose: Niveau de verbosité
        use_adaptive_params: Utiliser des paramètres adaptatifs améliorés
        
    Returns:
        Dictionnaire des historiques d'évaluation, historique de μ, et métriques du modèle centralisé
    """
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Paramètres adaptatifs
    client_fraction = 0.8 if use_adaptive_params else 1.0  # Sélectionner 80% des clients par round
    dirichlet_alpha = 0.5  # Distribution non-IID naturelle
    
    # Exécuter FedAvg
    print("\n--- Exécution de FedAvg ---")
    fedavg_history = run_federated_learning_standard(
        mnist_base_path, 
        test_dataset, 
        FedProxModel, 
        mu=0.0,  # FedAvg
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        input_shape=input_shape, 
        folders_per_edge=folders_per_edge,
        verbose=verbose
    )
    
    # Exécuter FedProx standard
    print("\n--- Exécution de FedProx standard ---")
    fedprox_history = run_federated_learning_standard(
        mnist_base_path, 
        test_dataset, 
        FedProxModel, 
        mu=0.1,  # FedProx avec mu fixe
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        input_shape=input_shape, 
        folders_per_edge=folders_per_edge,
        verbose=verbose
    )
    
    # Exécuter FedProx avec μ adaptatif
    print("\n--- Exécution de FedProx avec μ adaptatif ---")
    adaptive_history, mu_history = run_federated_learning_adaptive_mu(
        mnist_base_path, 
        test_dataset, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        input_shape=input_shape, 
        folders_per_edge=folders_per_edge,
        verbose=verbose,
        client_fraction=client_fraction,
        dirichlet_alpha=dirichlet_alpha
    )
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    central_model = FedProxModel(input_shape, nbclasses=10)
    train_dataset, _ = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Ajouter des callbacks pour améliorer l'entraînement centralisé
    callbacks = []
    if tf.keras.callbacks.EarlyStopping is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))
    if tf.keras.callbacks.ReduceLROnPlateau is not None:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2))
    
    central_model.model.fit(
        train_dataset,
        epochs=num_rounds * edge_epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=verbose
    )
    
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Tracer les résultats
    plot_comparison_with_adaptive(fedavg_history, fedprox_history, adaptive_history, 
                                central_accuracy, mu_history, num_rounds)
    
    return {
        "FedAvg": fedavg_history,
        "FedProx": fedprox_history,
        "AdaptiveFedProx": adaptive_history
    }, mu_history, (central_loss, central_accuracy)


def plot_comparison_with_adaptive(fedavg_history, fedprox_history, adaptive_history, 
                                central_accuracy, mu_history, num_rounds):
    """
    Génère et sauvegarde des graphiques comparant les performances des algorithmes.
    
    Args:
        fedavg_history: Historique d'évaluation pour FedAvg
        fedprox_history: Historique d'évaluation pour FedProx standard
        adaptive_history: Historique d'évaluation pour FedProx avec μ adaptatif
        central_accuracy: Précision du modèle centralisé
        mu_history: Historique des valeurs de μ pour FedProx adaptatif
        num_rounds: Nombre de rounds de fédération
    """
    # Créer le répertoire pour les figures si nécessaire
    os.makedirs('figures', exist_ok=True)
    
    # Préparer les données pour les graphiques
    rounds = list(range(1, num_rounds + 1))
    
    # Extraire les précisions
    fedavg_accuracies = [x[1] for x in fedavg_history]
    fedprox_accuracies = [x[1] for x in fedprox_history]
    adaptive_accuracies = [x[1] for x in adaptive_history]
    
    # Graphique de précision
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, fedavg_accuracies, marker='o', label='FedAvg')
    plt.plot(rounds, fedprox_accuracies, marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, adaptive_accuracies, marker='^', label='FedProx (μ adaptatif)')
    plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Comparaison de la précision: FedAvg vs FedProx vs FedProx Adaptatif', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/accuracy_comparison_adaptive.png', dpi=300, bbox_inches='tight')
    
    # Graphique des pertes
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, [x[0] for x in fedavg_history], marker='o', label='FedAvg')
    plt.plot(rounds, [x[0] for x in fedprox_history], marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, [x[0] for x in adaptive_history], marker='^', label='FedProx (μ adaptatif)')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Perte', fontsize=12)
    plt.title('Comparaison de la perte: FedAvg vs FedProx vs FedProx Adaptatif', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/loss_comparison_adaptive.png', dpi=300, bbox_inches='tight')
    
    # Graphique de l'évolution de μ
    plt.figure(figsize=(12, 6))
    
    # Calculer les statistiques de μ par round
    mu_means = [np.mean(mus) for mus in mu_history]
    mu_mins = [np.min(mus) for mus in mu_history]
    mu_maxs = [np.max(mus) for mus in mu_history]
    
    plt.plot(rounds, mu_means, marker='o', label='μ moyen')
    plt.fill_between(rounds, mu_mins, mu_maxs, alpha=0.2, label='Plage de μ')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Valeur de μ', fontsize=12)
    plt.title('Évolution du paramètre μ adaptatif', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/adaptive_mu_evolution.png', dpi=300, bbox_inches='tight')
    
    print("Graphiques sauvegardés dans le dossier 'figures'")
    
    # Graphique combiné μ vs précision
    plt.figure(figsize=(14, 6))
    
    # Deux axes y pour afficher μ et précision sur le même graphique
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Tracer la moyenne de μ
    line1 = ax1.plot(rounds, mu_means, 'b-', marker='o', label='μ moyen')
    # Tracer la plage de μ
    ax1.fill_between(rounds, mu_mins, mu_maxs, alpha=0.2, color='b')
    
    # Tracer la précision
    line2 = ax2.plot(rounds, adaptive_accuracies, 'r-', marker='s', label='Précision')
    
    # Configurer les axes
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Valeur de μ', fontsize=12, color='b')
    ax2.set_ylabel('Précision', fontsize=12, color='r')
    
    # Configurer les couleurs des axes
    ax1.tick_params(axis='y', colors='b')
    ax2.tick_params(axis='y', colors='r')
    
    # Combiner les légendes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    plt.title('Évolution du μ adaptatif et de la précision', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/mu_vs_accuracy.png', dpi=300, bbox_inches='tight')
    
    # Graphique de comparaison avec le modèle centralisé
    plt.figure(figsize=(14, 7))
    
    # Calculer l'écart relatif avec le modèle centralisé
    fedavg_gap = [(central_accuracy - acc) / central_accuracy * 100 for acc in fedavg_accuracies]
    fedprox_gap = [(central_accuracy - acc) / central_accuracy * 100 for acc in fedprox_accuracies]
    adaptive_gap = [(central_accuracy - acc) / central_accuracy * 100 for acc in adaptive_accuracies]
    
    plt.plot(rounds, fedavg_gap, marker='o', label='FedAvg')
    plt.plot(rounds, fedprox_gap, marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, adaptive_gap, marker='^', label='FedProx (μ adaptatif)')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Écart relatif avec le modèle centralisé (%)', fontsize=12)
    plt.title('Comparaison de l\'écart avec le modèle centralisé', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ajouter des annotations pour les valeurs finales
    final_round = num_rounds - 1
    plt.annotate(f"{fedavg_gap[-1]:.1f}%", 
                xy=(rounds[-1], fedavg_gap[-1]), 
                xytext=(5, 0), 
                textcoords='offset points')
    plt.annotate(f"{fedprox_gap[-1]:.1f}%", 
                xy=(rounds[-1], fedprox_gap[-1]), 
                xytext=(5, 0), 
                textcoords='offset points')
    plt.annotate(f"{adaptive_gap[-1]:.1f}%", 
                xy=(rounds[-1], adaptive_gap[-1]), 
                xytext=(5, 0), 
                textcoords='offset points')
    
    plt.savefig('figures/centralized_comparison.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Définir le chemin vers le dataset MNIST
    mnist_base_path = input("Entrez le chemin vers votre dataset MNIST: ")
    
    # S'assurer que le chemin est correct
    while not os.path.exists(mnist_base_path):
        print(f"Le chemin {mnist_base_path} n'existe pas.")
        mnist_base_path = input("Veuillez entrer le chemin correct vers le dataset MNIST: ")
    
    # Paramètres de l'expérience
    num_edges = 20
    num_rounds = 30
    edge_epochs = 2  # Augmenté de 1 à 2 pour améliorer les performances
    folders_per_edge = 2  # Distribution non-IID: 2 classes par edge
    verbose = 1
    
    # Comparer FedAvg, FedProx standard et FedProx avec μ adaptatif
    print("\n=== Comparaison de FedAvg, FedProx standard et FedProx avec μ adaptatif ===")
    results, mu_history, central_metrics = compare_fedavg_fedprox_adaptive(
        mnist_base_path, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        folders_per_edge=folders_per_edge,
        verbose=verbose,
        use_adaptive_params=True  # Activer les paramètres adaptatifs améliorés
    )
    
    # Afficher les résultats finaux
    print("\n=== Résultats finaux ===")
    print(f"FedAvg - Précision finale: {results['FedAvg'][-1][1]:.4f}")
    print(f"FedProx standard - Précision finale: {results['FedProx'][-1][1]:.4f}")
    print(f"FedProx μ adaptatif - Précision finale: {results['AdaptiveFedProx'][-1][1]:.4f}")
    print(f"Modèle centralisé - Précision: {central_metrics[1]:.4f}")
    
    # Afficher l'écart avec le modèle centralisé
    print("\n=== Écart avec le modèle centralisé ===")
    fedavg_gap = (central_metrics[1] - results['FedAvg'][-1][1]) / central_metrics[1] * 100
    fedprox_gap = (central_metrics[1] - results['FedProx'][-1][1]) / central_metrics[1] * 100
    adaptive_gap = (central_metrics[1] - results['AdaptiveFedProx'][-1][1]) / central_metrics[1] * 100
    
    print(f"FedAvg - Écart: {fedavg_gap:.2f}%")
    print(f"FedProx standard - Écart: {fedprox_gap:.2f}%")
    print(f"FedProx μ adaptatif - Écart: {adaptive_gap:.2f}%")
    
    # Identifier le meilleur algorithme
    best_accuracy = max(results['FedAvg'][-1][1], results['FedProx'][-1][1], results['AdaptiveFedProx'][-1][1])
    if best_accuracy == results['FedAvg'][-1][1]:
        best_algo = "FedAvg"
    elif best_accuracy == results['FedProx'][-1][1]:
        best_algo = "FedProx standard"
    else:
        best_algo = "FedProx μ adaptatif"
    
    print(f"\nMeilleur algorithme: {best_algo} avec une précision de {best_accuracy:.4f}")