"""
Script principal pour exécuter des expériences comparant FedAvg et FedProx sur MNIST.
Ce script permet de comparer les performances des deux algorithmes dans différentes
configurations.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random

# Importer les modules personnalisés
from fedprox_model import FedProxModel
from federated_server import FederatedServer
from edge_node import EdgeNode

# Importation des modules de base pour les données et l'évaluation
import fl_dataquest

def run_federated_learning(mnist_base_path, test_dataset, model_class, mu=0.0, 
                          num_edges=10, num_rounds=5, edge_epochs=1, 
                          input_shape=(28, 28), num_classes=10, 
                          folders_per_edge=2, verbose=1):
    """
    Exécute le processus d'apprentissage fédéré complet avec FedAvg ou FedProx.
    
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
    # Initialiser le serveur
    server = FederatedServer(model_class, input_shape, num_classes, verbose)
    server.set_test_data(test_dataset)
    
    # Initialiser les edges
    edges = []
    for i in range(num_edges):
        edge = EdgeNode(f"edge_{i}", mnist_base_path, model_class, mu=mu, verbose=verbose)
        edges.append(edge)
    
    # Charger les données pour chaque edge
    # Chaque edge charge ses propres données
    for edge in edges:
        edge.load_data(num_classes, folders_per_edge=folders_per_edge)
    
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


def compare_fedavg_fedprox(mnist_base_path, num_edges=10, num_rounds=10, edge_epochs=1, 
                          folders_per_edge=2, mu_values=[0.0, 0.01, 0.1], verbose=1):
    """
    Compare les performances de FedAvg et FedProx avec différentes valeurs de mu.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        folders_per_edge: Nombre de dossiers (classes) par edge 
        mu_values: Liste des valeurs de mu à tester (0.0 = FedAvg)
        verbose: Niveau de verbosité
    """
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Historiques d'évaluation pour chaque valeur de mu
    histories = {}
    
    # Exécuter les expériences pour chaque valeur de mu
    for mu in mu_values:
        algorithm_name = "FedAvg" if mu == 0.0 else f"FedProx (μ={mu})"
        print(f"\n--- Exécution de {algorithm_name} ---")
        
        start_time = time.time()
        history = run_federated_learning(
            mnist_base_path, 
            test_dataset, 
            FedProxModel, 
            mu=mu,
            num_edges=num_edges, 
            num_rounds=num_rounds, 
            edge_epochs=edge_epochs, 
            input_shape=input_shape, 
            folders_per_edge=folders_per_edge,
            verbose=verbose
        )
        elapsed_time = time.time() - start_time
        
        histories[mu] = {"history": history, "time": elapsed_time}
        
        # Extraire les résultats finaux
        final_loss, final_accuracy = history[-1]
        print(f"{algorithm_name} - Perte finale: {final_loss:.4f}, Précision finale: {final_accuracy:.4f}")
        print(f"Temps d'exécution: {elapsed_time:.2f} secondes")
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    central_model = FedProxModel(input_shape, nbclasses=10)
    train_dataset, _ = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    central_model.fit_it(trains=train_dataset, epochs=num_rounds * edge_epochs, tests=test_dataset, verbose=verbose)
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Tracer les résultats
    plot_comparison(histories, central_accuracy, num_rounds)
    
    return histories, (central_loss, central_accuracy)


def plot_comparison(histories, central_accuracy, num_rounds):
    """
    Génère et sauvegarde des graphiques comparant les performances des différents algorithmes.
    
    Args:
        histories: Dictionnaire contenant les historiques d'évaluation pour chaque valeur de mu
        central_accuracy: Précision du modèle centralisé
        num_rounds: Nombre de rounds de fédération
    """
    # Créer le répertoire pour les figures si nécessaire
    os.makedirs('figures', exist_ok=True)
    
    # Préparer les données pour les graphiques
    rounds = list(range(1, num_rounds + 1))
    
    # Graphique de précision
    plt.figure(figsize=(12, 5))
    
    # Tracer l'accuracy pour chaque valeur de mu
    for mu, data in histories.items():
        history = data["history"]
        label = "FedAvg" if mu == 0.0 else f"FedProx (μ={mu})"
        accuracies = [x[1] for x in history]
        plt.plot(rounds, accuracies, marker='o', label=label)
    
    # Tracer la ligne de référence pour le modèle centralisé
    plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
    
    plt.xlabel('Round')
    plt.ylabel('Précision')
    plt.title('Comparaison de la précision: FedAvg vs FedProx')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    # Graphique de perte
    plt.figure(figsize=(12, 5))
    
    # Tracer la perte pour chaque valeur de mu
    for mu, data in histories.items():
        history = data["history"]
        label = "FedAvg" if mu == 0.0 else f"FedProx (μ={mu})"
        losses = [x[0] for x in history]
        plt.plot(rounds, losses, marker='o', label=label)
    
    plt.xlabel('Round')
    plt.ylabel('Perte')
    plt.title('Comparaison de la perte: FedAvg vs FedProx')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/loss_comparison.png', dpi=300, bbox_inches='tight')
    
    # Graphique du temps d'exécution
    plt.figure(figsize=(8, 5))
    mus = []
    times = []
    
    for mu, data in sorted(histories.items()):
        mus.append("FedAvg" if mu == 0.0 else f"μ={mu}")
        times.append(data["time"])
    
    plt.bar(mus, times)
    plt.xlabel('Algorithme')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Temps d\'exécution par algorithme')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('figures/execution_time.png', dpi=300, bbox_inches='tight')
    
    print("Graphiques sauvegardés dans le dossier 'figures'")
    


def experiment_non_iid_effect(mnist_base_path, num_edges=10, num_rounds=10, edge_epochs=1, verbose=1):
    """
    Expérimente l'effet de différentes distributions non-IID sur FedAvg et FedProx.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge
        verbose: Niveau de verbosité
    """
    print("\n--- Expérience: Effet de la distribution non-IID ---")
    
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Définir différents niveaux de non-IID (nombre de classes par edge)
    non_iid_levels = [1, 2, 5, 10]  # 10 = toutes les classes = IID
    
    # Résultats pour FedAvg et FedProx
    results = {"FedAvg": {}, "FedProx": {}}
    
    # Pour chaque niveau de non-IID
    for folders_per_edge in non_iid_levels:
        distribution_type = "IID" if folders_per_edge == 10 else f"{folders_per_edge}/10 classes par edge"
        print(f"\n--- Distribution: {distribution_type} ---")
        
        # Exécuter FedAvg
        print(f"Exécution de FedAvg avec {distribution_type}")
        fedavg_history = run_federated_learning(
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
        results["FedAvg"][folders_per_edge] = fedavg_history
        
        # Exécuter FedProx
        print(f"Exécution de FedProx avec {distribution_type}")
        fedprox_history = run_federated_learning(
            mnist_base_path, 
            test_dataset, 
            FedProxModel, 
            mu=0.1,  # FedProx avec mu=0.1
            num_edges=num_edges, 
            num_rounds=num_rounds, 
            edge_epochs=edge_epochs, 
            input_shape=input_shape, 
            folders_per_edge=folders_per_edge,
            verbose=verbose
        )
        results["FedProx"][folders_per_edge] = fedprox_history
    
    # Tracer les résultats
    plot_non_iid_comparison(results, num_rounds)
    
    return results


def plot_non_iid_comparison(results, num_rounds):
    """
    Génère et sauvegarde des graphiques comparant les performances de FedAvg et FedProx
    pour différents niveaux de non-IID.
    
    Args:
        results: Dictionnaire contenant les historiques d'évaluation pour chaque configuration
        num_rounds: Nombre de rounds de fédération
    """
    # Créer le répertoire pour les figures si nécessaire
    os.makedirs('figures', exist_ok=True)
    
    # Préparer les données pour les graphiques
    rounds = list(range(1, num_rounds + 1))
    non_iid_levels = sorted(results["FedAvg"].keys())
    
    # Graphique de l'impact de la non-IID sur la précision finale
    plt.figure(figsize=(12, 6))
    
    fedavg_final_accuracies = []
    fedprox_final_accuracies = []
    
    for level in non_iid_levels:
        fedavg_final_accuracies.append(results["FedAvg"][level][-1][1])
        fedprox_final_accuracies.append(results["FedProx"][level][-1][1])
    
    x_labels = ["IID" if level == 10 else f"{level}/10" for level in non_iid_levels]
    x_pos = np.arange(len(x_labels))
    
    width = 0.35
    plt.bar(x_pos - width/2, fedavg_final_accuracies, width, label='FedAvg')
    plt.bar(x_pos + width/2, fedprox_final_accuracies, width, label='FedProx (μ=0.1)')
    
    plt.xlabel('Niveau de non-IID (classes par edge)')
    plt.ylabel('Précision finale')
    plt.title('Impact du niveau de non-IID sur la précision finale')
    plt.xticks(x_pos, x_labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('figures/non_iid_impact.png', dpi=300, bbox_inches='tight')
    
    # Graphique de la convergence pour chaque niveau de non-IID
    for level in non_iid_levels:
        plt.figure(figsize=(10, 5))
        
        # Extraire les précisions
        fedavg_accuracies = [x[1] for x in results["FedAvg"][level]]
        fedprox_accuracies = [x[1] for x in results["FedProx"][level]]
        
        plt.plot(rounds, fedavg_accuracies, marker='o', label='FedAvg')
        plt.plot(rounds, fedprox_accuracies, marker='s', label='FedProx (μ=0.1)')
        
        distribution_type = "IID" if level == 10 else f"{level}/10 classes par edge"
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.title(f'Convergence avec {distribution_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'figures/convergence_classes_{level}.png', dpi=300, bbox_inches='tight')
    
    print("Graphiques de l'expérience sur la non-IID sauvegardés dans le dossier 'figures'")


if __name__ == "__main__":
    # Définir le chemin vers le dataset MNIST
    mnist_base_path = '/Users/moneillon/Programmes/menez2/MNIST/trainingSet/trainingSet/'
    
    # S'assurer que le chemin est correct
    while not os.path.exists(mnist_base_path):
        print(f"Le chemin {mnist_base_path} n'existe pas.")
        mnist_base_path = input("Veuillez entrer le chemin correct vers le dataset MNIST: ")
    
    # Paramètres de l'expérience
    num_edges = 10
    num_rounds = 30
    edge_epochs = 1
    verbose = 1
    
    # Comparer FedAvg et FedProx
    print("\n=== Comparaison de FedAvg et FedProx ===")
    compare_fedavg_fedprox(
        mnist_base_path, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        folders_per_edge=2,  # Distribution non-IID: 2 classes par edge
        mu_values=[0.0, 0.01, 0.1],  # 0.0 = FedAvg, 0.01 et 0.1 = FedProx
        verbose=verbose
    )
    
    # Expérimenter l'effet de différentes distributions non-IID
    print("\n=== Expérimentation de l'effet de la non-IID ===")
    experiment_non_iid_effect(
        mnist_base_path, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        verbose=verbose
    )