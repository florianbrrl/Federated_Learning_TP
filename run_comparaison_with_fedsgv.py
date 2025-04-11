"""
Script principal pour comparer les méthodes FedSGD, FedAvg, FedProx standard et FedProx avec μ adaptatif.
Ce script génère des graphiques comparant les performances des différentes approches.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random
from collections import Counter

# Importer les modules personnalisés
from fedprox_model import FedProxModel
from federated_server import FederatedServer
from adaptive_edge_node import AdaptiveEdgeNode
from adaptive_federated_server import AdaptiveFederatedServer
from adaptive_mu_calculator import extract_class_distribution
from fedsgd_implementation import run_federated_learning_sgd

# Importer les fonctions de run_adaptive_fedprox.py
from run_adaptive_fedprox import (
    run_federated_learning_adaptive_mu,
    run_federated_learning_standard
)

# Importation des modules de base pour les données et l'évaluation
import fl_dataquest


def compare_all_methods(mnist_base_path, num_edges=10, num_rounds=20, edge_epochs=1, 
                      folders_per_edge=2, verbose=1):
    """
    Compare les performances de FedSGD, FedAvg, FedProx standard et FedProx avec μ adaptatif.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        num_edges: Nombre d'edges participants
        num_rounds: Nombre de rounds de fédération
        edge_epochs: Nombre d'époques d'entraînement par edge (pour FedAvg et FedProx)
        folders_per_edge: Nombre de dossiers (classes) par edge 
        verbose: Niveau de verbosité
        
    Returns:
        Dictionnaire des historiques d'évaluation, historique de μ, et métriques du modèle centralisé
    """
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Exécuter FedSGD
    print("\n--- Exécution de FedSGD ---")
    fedsgd_history = run_federated_learning_sgd(
        mnist_base_path, 
        test_dataset, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        input_shape=input_shape, 
        folders_per_edge=folders_per_edge,
        verbose=verbose
    )
    
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
        verbose=verbose
    )
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    central_model = FedProxModel(input_shape, nbclasses=10)
    train_dataset, _ = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    central_model.fit_it(trains=train_dataset, epochs=num_rounds * edge_epochs, tests=test_dataset, verbose=verbose)
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Tracer les résultats avec FedSGD
    plot_comparison_with_fedsgd(fedsgd_history, fedavg_history, fedprox_history, adaptive_history, 
                              central_accuracy, mu_history, num_rounds)
    
    return {
        "FedSGD": fedsgd_history,
        "FedAvg": fedavg_history,
        "FedProx": fedprox_history,
        "AdaptiveFedProx": adaptive_history
    }, mu_history, (central_loss, central_accuracy)


def plot_comparison_with_fedsgd(fedsgd_history, fedavg_history, fedprox_history, adaptive_history, 
                              central_accuracy, mu_history, num_rounds):
    """
    Génère et sauvegarde des graphiques comparant les performances des algorithmes, incluant FedSGD.
    
    Args:
        fedsgd_history: Historique d'évaluation pour FedSGD
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
    fedsgd_accuracies = [x[1] for x in fedsgd_history]
    fedavg_accuracies = [x[1] for x in fedavg_history]
    fedprox_accuracies = [x[1] for x in fedprox_history]
    adaptive_accuracies = [x[1] for x in adaptive_history]
    
    # Graphique de précision
    plt.figure(figsize=(14, 7))
    plt.plot(rounds, fedsgd_accuracies, marker='d', label='FedSGD')
    plt.plot(rounds, fedavg_accuracies, marker='o', label='FedAvg')
    plt.plot(rounds, fedprox_accuracies, marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, adaptive_accuracies, marker='^', label='FedProx (μ adaptatif)')
    plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Comparaison de la précision: FedSGD vs FedAvg vs FedProx vs FedProx Adaptatif', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/accuracy_comparison_all.png', dpi=300, bbox_inches='tight')
    
    # Graphique des pertes
    plt.figure(figsize=(14, 7))
    plt.plot(rounds, [x[0] for x in fedsgd_history], marker='d', label='FedSGD')
    plt.plot(rounds, [x[0] for x in fedavg_history], marker='o', label='FedAvg')
    plt.plot(rounds, [x[0] for x in fedprox_history], marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, [x[0] for x in adaptive_history], marker='^', label='FedProx (μ adaptatif)')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Perte', fontsize=12)
    plt.title('Comparaison de la perte: FedSGD vs FedAvg vs FedProx vs FedProx Adaptatif', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/loss_comparison_all.png', dpi=300, bbox_inches='tight')
    
    # Graphique de précision relative (normalisée par rapport à FedSGD)
    plt.figure(figsize=(14, 7))
    fedsgd_base = np.array(fedsgd_accuracies)
    plt.plot(rounds, np.ones_like(fedsgd_base), marker='d', label='FedSGD (référence)')
    plt.plot(rounds, np.array(fedavg_accuracies) / fedsgd_base, marker='o', label='FedAvg')
    plt.plot(rounds, np.array(fedprox_accuracies) / fedsgd_base, marker='s', label='FedProx (μ=0.1)')
    plt.plot(rounds, np.array(adaptive_accuracies) / fedsgd_base, marker='^', label='FedProx (μ adaptatif)')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Précision relative à FedSGD', fontsize=12)
    plt.title('Gain de précision par rapport à FedSGD', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/relative_accuracy.png', dpi=300, bbox_inches='tight')
    
    print("Graphiques sauvegardés dans le dossier 'figures'")


if __name__ == "__main__":
    # Définir le chemin vers le dataset MNIST
    mnist_base_path = input("Entrez le chemin vers votre dataset MNIST: ")
    
    # S'assurer que le chemin est correct
    while not os.path.exists(mnist_base_path):
        print(f"Le chemin {mnist_base_path} n'existe pas.")
        mnist_base_path = input("Veuillez entrer le chemin correct vers le dataset MNIST: ")
    
    # Paramètres de l'expérience
    num_edges = 10
    num_rounds = 20
    edge_epochs = 1
    folders_per_edge = 2  # Distribution non-IID: 2 classes par edge
    verbose = 1
    
    # Comparer toutes les méthodes
    print("\n=== Comparaison de FedSGD, FedAvg, FedProx standard et FedProx avec μ adaptatif ===")
    results, mu_history, central_metrics = compare_all_methods(
        mnist_base_path, 
        num_edges=num_edges, 
        num_rounds=num_rounds, 
        edge_epochs=edge_epochs, 
        folders_per_edge=folders_per_edge,
        verbose=verbose
    )
    
    # Afficher les résultats finaux
    print("\n=== Résultats finaux ===")
    print(f"FedSGD - Précision finale: {results['FedSGD'][-1][1]:.4f}")
    print(f"FedAvg - Précision finale: {results['FedAvg'][-1][1]:.4f}")
    print(f"FedProx standard - Précision finale: {results['FedProx'][-1][1]:.4f}")
    print(f"FedProx μ adaptatif - Précision finale: {results['AdaptiveFedProx'][-1][1]:.4f}")
    print(f"Modèle centralisé - Précision: {central_metrics[1]:.4f}")