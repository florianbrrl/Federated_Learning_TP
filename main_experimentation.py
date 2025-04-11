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
                          num_edges=10, num_rounds=10, edge_epochs=10, 
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


def compare_fedavg_fedprox(mnist_base_path, num_edges=10, num_rounds=10, edge_epochs=10, 
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
    

def experiment_non_iid_effect(mnist_base_path, num_edges=10, num_rounds=10, edge_epochs=10, verbose=1):
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


def run_experiments(mnist_base_path, configurations, num_rounds=30, verbose=1):
    """
    Exécute des expériences d'apprentissage fédéré selon les configurations spécifiées.
    
    Args:
        mnist_base_path: Chemin de base vers les données MNIST
        configurations: Liste des configurations à tester
        num_rounds: Nombre de rounds de fédération
        verbose: Niveau de verbosité
    
    Returns:
        Dictionnaire contenant les résultats pour chaque configuration
    """
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # Charger les données de test centralisées
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(mnist_base_path, verbose=verbose)
    # Créer le jeu de données de test
    _, test_dataset = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    
    # Entraîner un modèle centralisé pour comparaison
    print("\n--- Entraînement du modèle centralisé pour comparaison ---")
    central_model = FedProxModel(input_shape, nbclasses=10)
    train_dataset, _ = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=verbose)
    central_model.fit_it(trains=train_dataset, epochs=num_rounds * 3, tests=test_dataset, verbose=verbose)
    central_loss, central_accuracy = central_model.evaluate(test_dataset, verbose=verbose)
    print(f"\nModèle centralisé - Perte: {central_loss:.4f}, Précision: {central_accuracy:.4f}")
    
    # Exécuter chaque configuration
    for config in configurations:
        name = config['name']
        num_clients = config['num_clients']
        edge_epochs = config['epochs']
        
        # Gérer la distribution IID vs non-IID
        # Pour IID, on donne toutes les classes (10)
        # Pour non-IID, on limite à 2 classes par client
        folders_per_edge = 10 if config['distribution'] == 'iid' else 2
        
        # Gérer l'algorithme
        if config['algo'] == 'fedavg':
            mu = 0.0
            algo_name = "FedAvg"
        elif config['algo'] == 'fedprox':
            mu = 0.1
            algo_name = "FedProx"
        elif config['algo'] == 'fedsgd':
            mu = 0.0
            algo_name = "FedSGD"
            # Pour FedSGD, on pourrait ajuster le taux d'apprentissage
            # Cela dépendrait de l'implémentation de ton FedProxModel
            # Par exemple: learning_rate = config.get('lr', 0.01)
        else:
            # Algorithme par défaut
            mu = 0.0
            algo_name = "Default"
        
        print(f"\n--- Exécution de {name}: {algo_name}, {num_clients} clients, {edge_epochs} époques ---")
        
        # Mesurer le temps d'exécution
        start_time = time.time()
        
        # Exécuter l'apprentissage fédéré
        history = run_federated_learning(
            mnist_base_path,
            test_dataset,
            FedProxModel,
            mu=mu,
            num_edges=num_clients,
            num_rounds=num_rounds,
            edge_epochs=edge_epochs,
            input_shape=input_shape,
            folders_per_edge=folders_per_edge,
            verbose=verbose
        )
        
        elapsed_time = time.time() - start_time
        
        # Stocker les résultats
        final_loss, final_accuracy = history[-1]
        results[name] = {
            "history": history,
            "time": elapsed_time,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "config": config
        }
        
        print(f"{name} - Perte finale: {final_loss:.4f}, Précision finale: {final_accuracy:.4f}")
        print(f"Temps d'exécution: {elapsed_time:.2f} secondes")
    
    # Tracer les résultats
    plot_experiment_results(results, central_accuracy, num_rounds)
    
    return results, (central_loss, central_accuracy)


def plot_experiment_results(results, central_accuracy, num_rounds):
    """
    Génère et sauvegarde des graphiques comparant les performances des différentes configurations.
    
    Args:
        results: Dictionnaire contenant les résultats de chaque configuration
        central_accuracy: Précision du modèle centralisé
        num_rounds: Nombre de rounds de fédération
    """
    # Créer le répertoire pour les figures si nécessaire
    os.makedirs('figures', exist_ok=True)
    
    # Préparer les données pour les graphiques
    rounds = list(range(1, num_rounds + 1))
    
    # --- Comparaison du nombre de clients ---
    # Filtrer les résultats pour la comparaison du nombre de clients
    client_configs = [k for k in results.keys() if k.startswith('Clients_')]
    
    if client_configs:
        plt.figure(figsize=(12, 5))
        for config_name in client_configs:
            accuracies = [x[1] for x in results[config_name]["history"]]
            plt.plot(rounds, accuracies, marker='o', label=config_name)
        
        plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.title('Impact du nombre de clients sur la précision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/client_number_comparison.png', dpi=300, bbox_inches='tight')
    
    # --- Comparaison des algorithmes ---
    # Filtrer les résultats pour la comparaison des algorithmes
    algo_configs = ['FedAvg_10', 'FedProx_10']
    if 'FedSGD_LR0.1' in results:
        algo_configs.append('FedSGD_LR0.1')
    
    valid_algo_configs = [k for k in algo_configs if k in results]
    
    if valid_algo_configs:
        plt.figure(figsize=(12, 5))
        for config_name in valid_algo_configs:
            accuracies = [x[1] for x in results[config_name]["history"]]
            plt.plot(rounds, accuracies, marker='o', label=config_name)
        
        plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.title('Comparaison des algorithmes d\'agrégation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    
    # --- Comparaison IID vs Non-IID ---
    if 'NonIID_10clients' in results and 'FedAvg_10' in results:
        plt.figure(figsize=(12, 5))
        
        iid_accuracies = [x[1] for x in results['FedAvg_10']["history"]]
        noniid_accuracies = [x[1] for x in results['NonIID_10clients']["history"]]
        
        plt.plot(rounds, iid_accuracies, marker='o', label='IID (10 clients)')
        plt.plot(rounds, noniid_accuracies, marker='s', label='Non-IID (10 clients)')
        
        plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.title('Impact de la distribution des données (IID vs Non-IID)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/iid_vs_noniid.png', dpi=300, bbox_inches='tight')
    
    # --- Comparaison du nombre d'époques ---
    epochs_configs = ['Epochs_1', 'Epochs_3', 'Epochs_5'] 
    valid_epochs_configs = [k for k in epochs_configs if k in results]
    
    if len(valid_epochs_configs) > 1:
        plt.figure(figsize=(12, 5))
        for config_name in valid_epochs_configs:
            accuracies = [x[1] for x in results[config_name]["history"]]
            plt.plot(rounds, accuracies, marker='o', label=config_name)
        
        plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
        plt.xlabel('Round')
        plt.ylabel('Précision')
        plt.title('Impact du nombre d\'époques locales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/local_epochs_comparison.png', dpi=300, bbox_inches='tight')
    
    # --- Temps d'exécution ---
    plt.figure(figsize=(14, 6))
    names = list(results.keys())
    times = [results[name]["time"] for name in names]
    
    # Trier par temps d'exécution
    sorted_indices = np.argsort(times)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    
    plt.bar(sorted_names, sorted_times)
    plt.xlabel('Configuration')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Temps d\'exécution par configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/execution_time_all.png', dpi=300, bbox_inches='tight')
    
    # --- Précisions finales ---
    plt.figure(figsize=(14, 6))
    final_accuracies = [results[name]["final_accuracy"] for name in names]
    
    # Trier par précision finale
    sorted_indices = np.argsort(final_accuracies)[::-1]  # Ordre décroissant
    sorted_names = [names[i] for i in sorted_indices]
    sorted_accuracies = [final_accuracies[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_names, sorted_accuracies)
    plt.axhline(y=central_accuracy, color='r', linestyle='--', label='Centralisé')
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Configuration')
    plt.ylabel('Précision finale')
    plt.title('Précision finale par configuration')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/final_accuracy_all.png', dpi=300, bbox_inches='tight')
    
    print("Graphiques sauvegardés dans le dossier 'figures'")


if __name__ == "__main__":
    # Définir le chemin vers le dataset MNIST
    mnist_base_path = '/Users/moneillon/Programmes/menez2/MNIST/trainingSet/trainingSet/'
    
    # S'assurer que le chemin est correct
    while not os.path.exists(mnist_base_path):
        print(f"Le chemin {mnist_base_path} n'existe pas.")
        mnist_base_path = input("Veuillez entrer le chemin correct vers le dataset MNIST: ")
    
    # Définir les configurations des expériences
    configurations = [
        # Nombre de clients
        {'name': 'Clients_5', 'num_clients': 5, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3},
        {'name': 'Clients_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3},
        {'name': 'Clients_20', 'num_clients': 20, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3},
        
        # Distribution des données
        {'name': 'NonIID_10clients', 'num_clients': 10, 'distribution': 'non_iid', 'algo': 'fedavg', 'epochs': 3},
        
        # Algorithmes d'agrégation
        {'name': 'FedAvg_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 3},
        {'name': 'FedProx_10', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedprox', 'epochs': 3},
        
        # Époques locales
        {'name': 'Epochs_1', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 1},
        {'name': 'Epochs_5', 'num_clients': 10, 'distribution': 'iid', 'algo': 'fedavg', 'epochs': 5},
    ]
    
    # Paramètres de l'expérience
    num_rounds = 30
    verbose = 1
    
    # Choix du mode d'expérimentation
    print("\n=== Choix du mode d'expérimentation ===")
    print("1. Expérimentations avec les configurations personnalisées")
    print("2. Comparaison traditionnelle FedAvg et FedProx")
    print("3. Expérimentation de l'effet non-IID")
    choice = input("\nVeuillez choisir le mode d'expérimentation (1, 2 ou 3): ")
    
    if choice == '1':
        # Exécuter les expériences avec les configurations personnalisées
        print("\n=== Exécution des expériences avec configurations personnalisées ===")
        results, central_results = run_experiments(
            mnist_base_path, 
            configurations, 
            num_rounds=num_rounds, 
            verbose=verbose
        )
        
        # Afficher un résumé des résultats
        print("\n=== Résumé des résultats ===")
        print(f"Modèle centralisé - Précision: {central_results[1]:.4f}")
        
        for name, data in results.items():
            print(f"{name} - Précision: {data['final_accuracy']:.4f}, Temps: {data['time']:.2f}s")
    
    elif choice == '2':
        # Comparer FedAvg et FedProx (méthode originale)
        print("\n=== Comparaison de FedAvg et FedProx ===")
        compare_fedavg_fedprox(
            mnist_base_path, 
            num_edges=10, 
            num_rounds=num_rounds, 
            edge_epochs=3, 
            folders_per_edge=2,
            mu_values=[0.0, 0.01, 0.1],
            verbose=verbose
        )
    
    elif choice == '3':
        # Expérimenter l'effet de différentes distributions non-IID (méthode originale)
        print("\n=== Expérimentation de l'effet de la non-IID ===")
        experiment_non_iid_effect(
            mnist_base_path, 
            num_edges=10, 
            num_rounds=num_rounds, 
            edge_epochs=3,
            verbose=verbose
        )
    
    else:
        print("Choix invalide. Fin du programme.")