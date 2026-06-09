"""
Genetic Algorithm (GA) for Bike-Sharing System (BSS) Optimization

This module implements a genetic algorithm for optimizing BSS station placement. 
It provides advanced selection, crossover, and mutation operators specifically designed 
for BSS optimization problems.

The availableGA operators are:
1. **Parent Selection**: Roulette-wheel and tournament selection strategies (roulette-wheel and tournament selection)
2. **Elite Selection**: Keep the best solutions from the previous generation according to their fitness score
3. **Crossover Operations**: Multiple strategies for combining parent solutions (greedy, top-first, weighted random)
4. **Mutation Operations**: Mutation for maintaining solution diversity

Key Features:
- **Multi-objective**: Configurable weights for different optimization criteria
- **Constraint Handling**: Minimum distance constraints between stations and the number of stations
- **Graph Metrics**: Integration with network analysis metrics to account for the network structure
    that considers the proximity among the stations and the accessibility of the system

Author: Jordi Grau Escolano
"""

import sys
import os
from pathlib import Path
import datetime
import random
import time
import pandas as pd  # type:ignore
import geopandas as gpd  # type:ignore
import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import contextily as ctx  # type:ignore
from shapely import wkt  # type:ignore

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.helper_optimization as oh
import src.optimization.GA.helper_crossover_copy as ch
import src.optimization.GA.helper_mutate as hm
import src.optimization.GA.graph_metric.graph_normalization as gn

np.random.seed(1)
random.seed(1)

def select_parents(pop, fitness_scores, strategy='roulette', tournament_size=5):
    """
    Selects two parents from the population using either roulette-wheel or tournament selection.

    Args:
        pop (list): The population from which to select parents.
        fitness_scores (np.ndarray): Array of fitness scores corresponding to the population.
        strategy (str): The selection strategy to use ('roulette' or 'tournament').
        tournament_size (int): The number of candidates to consider in each tournament (only used for tournament selection).

    Returns:
        tuple: A tuple containing two selected parent solutions.

    Raises:
        ValueError: If an invalid selection strategy is provided.
    """
    if strategy == 'roulette':
        probabilities = fitness_scores / fitness_scores.sum()
        parent_indices = np.random.choice(len(pop), size=2, p=probabilities, replace=False)
        return pop[parent_indices[0]], pop[parent_indices[1]]
    
    elif strategy == 'tournament':
        if len(pop) < tournament_size:
            tournament_size = len(pop)  # Avoid errors when the population is small

        # Array for tournament selection
        parent_indices = np.empty(2, dtype=int)
        for i in range(2):  # Select two parents
            candidates = np.random.choice(len(pop), size=tournament_size, replace=False)
            parent_indices[i] = candidates[np.argmax(fitness_scores[candidates])]
        return pop[parent_indices[0]], pop[parent_indices[1]]

    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    

def crossover(parent1, parent2, node_scores, N, strategy, distance_matrix, 
              id_to_idx, idx_to_id, station_min_distance=0, graph_score=False, alpha=0.5, metric_bounds=None,
              required_nodes=[]):
    """
    Generates a child solution by combining nodes from two parent solutions and
    selecting nodes based on different strategies:
      - 'greedy': Sort by descending score (fully greedy).
      - 'top_first': Shuffle the top half, then pick from the rest.
      - 'weighted_random': Pick nodes stochastically, weighted by their scores.

    Args:
        parent1 (list): Node IDs of the first parent solution.
        parent2 (list): Node IDs of the second parent solution.
        node_scores (dict): Maps node IDs to their scores.
        N (int): Number of nodes required in the child solution.
        strategy (str): 'greedy', 'top_first', or 'weighted_random'.
        distance_matrix (np.ndarray): 1 => valid, 0 => invalid (too close).
        id_to_idx (dict): Maps node IDs to distance_matrix indices.
        idx_to_id (dict): Maps distance_matrix indices to node IDs.
        station_min_distance (int): Minimum distance between stations in meters.
        graph_score (bool): Whether to use graph-based metrics (default False)
        alpha (float): Weighting factor for dispersion vs accessibility (default 0.5)
        metric_bounds (tuple): Bounds for graph-based metrics (default None)
        required_nodes (list): List of node IDs that must be included in the solution

    Returns:
        list: A valid child solution (list of node IDs) of length N.

    Raises:
        ValueError: If no valid child of size N can be formed, even after fallback.
    """
    # 1. Handle required nodes
    if required_nodes is None:
        required_nodes = []
        
    N = N - len(required_nodes)
    
    if N < 0:
        raise ValueError("More required nodes than N")
    
    if N == 0:
        return required_nodes
        
    # 2. Combine unique nodes from both parents, excluding required nodes
    combined_nodes = list(set(parent1 + parent2) - set(required_nodes))
    if not combined_nodes:
        raise ValueError("No candidate nodes from parents.")

    # 3. Pick nodes according to the specified strategy
    if strategy == 'greedy':
        graph_score = False
        new_solution = ch._crossover_greedy(N, combined_nodes, node_scores, distance_matrix, id_to_idx, idx_to_id, station_min_distance,
                                             graph_score, alpha, metric_bounds)
    elif strategy == 'weighted_random':
        new_solution = ch._crossover_weighted_random(N, combined_nodes, node_scores, distance_matrix, id_to_idx, station_min_distance,
                                            graph_score, alpha, metric_bounds)
    elif strategy == 'top_first':
        new_solution = ch._crossover_top_first(N, combined_nodes, node_scores, distance_matrix, id_to_idx, station_min_distance)
    else:
        raise ValueError(f"Unknown crossover strategy: {strategy}")

    # 4. Add required nodes back to solution
    if new_solution and len(new_solution) == N:
        return required_nodes + new_solution

    # 5. Fallback strategy: random-based fill
    print("Crossover partial or failed; applying fallback.")
    new_solution = ch.fallback_fill(N, [], combined_nodes, distance_matrix, id_to_idx, station_min_distance)
    if new_solution and len(new_solution) == N:
        return required_nodes + new_solution

    raise ValueError("Cannot generate a valid child after applying fallback.")


def mutate(solution, mutation_rate, distance_matrix, id_to_idx, idx_to_id, 
           max_replacements=None, station_min_distance=True, batch_size=32,
           required_nodes=None,
):
    return hm.mutate_population([solution], mutation_rate, distance_matrix, id_to_idx, idx_to_id, 
                     max_replacements, station_min_distance, batch_size, required_nodes)[0]


def genetic_algorithm(
    df, population, 
    distance_matrix, station_min_distance, id_to_idx, idx_to_id, # Distance data
    required_nodes=None,
    generations=10_000, elite_proportion=0.05, mutation_rate=0.1, # Algorithm control
    selection_strategy='tournament', crossover_strategy='greedy',
    G=None, metric_bounds=None, graph_score=False, alpha=0.5, # Graph scoring parameters
    score_combination='multiply', penalty_power=2,  # Score combination parameters
    logging_rate=10,
):
    """
    Implements a genetic algorithm to optimize station locations.
    
    Args:
        df (GeoDataFrame): DataFrame containing node attributes and geometry
        population (list): Initial population of candidate solutions
        
        # Distance data
        distance_matrix (ndarray): Pre-computed matrix of distances between nodes in km
        station_min_distance (float): Minimum distance between stations in km
        id_to_idx (dict): Maps node IDs to matrix indices
        idx_to_id (dict): Maps matrix indices to node IDs

        # Required nodes
        required_nodes (list): List of node IDs that must be included in the solution
        
        # Algorithm control
        generations (int): Maximum number of generations to run (default 10,000)
        elite_proportion (float): Fraction of best solutions to preserve (default 0.05)
        mutation_rate (float): Probability of mutation per node (default 0.1)
        selection_strategy (str): Selection strategy ('roulette' or 'tournament')
        crossover_strategy (str): Crossover strategy ('greedy', 'top_first', or 'weighted_random')
        
        # Graph scoring parameters
        G (networkx.Graph, optional): Graph for network-based metrics
        graph_score (bool): Whether to use graph-based metrics (default False)
        alpha (float): Weighting factor for dispersion vs accessibility (default 0.5)
        
        # Score combination parameters
        score_combination (str): How to combine node and graph scores ('multiply', 'exponential', 'power_penalty')
        penalty_power (float): Power to use for penalizing bad graph scores (default 2)

        # Logging
        logging_rate (int): Number of generations between logging (default 10)
        
        # Optimization
        enable_crossover_optimization (bool): Whether to use optimized crossover functions (default True)
        
    Returns:
        tuple: (best_solution, best_score, ga_tracking) where:
            - best_solution (list): List of node IDs representing best station locations
            - best_score (float): Fitness score of the best solution
            - ga_tracking (tuple): Tracking data for algorithm performance
    """
    current_time = datetime.datetime.now().strftime("%H:%M")
    print(f"{current_time} | Starting GA")
    # Infer key parameters from the population
    population_size = len(population)
    N = len(population[0])
    
    # Convert fitness to dictionary for quick lookups
    nodes_score = df['norm_score'].to_dict()
    
    # Pre-compute graph metric bounds if using graph score and no bounds are provided
    if graph_score and metric_bounds is None:
        min_disp_bound, _ = gn.min_dispersion_bound(N, station_min_distance)
        max_disp_metric, _ = gn.max_dispersion_metric(N, distance_matrix, station_min_distance, idx_to_id)
        min_accessibility_bound, _ = gn.min_accessibility_bound_kmeans(G, N, distance_matrix, id_to_idx)
        max_accessibility_bound, _ = gn.max_accessibility_bound(N, distance_matrix, idx_to_id, station_min_distance)
        dispersion_bounds = (min_disp_bound, max_disp_metric)
        accessibility_bounds = (min_accessibility_bound, max_accessibility_bound)
        
        metric_bounds = (dispersion_bounds, accessibility_bounds)

    # Score initialization
    best_solution, best_score = None, -np.inf
    elite_count = int((elite_proportion * population_size) / 2)

    # Initialize early stopping criteria
    NO_IMPROVEMENT_BEST, NO_IMPROVEMENT_AVG = 300, 300
    best_counter, avg_counter = 0, 0
    best_fitness_seen, avg_fitness_seen = -np.inf, -np.inf

    # Log performance
    generations_population, generation_scores, average_scores = [], [], []
    generation_times, jaccard_diversities, mutation_rates = [], [], [] 
    
    # Validate initial solutions (for debugging)
    if station_min_distance > 0:
        for sol in population:
            number_of_invalid_solutions = 0
            if not oh.validate_solution(
                sol, distance_matrix, id_to_idx, idx_to_id, N, station_min_distance, required_nodes):
                number_of_invalid_solutions += 1
        if number_of_invalid_solutions > 0:
            raise ValueError(f"Invalid solutions found in the initial population: {number_of_invalid_solutions}")
    generations_population.append(population)

    start_time = time.time()
    for generation in range(generations):
        
        # 1. Evaluate fitness
        fitness_scores = oh.calculate_population_scores(
            population, df, distance_matrix, id_to_idx, graph_score, alpha, metric_bounds, score_combination=score_combination, penalty_power=penalty_power)

        # 2. Track best solution and log GA algorithm stats
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_score:
            best_solution, best_score = population[best_idx], fitness_scores[best_idx]

        if (generation + 1) % logging_rate == 0 or generation == generations - 1:
            generation_scores.append(best_score)
            average_scores.append(fitness_scores.mean())
            jaccard_diversities.append(oh.average_jaccard_diversity(population))
            generation_times.append(round(time.time() - start_time, 1))
            mutation_rates.append(mutation_rate)

        # 3. Compute average score
        current_best, current_avg = np.max(fitness_scores), np.mean(fitness_scores)

        # 4a. Early Stopping Mechanism (best)
        if current_best > best_fitness_seen:
            best_fitness_seen, best_counter = current_best, 0
        else:
            best_counter += 1

        # 4b. Early Stopping Mechanism (avg)
        if current_avg > avg_fitness_seen:
            avg_fitness_seen, avg_counter = current_avg, 0
        else:
            avg_counter += 1

        # 4c. Check if early stopping condition is met
        if best_counter >= NO_IMPROVEMENT_BEST and avg_counter >= NO_IMPROVEMENT_AVG:
            print(f"Stopping early -> pop:{population_size}, mut:{mutation_rate}, elite:{elite_proportion} at generation {generation+1} due to stagnation. Best: {best_score:.2f}, Avg: {average_scores[-1]:.2f}")
            break        

        # 5. Get elite solutions
        elite_indices = np.argpartition(-fitness_scores, elite_count)[:elite_count]
        elite_solutions = [population[i] for i in elite_indices]
        elite_scores = fitness_scores[elite_indices]

        # 6. Create next generation
        next_generation = []

        # 6a. Keep some elites directly
        if elite_count > 0:
            next_generation.extend(elite_solutions)

        # 6b. Use elite solutions to get some new children
        tournament_size = min(5, elite_count) if elite_count < 5 else 5
        non_elite_solutions = []  # Keep track of non-elite solutions separately
        if elite_count >= 2:
            while len(non_elite_solutions) < elite_count:
                parent1, parent2 = select_parents(elite_solutions, elite_scores, strategy=selection_strategy, tournament_size=tournament_size)
                child = crossover(
                    parent1, parent2, nodes_score, N, crossover_strategy, 
                    distance_matrix, id_to_idx, idx_to_id, station_min_distance,
                    graph_score=graph_score, alpha=alpha, metric_bounds=metric_bounds,
                    required_nodes=required_nodes
                )
                if len(child) == N:
                    non_elite_solutions.append(child)

        # 6c. Generate the rest from full population
        while len(non_elite_solutions) < population_size - elite_count:
            p1, p2 = select_parents(population, fitness_scores, strategy=selection_strategy, tournament_size=tournament_size)
            child = crossover(
                p1, p2, nodes_score, N, crossover_strategy, 
                distance_matrix, id_to_idx, idx_to_id, station_min_distance,
                graph_score=graph_score, alpha=alpha, metric_bounds=metric_bounds,
                required_nodes=required_nodes
            )
            if len(child) == N:
                non_elite_solutions.append(child)

        # 6d. Mutate only the non-elite solutions
        mutated_solutions = hm.mutate_population(
            non_elite_solutions, mutation_rate, distance_matrix, id_to_idx, idx_to_id, 
            max_replacements=round(len(population[0])*0.2), 
            station_min_distance=station_min_distance, batch_size=32,
            required_nodes=required_nodes
        )
        
        # 6e. Combine elites with mutated solutions
        next_generation.extend(mutated_solutions)
        population = next_generation

        # Print status every X generations
        if (generation + 1) % 100 == 0 or generation == generations - 1:
            current_time = datetime.datetime.now().strftime("%H:%M")
            print(
                f"{current_time} | Gen {generation + 1} | Bst Sc = {best_score:.2f} | "
                f"Avg Sc = {average_scores[-1]:.2f} |  Std Sc = {np.std(fitness_scores):.2f} | "
                f"Jaccard = {oh.average_jaccard_diversity(population):.3f}"
            )

        generations_population.append(population)
    
    # Validate the solutions one last time
    if station_min_distance > 0:
        for sol in population:
            if not oh.validate_solution(
                sol, distance_matrix, id_to_idx, idx_to_id, N, station_min_distance, required_nodes):
                raise ValueError("Invalid solution found in the final population.")
        
    # Change data formats
    best_score = round(best_score, 6)
    generation_scores = [round(float(score), 3) for score in generation_scores]
    average_scores = [round(float(score), 3) for score in average_scores]
    jaccard_diversities = [round(float(diversity), 3) for diversity in jaccard_diversities]

    ga_tracking = (average_scores, generation_scores, generation_times, jaccard_diversities, mutation_rates, generations_population)

    return best_solution, best_score, ga_tracking
