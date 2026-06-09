import ast
import os
import sys
import time
import random
from pathlib import Path
import multiprocessing as mp
import pandas as pd  # type:ignore
import numpy as np  # type:ignore

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.optimization.helper_optimization as ho
import src.optimization.GA.GA as ga
import src.optimization.experiments.helper_experiment as he
import src.optimization.GA.graph_metric.graph_normalization as gn
import src.assign_to_nodes.utils.class_node_assigner as cna
from src.optimization.experiments.graph_metrics.alpha_screening.alpha_screening import (
    parse_weights_from_string,
)

# Set random seeds
np.random.seed(1)
random.seed(1)

# Define cities to analyze
CITIES = [
    # "València",
    # "Sevilla",
    # "Zaragoza",
    # "Málaga",
    # "Murcia",
    # "Palma",
    "Palmas de Gran Canaria",
    # "Bilbao",    
]

# GA hyperparameters (imported from alpha_screening.py)
SELECTION_STRATEGY = 'tournament'
CROSSOVER_STRATEGY = 'greedy'
POPULATION_SIZE = 50
GENERATIONS = 10_000  # there is early stopping
MUTATION_RATE = 0.05
ELITE_FRACTION = 0.05

def compute_shared_data(city, epsg=25830, root='./'):
    """
    Load data for a specific city and return necessary variables for GA experiments.
    
    Args:
        city (str): City name in the format "City"
        
    Returns:
        tuple: (df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE)
            - df: GeoDataFrame with node attributes
            - G: NetworkX graph
            - distance_matrix: All-pairs shortest path distances
            - id_to_idx: Mapping from node IDs to matrix indices
            - idx_to_id: Mapping from matrix indices to node IDs
            - distance_matrix_binary: Binary matrix indicating if nodes are beyond min distance
            - STATION_MIN_DISTANCE: Minimum allowed distance between stations
    """    

    if city == "Barcelona":
        graph_path = f"/{RAW_GRAPH}/"
    else:
        graph_path = f"/{RAW_GRAPH}/different_cities/"

    # Set up paths for the specific city
    node_assigner = cna.NodeAttributesAssigner(city, graph_path=graph_path, crs=epsg, buffer_size=300, root=root)
    G = node_assigner.G
    df = node_assigner.node_attributes    
    
    # Load configuration for the city
    STATION_MIN_DISTANCE = 300  # Default minimum distance between stations
    
    # Compute distance matrix and mappings
    distance_matrix, id_to_idx, idx_to_id = ho.compute_all_pairs_shortest_paths_dijkstra(city, G, weight='weight', root=root)

    
    return df, G, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE


def run_city_alpha_experiment(city, N_STATIONS, weights, alpha, experiment_idx, experiment_output_file, 
                            score_combination='exponential', penalty_power=3, shared_data=None):
    """
    Run a single GA experiment for a specific city with given parameters and alpha value.
    
    Args:
        city (str): City name
        N_STATIONS (int): Number of stations to place
        weights (dict): Dictionary of weights for different metrics
        alpha (float): Alpha value for graph metrics
        experiment_idx (int): Index of the experiment
        experiment_output_file (str): Path to save experiment results
        score_combination (str): Method to combine scores
        penalty_power (int): Power for penalty calculation
        shared_data (tuple): Pre-loaded data for the city (df, G, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE)
    """
    # Initialize variables
    initial_populations = None
    df = None
    G = None
    distance_matrix = None
    id_to_idx = None
    idx_to_id = None
    STATION_MIN_DISTANCE = None

    # Check if the experiment has already been done
    df_results = he.load_existing_experiments(experiment_output_file)
    if not df_results.empty:
        weights_str = str([f"{k}: {v}" for k, v in weights.items() if v != 0])
        existing_experiment = df_results[
            (df_results['city'] == city) &
            (df_results['N_stations'] == N_STATIONS) & 
            (df_results['weights'] == weights_str) &
            (df_results['alpha'].isna() if alpha is None else df_results['alpha'] == alpha)
        ]
        
        if not existing_experiment.empty:
            print(f"  Skipping experiment {experiment_idx} for {city}: N={N_STATIONS}, Alpha={alpha}, Weights={weights}")
            return

    print(f"\nStarting experiment {experiment_idx} for {city} {score_combination}: N={N_STATIONS}, Alpha={alpha}, Weights={weights}")

    # Unpack shared data
    df, G, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE = shared_data

    # Generate initial population for this city - force n_jobs=1 to avoid nested parallelization
    initial_populations = ho.generate_initial_population(N_STATIONS, POPULATION_SIZE, distance_matrix, id_to_idx, STATION_MIN_DISTANCE, n_jobs=1)

    # Compute and normalize scores according to weights
    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), weights)
    logging_rate = 10

    # Execute genetic algorithm
    start_time = time.time()
    graph_score = True if alpha is not None else False
    best_nodes, best_score, ga_tracking = ga.genetic_algorithm(
        df=df_weighted, 
        population=initial_populations,
        distance_matrix=distance_matrix,
        station_min_distance=STATION_MIN_DISTANCE,
        id_to_idx=id_to_idx, 
        idx_to_id=idx_to_id,
        generations=GENERATIONS,
        elite_proportion=ELITE_FRACTION,
        mutation_rate=MUTATION_RATE,
        selection_strategy=SELECTION_STRATEGY,
        crossover_strategy=CROSSOVER_STRATEGY,
        G=G, 
        graph_score=graph_score,
        alpha=alpha,
        metric_bounds=None,
        score_combination=score_combination,
        penalty_power=penalty_power,
        logging_rate=logging_rate
    )
    generation_scores, _, _, _, _, _ = ga_tracking
    execution_time = (time.time() - start_time) / 60  # minutes

    # Create result dictionary
    result = {
        "city": city,
        "N_stations": N_STATIONS,
        "alpha": alpha,
        "best_score": best_score,
        "minutes_to_complete": execution_time,
        "generations": len(generation_scores) * logging_rate,
        "weights": str([f"{k}: {v}" for k, v in weights.items() if v != 0]),
        "best_solution": str(best_nodes),
        "score_combination": score_combination,
        "penalty_power": penalty_power if score_combination == 'power_penalty' else None
    }
    
    df_results = pd.DataFrame([result])
    he.save_experiment(df_results, experiment_output_file, experiment_idx)


def run_all_cities_alpha_experiments(n_stations_values, weight_combinations, alpha_values, score_combination, 
                                   penalty_power, experiment_dir, experiments_output_file, processes):
    """
    Run GA experiments for different cities, alpha values and weight combinations.
    Process one city at a time, loading its data once and running all experiments.
    """
    start_time = time.time()

    # Save parameters in a separate file
    config_info = {
        'cities': CITIES,
        'N_stations': n_stations_values,
        'selection_strategy': SELECTION_STRATEGY,
        'crossover_strategy': CROSSOVER_STRATEGY,
        'population_size': POPULATION_SIZE,
        'mutation_rate': MUTATION_RATE,
        'elite_fraction': ELITE_FRACTION,
        'max_generations': GENERATIONS,
        'early_stopping': True,
        'alpha_values': alpha_values,
        'score_combination': score_combination,
        'penalty_power': penalty_power,
        'number_of_weight_combinations': len(weight_combinations)
    }
    
    config_file = f"{experiment_dir}/experiment_config_{score_combination}.txt"
    if score_combination == 'power_penalty':
        config_file = f"{experiment_dir}/experiment_config_{score_combination}_{penalty_power}.txt"
    
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            if key == 'alpha_values':
                f.write(f"{key}: {[round(float(x), 1) for x in value]}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"Config file saved to {config_file}")
    
    # Check existing experiments
    df_results = he.load_existing_experiments(experiments_output_file)
    
    # Create a unique identifier for each experiment
    def get_experiment_id(city, n_stations, weights, alpha):
        weights_str = str([f"{k}: {v}" for k, v in weights.items() if v != 0])
        return f"{city}_{n_stations}_{weights_str}_{alpha}"
    
    # Create a mapping of experiment IDs to their indices
    completed_experiments = {}
    if not df_results.empty:
        for _, row in df_results.iterrows():
            exp_id = get_experiment_id(row['city'], row['N_stations'], 
                                     parse_weights_from_string(row['weights']), 
                                     row['alpha'])
            completed_experiments[exp_id] = row['experiment_idx']
    
    # Process each city sequentially
    experiment_counter = max(completed_experiments.values()) + 1 if completed_experiments else 1
    
    for city in CITIES:
        print(f"\nProcessing city: {city}")
        
        # Load data for this city once
        print(f"Loading data for {city}...")
        shared_data = compute_shared_data(city)
        
        # Prepare experiment arguments for this city
        city_experiment_args = []
        
        for weights in weight_combinations:
            # First run without graph metrics
            for n_stations in n_stations_values:
                exp_id = get_experiment_id(city, n_stations, weights, None)
                if exp_id not in completed_experiments:
                    args = (
                        city, n_stations, weights, None, experiment_counter, 
                        experiments_output_file, score_combination, penalty_power, shared_data
                    )
                    city_experiment_args.append(args)
                    experiment_counter += 1
            
            # Then run with graph metrics for each alpha
            for alpha in alpha_values:
                for n_stations in n_stations_values:
                    exp_id = get_experiment_id(city, n_stations, weights, alpha)
                    if exp_id not in completed_experiments:
                        args = (
                            city, n_stations, weights, alpha, experiment_counter, 
                            experiments_output_file, score_combination, penalty_power, shared_data
                        )
                        city_experiment_args.append(args)
                        experiment_counter += 1
        
        if not city_experiment_args:
            print(f"All experiments for {city} have been completed!")
            continue
            
        print(f"Running {len(city_experiment_args)} experiments for {city}")
        
        # If processes=1, run sequentially
        if processes == 1:
            for args in city_experiment_args:
                run_city_alpha_experiment(*args)
        else:
            # Use multiprocessing for multiple processes
            try:
                with mp.Pool(processes=processes) as pool:
                    pool.starmap(run_city_alpha_experiment, city_experiment_args)
            except Exception as e:
                print(f"\nError in parallel execution for {city}: {e}")
                print(f"Error type: {type(e).__name__}")
                # Continue with next city even if there's an error
                continue
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time/3600:.2f} hours")
    print(f"Results saved to {experiments_output_file}")

if __name__ == "__main__":
    # Define score combination
    score_combination = 'multiply'  # Options: 'multiply', 'exponential', 'power_penalty'
    penalty_power = None  # Higher power for more severe penalties
    
    # Create results directory
    RESULTS_DIR = f"{PR_EXP}/GA_alpha_screening/different_cities/"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if score_combination == 'power_penalty':
        RESULTS_FILE = f"{RESULTS_DIR}/alpha_comparison_{score_combination}_{penalty_power}.csv"
    else:
        RESULTS_FILE = f"{RESULTS_DIR}/alpha_comparison_{score_combination}.csv"

    # Define weight combinations
    weight_combinations = [
        {'population': 1}
    ]

    # Define alpha values to test
    alpha_values = [round(x, 1) for x in np.arange(0, 1.1, 0.2)]
    n_stations_values = [30]
    
    # Run experiments
    processes = 7  # Use multiple processes
    run_all_cities_alpha_experiments(
        n_stations_values, weight_combinations, alpha_values,
        score_combination, penalty_power,
        RESULTS_DIR, RESULTS_FILE, processes
    )
