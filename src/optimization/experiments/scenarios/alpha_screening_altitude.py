import os
import sys
import random
import yaml
from pathlib import Path
import multiprocessing as mp
import numpy as np  # type:ignore
import time
import ast
import pandas as pd


project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from paths import *
import src.optimization.helper_optimization as ho
import src.optimization.GA.GA as ga
import src.optimization.experiments.helper_experiment as he
import src.optimization.experiments.graph_metrics.synthetic_data.generate_synthetic_data as gsd

# Set random seeds
np.random.seed(1)
random.seed(1)

# Define global variables to share data between processes
df = None
G = None
distance_matrix = None
id_to_idx = None
idx_to_id = None
distance_matrix_binary = None
STATION_MIN_DISTANCE = None
initial_populations = None

# GA hyperparameters
SELECTION_STRATEGY = 'tournament'
CROSSOVER_STRATEGY = 'greedy'
POPULATION_SIZE = 50
GENERATIONS = 10_000  # there is early stopping
MUTATION_RATE = 0.05
ELITE_FRACTION = 0.05

    
def compute_shared_data():
    """Load data and set global variables"""
    global df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE
    df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE = he.load_data(
        use_elevation='parkin')

def generate_populations(n_stations_values):
    """Generate initial populations for each station count and store in global variable"""
    global initial_populations
    initial_populations = {}
    print("Generating initial populations for each station count...")
    for n_stations in n_stations_values:
        initial_populations[n_stations] = ho.generate_initial_population(
            n_stations, POPULATION_SIZE, distance_matrix, id_to_idx, STATION_MIN_DISTANCE, n_jobs=8
        )

def run_alpha_experiment(N_STATIONS, weights, alpha, experiment_idx, experiment_output_file, 
                     score_combination='exponential', penalty_power=3):
    """
    Run a single GA experiment with given parameters and alpha value.
    Using global variables instead of passing large data structures.
    
    Args:
        N_STATIONS: Number of stations
        weights: Dictionary of weights for different variables
        alpha: Alpha value for graph metrics (None for no graph metrics)
        experiment_idx: Index of the experiment
        experiment_output_file: Path to save results
        score_combination: Strategy for combining node and graph scores ('multiply', 'exponential', 'power_penalty')
        penalty_power: Power to use for penalizing bad graph scores (only used if score_combination='power_penalty')
    """
    # Check if the experiment has already been done
    df_results = he.load_existing_experiments(experiment_output_file)
    if not df_results.empty:
        # Convert weights to string format for comparison
        weights_str = str([f"{k}: {v}" for k, v in weights.items() if v != 0])
        
        # Check if this exact weight combination and alpha exists
        existing_experiment = df_results[
            (df_results['N_stations'] == N_STATIONS) & 
            (df_results['weights'] == weights_str) &
            (df_results['alpha'].isna() if alpha is None else df_results['alpha'] == alpha)
        ]
        
        if not existing_experiment.empty:
            print(f"  Skipping experiment {experiment_idx}: N={N_STATIONS}, Alpha={alpha}, Weights={weights}")
            return
        
    print(f"\nStarting experiment {experiment_idx}: N={N_STATIONS}, Alpha={alpha}, Weights={weights}")

    # Ensure shared data is loaded
    if df is None or initial_populations is None:
        raise RuntimeError("Shared data (df or initial_populations) not initialized. Call compute_shared_data() and generate_populations() first.")

    # Compute and normalize scores according to weights
    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), weights)
    logging_rate = 10

    # Execute genetic algorithm
    start_time = time.time()
    graph_score = True if alpha is not None else False
    
    current_population = initial_populations.get(N_STATIONS)
    if current_population is None:
        raise KeyError(f"Initial population for N_STATIONS={N_STATIONS} not found. Ensure generate_populations() was called for this value.")

    best_nodes, best_score, ga_tracking = ga.genetic_algorithm(
        df=df_weighted, 
        population=current_population,
        
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
        metric_bounds=None, # Computed in GA.py
        score_combination=score_combination,
        penalty_power=penalty_power,
        logging_rate=logging_rate
    )
    generation_scores, _, _, _, _, _ = ga_tracking
    execution_time = (time.time() - start_time) / 60  # minutes

    # Create result dictionary
    result = {
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
    
    # Save using helper function
    he.save_experiment(df_results, experiment_output_file, experiment_idx)

def run_all_alpha_experiments(n_stations_values, weight_combinations, alpha_values, score_combination, penalty_power, 
                              experiment_dir, experiments_output_file, processes):
    """
    Run GA experiments for different alpha values and weight combinations.
    
    Args:
        n_stations_values: List of station counts to test
        weight_combinations: List of weight combinations
        alpha_values: List of alpha values to test
        score_combination: Strategy for combining node and graph scores
        penalty_power: Power to use for penalizing bad graph scores
        experiment_dir: Directory for experiment results
        experiments_output_file: Path to save results
        processes: Number of processes to use
    """
    start_time = time.time()

    # Save parameters in a separate file
    config_info = {
        'N_stations': n_stations_values,
        'distance_constraint': STATION_MIN_DISTANCE,
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
    
    config_file = f"{experiment_dir}/config_alpha_screening_altitude.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            if key == 'alpha_values':
                f.write(f"{key}: {[float(x) for x in value]}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Check existing experiments
    df_results = he.load_existing_experiments(experiments_output_file)
    
    # Create a unique identifier for each experiment based on its parameters
    def get_experiment_id(n_stations, weights, alpha):
        weights_str = str([f"{k}: {v}" for k, v in weights.items() if v != 0])
        return f"{n_stations}_{weights_str}_{alpha}"
    
    # Create a mapping of experiment IDs to their indices
    completed_experiments = {}
    if not df_results.empty:
        for _, row in df_results.iterrows():
            exp_id = get_experiment_id(row['N_stations'], 
                                     {item.split(': ')[0]: float(item.split(': ')[1].rstrip("'")) 
                                      for item in ast.literal_eval(row['weights'])}, 
                                     row['alpha'])
            completed_experiments[exp_id] = row['experiment_idx']
    
    # Prepare experiment arguments
    experiment_args = []
    experiment_counter = max(completed_experiments.values()) + 1 if completed_experiments else 1
    
    for weights in weight_combinations:           
       # First run without graph metrics
        for n_stations in n_stations_values:
            exp_id = get_experiment_id(n_stations, weights, None)
            if exp_id not in completed_experiments:
                args = (
                    n_stations, weights, None, experiment_counter, 
                    experiments_output_file, score_combination, penalty_power
                )
                experiment_args.append(args)
                experiment_counter += 1
            
        # Then run with graph metrics for each alpha
        for alpha in alpha_values:
            for n_stations in n_stations_values:
                exp_id = get_experiment_id(n_stations, weights, alpha)
                if exp_id not in completed_experiments:
                    args = (
                        n_stations, weights, alpha, experiment_counter, 
                        experiments_output_file, score_combination, penalty_power
                    )
                    experiment_args.append(args)
                    experiment_counter += 1
    
    total_experiments = len(weight_combinations) * (len(alpha_values) + 1) * len(n_stations_values)
    remaining_experiments = len(experiment_args)

    if not experiment_args:
        print("All experiments have been completed!")
        return

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Completed experiments: {total_experiments - remaining_experiments}")
    print(f"Remaining experiments: {remaining_experiments}")

    # Run experiments in parallel
    print(f"\nRunning {remaining_experiments} experiments with {processes} processes")
    
    try:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(run_alpha_experiment, experiment_args)
    except Exception as e:
        print(f"\nError in parallel execution: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/3600:.2f} hours")
        print(f"Results saved to {experiments_output_file}")

if __name__ == "__main__":
    # Define score combination
    score_combination = 'multiply'  # Options: 'multiply', 'exponential', 'power_penalty'
    penalty_power = None  # Higher power for more severe penalties

    # Define alpha values to test
    alpha_values = [x for x in np.arange(0, 1.1, 1)]
    n_stations_values = [250]
    
    # Create results directory
    RESULTS_DIR = f"{PR_EXP}/scenarios/"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = f"{RESULTS_DIR}/alpha_screening_altitude.csv"

    # Initialize data and apply synthetic data strategies
    compute_shared_data()
    df, G = gsd.apply_strategies(df, G)
    
    # Load weight combinations from config file
    weight_file = f"{project_root}/src/optimization/experiments/scenarios/bss_weights.yaml"
    with open(weight_file, 'r') as f:
        weight_data = yaml.safe_load(f)
        weight_combinations = list(weight_data['weight_combinations'].values())

    weight_combinations = [weight_combinations[2]] + [weight_combinations[0]] + [weight_combinations[1]]
    print(weight_combinations)

    # Generate populations (no OpenMP operations)
    generate_populations(n_stations_values)

    # Run experiments
    processes = 3  # Use multiple processes
    run_all_alpha_experiments(
        n_stations_values, weight_combinations, alpha_values,
        score_combination, penalty_power,
        RESULTS_DIR, RESULTS_FILE, processes
    )