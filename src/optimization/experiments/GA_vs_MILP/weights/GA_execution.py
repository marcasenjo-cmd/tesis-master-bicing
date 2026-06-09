import itertools
import multiprocessing as mp
import os
import sys
import time
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.GA.GA as ga
import src.optimization.helper_optimization as ho
import src.optimization.experiments.helper_experiment as he

# Compute distance matrix once and share it globally
df = None
G = None
distance_matrix = None
id_to_idx = None
idx_to_id = None
distance_matrix_binary = None
STATION_MIN_DISTANCE = None

def compute_shared_data():
    """Load and compute shared data once for all experiments."""
    global df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE
    df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE = he.load_data(root='./')


def ga_experiment(df, N_STATIONS, weights, experiment_idx, experiment_output_file, initial_population):
    """
    Run a single GA experiment with given parameters.
    
    Saves results to experiment_output_file with:
    - N_stations: number of stations
    - best_score: best fitness score found
    - best_solution: best solution found
    - generations: number of generations run
    - minutes_to_complete: execution time
    - weights: weight combination used
    
    Skips execution if identical parameter combination was already tested.
    """
    # Check if the experiment has already been done
    df_results = he.load_existing_experiments(experiment_output_file)
    if not df_results.empty:
        # Convert weights to string format for comparison
        weights_str = str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0]))
        
        # Check if this exact weight combination exists
        existing_experiment = df_results[
            (df_results['experiment_idx'] == experiment_idx) & 
            (df_results['weights'] == weights_str)
        ]
        
        if not existing_experiment.empty:
            print(f"  Skipping experiment {experiment_idx}: Weights={weights}")
            return
        
    print(f"\nStarting experiment {experiment_idx}: Weights={weights}")

    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), weights)
    
    # Execute genetic algorithm
    start_time = time.time()
    logging_rate = 10
    best_solution, best_score, ga_tracking = ga.genetic_algorithm(
        df=df_weighted,
        population=initial_population,
        
        distance_matrix=distance_matrix,
        station_min_distance=STATION_MIN_DISTANCE,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        
        generations=GENERATIONS,
        elite_proportion=ELITE_FRACTION,
        mutation_rate=MUTATION_RATE,
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        
        G=G,
        graph_score=False, 
        alpha=None,
        metric_bounds=None,
        logging_rate=logging_rate
    )
    _, generation_scores, _, _, _, _ = ga_tracking
    

    # Create df and save
    execution_time = time.time() - start_time
    result = {
        "N_stations": N_STATIONS,
        "best_score": best_score,
        "minutes_to_complete": execution_time/60,
        "generations": len(generation_scores)*logging_rate,
        "weights": str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0])),
        "best_solution": str(best_solution),
    }
    
    df_results = pd.DataFrame([result])
    
    # Save using helper function
    success = he.save_experiment(df_results, experiment_output_file, experiment_idx)
    if not success:
        # If save fails, write to a backup file
        backup_file = experiment_output_file.replace('.csv', f'_backup_{experiment_idx}.csv')
        df_results.to_csv(backup_file, index=False)


def run_all_ga_experiments(df, N_STATIONS, weight_combinations, experiment_dir, experiments_output_file, processes):
    """
    Run GA experiments for different weight combinations.
    
    Outputs (in {PR_EXP}/GA_MILP_weights/):
    1. experiment_config.txt: Contains all experiment parameters
    2. GA_scores_in_weights_experiment.csv: Main results file with:
       - experiment configuration
       - best solutions and scores
       - execution metrics (generations, time)
    """
    start_time = time.time()

    # Save weights and parameters in a separate file
    config_info = {
        'N_stations': N_STATIONS,
        'distance_constraint': STATION_MIN_DISTANCE,
        'selection_strategy': selection_strategy,
        'crossover_strategy': crossover_strategy,
        'population_size': POPULATION_SIZE,
        'mutation_rate': MUTATION_RATE,
        'elite_fraction': ELITE_FRACTION,
        'max_generations': GENERATIONS,
        'early_stopping': True
    }
    config_file = f"{experiment_dir}/experiment_config_GA.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            if 'weights' in key: continue
            f.write(f"{key}: {value}\n")
    
    # Generate initial population
    initial_population = ho.generate_initial_population(
        N_STATIONS, POPULATION_SIZE, distance_matrix, id_to_idx, STATION_MIN_DISTANCE, n_jobs=8
    )
    
    # Check existing experiments
    df_results = he.load_existing_experiments(experiments_output_file)
    completed_experiments = set(df_results['experiment_idx']) if not df_results.empty else set()  
    
    # Prepare experiment arguments
    experiment_args = []
    for i, weights in enumerate(weight_combinations, 1):
        if i not in completed_experiments:
            args = (df, N_STATIONS, weights, i, experiments_output_file, initial_population)
            experiment_args.append(args)
    
    total_experiments = len(weight_combinations)
    remaining_experiments = len(experiment_args)

    if not experiment_args:
        print("All experiments have been completed!")
        return

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Completed experiments: ({len(completed_experiments)}) {completed_experiments}")
    print(f"Remaining experiments: ({remaining_experiments}) {[args[3] for args in experiment_args]}")

    # Run experiments in parallel
    print(f"\nRunning {remaining_experiments} experiments with {processes} processes")
    try:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(ga_experiment, experiment_args)
    except Exception as e:
        print(f"\nError in parallel execution: {e}")
        raise
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/3600:.2f} hours")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # GA hyperparameters
    selection_strategy = 'roulette'
    crossover_strategy = 'greedy'
    POPULATION_SIZE = 50
    GENERATIONS = 10_000  # there is early stopping
    MUTATION_RATE = 0.05
    ELITE_FRACTION = 0.05
    N_STATIONS = 60

    # Load configuration and data
    compute_shared_data()  # This sets up our global variables including df

    # Generate weight combinations using Monte Carlo sampling
    columns_to_drop = ['node_id', 'geometry']
    target_vars = df.columns.drop(columns_to_drop)
    weight_combinations = he.generate_weight_vectors(
        variables=target_vars,
        n_samples=118,
        seed=42
    )

    # Output files
    experiment_dir = f"{PR_EXP}/GA_MILP_weights"
    os.makedirs(experiment_dir, exist_ok=True)
    experiments_output_file = f"{experiment_dir}/GA_scores_in_weights_experiment.csv"

    # Run optimization
    processes = 8
    run_all_ga_experiments(df, N_STATIONS, weight_combinations, experiment_dir, experiments_output_file, processes)
