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


def run_ga_experiment(df, N_STATIONS, weights, experiment_idx, experiment_output_file, initial_populations):
    """
    Run a GA experiment with given parameters.

    Args:
        df: pandas DataFrame containing the node data
        N_STATIONS: number of stations to optimize
        weights: dictionary of weights for the nodes
        experiment_idx: index of the experiment
        experiment_output_file: file to save the results
        initial_populations: dictionary of initial populations for different station counts
    
    Saves results to experiment_output_file with:
    - experiment_idx: index of the experiment
    - N_stations: number of stations
    - ga_score: GA fitness score
    - ga_solution: GA solution found
    - ga_time: GA execution time
    - generations: number of generations the GA ran
    - weights: weight combination used
    
    Skips execution if identical parameter combination was already tested.
    """
    # Check if the experiment has already been done
    df_results = he.load_existing_experiments(experiment_output_file)
    if not df_results.empty:
        # Convert weights to string format for comparison
        weights_str = str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0]))
        
        # Check if this exact weight combination and station count exists
        existing_experiment = df_results[
            (df_results['N_stations'] == N_STATIONS) & 
            (df_results['weights'] == weights_str)
        ]
        
        if not existing_experiment.empty:
            print(f"  Skipping experiment {experiment_idx}: N={N_STATIONS}, Weights={weights}")
            return
        
    print(f"\nStarting experiment {experiment_idx}: N={N_STATIONS}, Weights={weights}")

    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), weights)
    
    # Get the initial population for this station count
    initial_population = initial_populations[N_STATIONS]
    
    # Execute genetic algorithm
    ga_start_time = time.time()
    logging_rate = 10
    ga_solution, ga_score, ga_tracking = ga.genetic_algorithm(
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
    ga_time = (time.time() - ga_start_time) / 60  # minutes
    
    # Create result and save
    result = {
        "N_stations": N_STATIONS,
        "best_score": ga_score,
        "minutes_to_complete": ga_time,
        "generations": len(generation_scores)*logging_rate,
        "weights": str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0])),
        "best_solution": ga_solution
    }
    
    df_results = pd.DataFrame([result])
    
    # Save using helper function
    he.save_experiment(df_results, experiment_output_file, experiment_idx)


def run_ga_experiments(df, station_counts, weight_combinations, experiment_dir, experiments_output_file, processes):
    """
    Run GA experiments for different station counts and weight combinations.
    
    Outputs (in {PR_EXP}/GA_MILP_stations/):
    1. experiment_config_GA.txt: Contains GA experiment parameters
    2. GA_results.csv: Main results file with GA data
    """
    start_time = time.time()

    # Save parameters in a separate file
    config_info = {
        'station_counts': station_counts,
        'distance_constraint': STATION_MIN_DISTANCE,
        'selection_strategy': selection_strategy,
        'crossover_strategy': crossover_strategy,
        'population_size': POPULATION_SIZE,
        'mutation_rate': MUTATION_RATE,
        'elite_fraction': ELITE_FRACTION,
        'max_generations': GENERATIONS,
        'early_stopping': True,
        'number_of_weight_combinations': len(weight_combinations)
    }
    config_file = f"{experiment_dir}/experiment_config_GA.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    
    # Generate initial populations for each station count
    print("Generating initial populations for each station count...")
    initial_populations = {}
    for n_stations in station_counts:
        print(f"  Generating population for {n_stations} stations")
        initial_populations[n_stations] = ho.generate_initial_population(
            n_stations, POPULATION_SIZE, distance_matrix, id_to_idx, STATION_MIN_DISTANCE, n_jobs=8
        )
    
    # Check existing experiments
    df_results = he.load_existing_experiments(experiments_output_file)
    
    # Prepare experiment arguments
    experiment_args = []
    experiment_counter = 1
    
    for n_stations in station_counts:
        for weights in weight_combinations:
            # Create a unique ID based on station count and weights
            weights_str = str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0]))
            
            # Check if experiment already exists
            if not df_results.empty:
                existing = df_results[
                    (df_results['N_stations'] == n_stations) & 
                    (df_results['weights'] == weights_str)
                ]
                if not existing.empty:
                    print(f"Skipping existing experiment: N={n_stations}, Weights={weights}")
                    experiment_counter += 1
                    continue
            
            args = (df, n_stations, weights, experiment_counter, experiments_output_file, initial_populations)
            experiment_args.append(args)
            experiment_counter += 1
    
    total_experiments = len(station_counts) * len(weight_combinations)
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
            pool.starmap(run_ga_experiment, experiment_args)
    except Exception as e:
        print(f"\nError in parallel execution: {e}")
        raise
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/3600:.2f} hours")
        print(f"Results saved to {experiments_output_file}")


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
    
    # Load configuration and data
    compute_shared_data()  # This sets up our global variables including df

    # Define station counts to test
    station_counts = [30, 60, 90, 120]
    
    # Generate weight combinations using Monte Carlo sampling
    # We'll use fewer weights per station count to keep total experiments manageable
    columns_to_drop = ['node_id', 'geometry']
    target_vars = df.columns.drop(columns_to_drop)
    weight_combinations = he.generate_weight_vectors(
        variables=target_vars,
        n_samples=24,  # 24 weights x 4 station counts = 96 experiments
        seed=42
    )

    # Output files
    experiment_dir = f"{PR_EXP}/GA_MILP_stations"
    os.makedirs(experiment_dir, exist_ok=True)
    experiments_output_file = f"{experiment_dir}/GA_results.csv"

    # Run optimization
    processes = 5
    run_ga_experiments(df, station_counts, weight_combinations, experiment_dir, experiments_output_file, processes)
