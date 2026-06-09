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
from src.optimization.experiments.GA_vs_MILP.MILP_algorithm import process_one_weight_combination
import src.optimization.experiments.helper_experiment as he

# Compute distance matrix once and share it globally
df = None
G = None
distance_matrix = None
distance_matrix_binary = None
id_to_idx = None
idx_to_id = None
STATION_MIN_DISTANCE = None

def compute_shared_data():
    """Load and compute shared data once for all experiments."""
    global df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE
    df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE = he.load_data(root='./')

def milp_experiment(df, N_STATIONS, weights, experiment_idx, experiment_output_file):
    """
    Run a single MILP experiment with given parameters.
    
    Saves results to experiment_output_file with:
    - experiment_idx: index of the experiment
    - N_stations: number of stations
    - milp_score: optimal value found
    - milp_solution: optimal solution found
    - milp_time: execution time in minutes
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
    
    # Execute MILP
    milp_score, milp_solution, _, milp_time = process_one_weight_combination(
            weights, df, N_STATIONS, distance_matrix, id_to_idx, idx_to_id, 
            STATION_MIN_DISTANCE, solver='gurobi'
        )

    # Create result and save
    result = {
        "N_stations": N_STATIONS,
        "best_score": milp_score,
        "minutes_to_complete": milp_time,
        "weights": str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0])),
        "best_solution": milp_solution,
    }
    
    df_results = pd.DataFrame([result])
    
    # Save using helper function
    he.save_experiment(df_results, experiment_output_file, experiment_idx)


def run_milp_experiments(df, station_counts, weight_combinations, experiment_dir, experiments_output_file, processes):
    """
    Run MILP experiments for different station counts and weight combinations.
    
    Outputs (in {PR_EXP}/GA_MILP_stations/):
    1. experiment_config_MILP.txt: Contains all MILP experiment parameters
    2. MILP_results.csv: Main results file with optimal solutions and scores
    """
    start_time = time.time()

    # Save parameters in a separate file
    config_info = {
        'station_counts': station_counts,
        'distance_constraint': STATION_MIN_DISTANCE,
        'solver': 'gurobi',
        'number_of_weight_combinations': len(weight_combinations)
    }
    config_file = f"{experiment_dir}/experiment_config_MILP.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    
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
            
            args = (df, n_stations, weights, experiment_counter, experiments_output_file)
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
            pool.starmap(milp_experiment, experiment_args)
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
    
    # Load configuration and data
    compute_shared_data()  # This sets up our global variables including df

    # Define station counts to test - MUST MATCH the ones in GA_execution.py
    station_counts = [30, 60, 90, 120]
    
    # Generate weight combinations using Monte Carlo sampling - MUST MATCH the ones in GA_execution.py
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
    experiments_output_file = f"{experiment_dir}/MILP_results.csv"

    # Run optimization
    processes = 8
    run_milp_experiments(df, station_counts, weight_combinations, experiment_dir, experiments_output_file, processes)
