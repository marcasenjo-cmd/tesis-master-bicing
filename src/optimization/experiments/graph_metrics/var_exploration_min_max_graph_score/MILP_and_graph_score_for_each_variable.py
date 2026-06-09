import multiprocessing as mp
import os
import sys
import time
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning

# Suppress Numba warnings
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
from src.optimization.experiments.GA_vs_MILP.MILP_algorithm import process_one_weight_combination
import src.optimization.experiments.helper_experiment as he
import src.optimization.GA.graph_metric.graph_normalization as gn

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
    - N_stations: number of stations
    - best_score: optimal value found
    - best_solution: optimal solution found
    - minutes_to_complete: execution time
    - weights: weight combination used
    - dispersion_score: normalized dispersion score
    - accessibility_score: normalized accessibility score
    
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
    
    # Execute MILP
    optimal_value, selected_nodes, _, execution_time = process_one_weight_combination(
        weights, df, N_STATIONS, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE, solver='gurobi'
    )

    try:
        # Compute graph scores
        alpha = 0
        min_disp_bound, _ = gn.min_dispersion_bound(N_STATIONS, STATION_MIN_DISTANCE)
        max_disp_metric, _ = gn.max_dispersion_metric(N_STATIONS, distance_matrix, STATION_MIN_DISTANCE, idx_to_id)
        min_accessibility_bound, _ = gn.min_accessibility_bound_kmeans(G, N_STATIONS, distance_matrix, id_to_idx)
        max_accessibility_bound, _ = gn.max_accessibility_bound(N_STATIONS, distance_matrix, idx_to_id, STATION_MIN_DISTANCE)
        
        # Convert solution to indices
        solution_indices = [id_to_idx[node_id] for node_id in selected_nodes]
        
        # Compute normalized graph scores
        dispersion, accessibility, _ = gn.evaluate_normalize_and_invert_stations_set(
            solution_indices,
            distance_matrix,
            alpha,
            (min_disp_bound, max_disp_metric),
            (min_accessibility_bound, max_accessibility_bound)
        )
    except Exception as e:
        print(f"Error in experiment {experiment_idx}: {e}")
        return

    # Create df and save
    result = {
        "N_stations": N_STATIONS,
        "best_score": optimal_value,
        "minutes_to_complete": execution_time,
        "weights": str(sorted([f"{k}: {v}" for k, v in weights.items() if v != 0])),
        "best_solution": str(selected_nodes),
        "dispersion_score": dispersion,
        "accessibility_score": accessibility
    }
    
    df_results = pd.DataFrame([result])
    
    # Save using helper function
    success = he.save_experiment(df_results, experiment_output_file, experiment_idx)
    if not success:
        # If save fails, write to a backup file
        backup_file = experiment_output_file.replace('.csv', f'_backup_{experiment_idx}.csv')
        df_results.to_csv(backup_file, index=False)

def run_all_milp_experiments(df, N_STATIONS, weight_combinations, experiment_dir, 
                             experiments_output_file, num_processes=6):
    """
    Run MILP experiments for different weight combinations.
    
    Outputs (in {PR_EXP}/GA_MILP_weights/):
    1. experiment_config.txt: Contains all experiment parameters
    2. MILP_scores_in_weights_experiment.csv: Main results file with:
       - experiment configuration
       - optimal solutions and scores
       - execution metrics (time)
    """
    start_time = time.time()

    # Save parameters in a separate file
    config_info = {
        'N_stations': N_STATIONS,
        'distance_constraint': STATION_MIN_DISTANCE,
        'solver': 'gurobi'
    }
    config_file = f"{experiment_dir}/experiment_config_MILP.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            if 'weights' in key: continue
            f.write(f"{key}: {value}\n")
    
    # Check existing experiments
    df_results = he.load_existing_experiments(experiments_output_file)
    completed_experiments = set(df_results['experiment_idx']) if not df_results.empty else set()
    
    print(f"Number of weight combinations: {len(weight_combinations)}")
    print(f"Completed experiments: {completed_experiments}")
    
    # Prepare experiment arguments
    experiment_args = []
    for i, weights in enumerate(weight_combinations, 1):
        if i not in completed_experiments:
            args = (df, N_STATIONS, weights, i, experiments_output_file)
            experiment_args.append(args)
    
    total_experiments = len(weight_combinations)
    remaining_experiments = len(experiment_args)

    if not experiment_args:
        print("All experiments have been completed!")
        return

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Completed experiments: {total_experiments - remaining_experiments}")
    print(f"Remaining experiments: {remaining_experiments}")
    print(f"Experiment indices to run: {[args[3] for args in experiment_args]}")

    # Run experiments in parallel
    print(f"\nRunning {remaining_experiments} experiments with {num_processes} processes")
    try:
        with mp.Pool(processes=num_processes) as pool:
            # Use starmap_async to allow for immediate result processing
            pool.starmap_async(milp_experiment, experiment_args)
            pool.close()
            pool.join()  # Wait for all processes to complete
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
    
    # Parameters
    N_STATIONS = 30

    # Load configuration and data
    compute_shared_data()  # This sets up our global variables including df

    # Get all variables from the config
    columns_to_drop = ['node_id', 'geometry']
    target_vars = df.columns.drop(columns_to_drop)

    # Create weight combinations - one for each variable
    weight_combinations = []
    for var in target_vars:
        weights = {var: 1}
        weight_combinations.append(weights)

    # Output files
    experiment_dir = f"{PR_EXP}/MILP_for_each_var"
    os.makedirs(experiment_dir, exist_ok=True)
    experiments_output_file = f"{experiment_dir}/MILP_scores_for_each_var.csv"

    # Run optimization
    run_all_milp_experiments(df, N_STATIONS, weight_combinations, experiment_dir, 
                             experiments_output_file, num_processes=6)
