import itertools
import multiprocessing as mp
import os
import sys
import time
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.GA.GA as ga
from src.optimization.experiments.helper_experiment import (
    load_data, load_existing_experiments, save_experiment
)
import src.optimization.helper_optimization as ho

# Compute distance matrix once and share it globally
df = None
G = None
distance_matrix = None
id_to_idx = None
idx_to_id = None
distance_matrix_binary = None
STATION_MIN_DISTANCE = None

def filter_invalid_combinations(param_combinations):
    """
    Filter out invalid parameter combinations based on various constraints.
    
    Current constraints checked:
    1. Elite count >= 2: elite_count = int((elite_frac * pop_size) / 2) >= 2
    2. Roulette selection population size >= 2 (to select 2 parents without replacement)
    3. Tournament selection tournament size <= population size (handled automatically in GA code)
    
    Args:
        param_combinations (list): List of parameter tuples
        
    Returns:
        tuple: (valid_combinations, invalid_combinations)
    """
    valid_combinations = []
    invalid_combinations = []
    
    for combo in param_combinations:
        max_gen, pop_size, mut_rate, elite_frac, sel_str, cross_str = combo
        elite_count = int((elite_frac * pop_size) / 2)
        
        # Check constraint 1: Elite count must be >= 2 (required for elite-based crossover)
        if elite_count < 2:
            invalid_combinations.append(combo)
            continue
            
        # Check constraint 1b: Population must be large enough to have meaningful elites and non-elites
        non_elite_count = pop_size - elite_count
        if non_elite_count < 1:
            invalid_combinations.append(combo)
            continue
            
        # Check constraint 2: Roulette selection needs population size >= 2
        # (to select 2 parents without replacement)
        if sel_str == 'roulette' and pop_size < 2:
            invalid_combinations.append(combo)
            continue
            
        # Check constraint 3: Mutation rate should be reasonable (0 <= rate <= 1)
        if mut_rate < 0 or mut_rate > 1:
            invalid_combinations.append(combo)
            continue
            
        # Check constraint 4: Elite fraction should be reasonable (0 < frac < 1)
        if elite_frac <= 0 or elite_frac >= 1:
            invalid_combinations.append(combo)
            continue
        
        # If all constraints pass, it's valid
        valid_combinations.append(combo)
    
    return valid_combinations, invalid_combinations


def compute_shared_data():
    global df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE
    df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE = load_data(root='./')


def track_exploration_metrics(generations_population, total_graph_nodes, experiment_index, output_file):
    """
    Track and save exploration metrics for each generation.
    
    Saves to CSV with columns:
    - experiment_idx: experiment identifier
    - gen: how many generations have been run
    - unique_nodes_count: number of unique nodes in current generation
    - fraction_new_nodes: fraction of new nodes vs previous generation
    - generation_coverage: fraction of total graph nodes covered
    - cumulative_exploration_fraction: cumulative fraction of explored nodes
    """
    exploration_metrics = []
    prev_gen_nodes = set()
    cumulative_explored_nodes = set()
    
    for gen, population in enumerate(generations_population):
        current_generation_nodes = set(np.concatenate(population))
        
        # Count the number of unique nodes in the current generation
        unique_nodes_count = len(current_generation_nodes)
        
        # Calculate the number of new nodes introduced in this generation
        new_nodes_count = len(current_generation_nodes - prev_gen_nodes)
        
        # Calculate the fraction of new nodes relative to the unique nodes
        fraction_new_nodes = new_nodes_count / unique_nodes_count if unique_nodes_count else 0
        
        # Calculate the coverage of the current generation in relation to total graph nodes
        generation_coverage = unique_nodes_count / total_graph_nodes
        
        # Update cumulative explored nodes with the current generation's nodes
        cumulative_explored_nodes.update(current_generation_nodes)
        
        # Store the calculated metrics in a dictionary
        metrics = {
            'experiment_index': experiment_index,
            'gen': gen,
            'unique_nodes_count': unique_nodes_count,
            'fraction_new_nodes': fraction_new_nodes,
            'generation_coverage': generation_coverage,
            'cumulative_exploration_fraction': len(cumulative_explored_nodes) / total_graph_nodes
        }
        
        # Append the metrics for the current generation to the exploration metrics list
        exploration_metrics.append(metrics)

        # Update previous generation nodes for the next iteration
        prev_gen_nodes = current_generation_nodes
    
    # Convert the exploration metrics list to a DataFrame
    df_metrics = pd.DataFrame(exploration_metrics)
    df_metrics = df_metrics.round(3)
    
    # Save the metrics DataFrame to a CSV file, appending if the file already exists
    df_metrics.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))

    
def ga_experiment(df_weighted, N_STATIONS, params, experiment_index, experiments_output_file, 
                  exploration_tracking_file, selection_strategy, crossover_strategy):
    """
    Run a single GA experiment with given parameters.
    
    Saves results to:
    1. experiments_output_file: experiment configuration and results
    2. exploration_tracking_file: generation-by-generation exploration metrics
    
    Skips execution if identical parameter combination was already tested.
    """
    max_generations, pop_size, mutation_rate, elite_fraction, selection_strategy, crossover_strategy = params
    
    # Convert list strategies to strings for comparison
    sel_str = selection_strategy[0] if isinstance(selection_strategy, list) else selection_strategy
    cross_str = crossover_strategy[0] if isinstance(crossover_strategy, list) else crossover_strategy
    
    # Check if the experiment has already been done
    df_results = load_existing_experiments(experiments_output_file)
    if not df_results.empty:
        # Convert to numeric to handle string values like '116.0'
        completed_experiments = set(pd.to_numeric(df_results['experiment_index'], errors='coerce').dropna())
    else:
        completed_experiments = set()
    
    if experiment_index in completed_experiments:
        print(f"  Skipping experiment {experiment_index}: Pop={pop_size}, Mut={mutation_rate}, "
              f"Elit={elite_fraction}, Sel:{sel_str}, Cro:{cross_str}")
        return
        
    print(f"\nStarting experiment {experiment_index}: Pop={pop_size}, Mut={mutation_rate}, "
          f"Elit={elite_fraction}, Sel:{sel_str}, Cro:{cross_str}")
    
    # Generate initial populations
    initial_population = ho.generate_initial_population(
        N_STATIONS, pop_size, distance_matrix, id_to_idx, STATION_MIN_DISTANCE, n_jobs=1
    )
    
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
        
        generations=max_generations,
        elite_proportion=elite_fraction,
        mutation_rate=mutation_rate,
        selection_strategy=sel_str,
        crossover_strategy=cross_str,
        
        G=G,
        graph_score=False, 
        alpha=None, # type: ignore
        metric_bounds=None,
        logging_rate=logging_rate
    )
    avg_scores, generation_scores, generation_times, jaccard_div, mutation_rates, generations_population = ga_tracking
    
    # Save evolution of the GA metrics
    track_exploration_metrics(generations_population, len(df_weighted.index), 
                            experiment_index, exploration_tracking_file)

    # Create df and save
    execution_time = time.time() - start_time
    result = {
        "experiment_index": experiment_index,
        "population_size": pop_size,
        "mutation_rate": mutation_rate,
        "elite_fraction": elite_fraction,
        "selection_strategy": sel_str,
        "crossover_strategy": cross_str,
        "best_score": best_score,
        "best_solution": str(best_solution),
        "generations": len(generation_scores)*logging_rate,
        "minutes_to_complete": execution_time/60,
        "generation_scores": str(generation_scores),
        "avg_scores": str(avg_scores),
        "jaccard_diversities": str(jaccard_div)
    }
    
    print(f"\nSaving results for experiment {experiment_index}...")
    df_results = pd.DataFrame([result])
    
    # Use the helper function to save experiments properly
    try:
        save_experiment(df_results, experiments_output_file, experiment_index)
    except Exception as e:
        print(f"Error saving experiment {experiment_index}: {e}")
        # If save fails, write to a backup file
        backup_file = experiments_output_file.replace('.csv', f'_backup_{experiment_index}.csv')
        df_results.to_csv(backup_file, index=False)
        print(f"Warning: Saved experiment {experiment_index} to backup file: {backup_file}", flush=True)


def run_all_ga_experiments(df_weighted, N_STATIONS, WEIGHTS, selection_strategy, crossover_strategy):
    """
    Run grid search for GA hyperparameter optimization.
    
    Outputs (in {PR_EXP}/GA_hyperparameter_optimization_{N_STATIONS}_stations/):
    1. experiment_config.txt: Contains all experiment parameters
    2. GA_scores_and_nodes.csv: Main results file with:
       - experiment configuration
       - best solutions and scores
       - evolution metrics (scores, diversity)
    3. GA_exploration_tracking.csv: Exploration metrics per generation:
       - unique nodes count
       - new nodes fraction
       - coverage metrics
    """
    start_time = time.time()
    
    # Output files
    experiment_dir = f"{PR_EXP}/GA_hyperparameter_optimization_{N_STATIONS}_stations"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save weights and parameters in a separate file
    config_info = {
        'N_stations': N_STATIONS,
        'distance_constraint': STATION_MIN_DISTANCE,
        'weights': WEIGHTS,
        'selection_strategy': selection_strategy,
        'crossover_strategy': crossover_strategy,
        'population_sizes': POPULATION_SIZES,
        'mutation_rates': MUTATION_RATES,
        'elite_fractions': ELITE_FRACTIONS,
        'max_generations': MAX_GENERATIONS[0],
        'early_stopping': True
    }
    config_file = f"{experiment_dir}/experiment_config.txt"
    with open(config_file, 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    experiments_output_file = f"{experiment_dir}/GA_scores_and_nodes.csv"
    exploration_tracking_file = f"{experiment_dir}/GA_exploration_tracking.csv"
    
    # Parallel execution 
    all_param_combinations = sorted(list(itertools.product(
        MAX_GENERATIONS, POPULATION_SIZES, MUTATION_RATES, ELITE_FRACTIONS, 
        selection_strategy, crossover_strategy)))

    # Filter out invalid parameter combinations
    valid_combinations, invalid_combinations = filter_invalid_combinations(all_param_combinations)
    
    # Print info about invalid combinations but don't save them to results file
    if invalid_combinations:
        print(f"\nDiscarded {len(invalid_combinations)} invalid parameter combinations:")
        
        # Group by constraint type for better reporting
        elite_count_issues = []
        other_issues = []
        
        for combo in invalid_combinations:
            max_gen, pop_size, mut_rate, elite_frac, sel_str, cross_str = combo
            elite_count = int((elite_frac * pop_size) / 2)
            
            # Determine the reason for invalidity
            non_elite_count = pop_size - elite_count
            if elite_count < 2:
                elite_count_issues.append((combo, f"elite_count={elite_count} < 2"))
            elif non_elite_count < 1:
                elite_count_issues.append((combo, f"non_elite_count={non_elite_count} < 1 (pop={pop_size}, elite={elite_count})"))
            elif sel_str == 'roulette' and pop_size < 2:
                other_issues.append((combo, f"roulette selection requires pop_size >= 2"))
            elif mut_rate < 0 or mut_rate > 1:
                other_issues.append((combo, f"invalid mutation_rate={mut_rate}"))
            elif elite_frac <= 0 or elite_frac >= 1:
                other_issues.append((combo, f"invalid elite_fraction={elite_frac}"))
            else:
                other_issues.append((combo, "unknown constraint violation"))
        
        if elite_count_issues:
            print(f"  Elite count issues ({len(elite_count_issues)} combinations):")
            for i, (combo, reason) in enumerate(elite_count_issues[:10], 1):  # Show first 10
                max_gen, pop_size, mut_rate, elite_frac, sel_str, cross_str = combo
                print(f"    {i}. Pop={pop_size}, Elite_frac={elite_frac} -> {reason}")
            if len(elite_count_issues) > 10:
                print(f"    ... and {len(elite_count_issues) - 10} more")
        
        if other_issues:
            print(f"  Other issues ({len(other_issues)} combinations):")
            for i, (combo, reason) in enumerate(other_issues, 1):
                max_gen, pop_size, mut_rate, elite_frac, sel_str, cross_str = combo
                print(f"    {i}. Pop={pop_size}, Mut={mut_rate}, Elite={elite_frac}, Sel={sel_str} -> {reason}")

    # Use valid combinations only
    param_combinations = valid_combinations

    # Check existing experiments
    df_results = load_existing_experiments(experiments_output_file)
    
    # Create a set of completed hyperparameter combinations
    completed_combinations = set()
    if not df_results.empty:
        for _, row in df_results.iterrows():
            # Extract hyperparameters from the results row
            pop_size = row['population_size']
            mut_rate = row['mutation_rate']
            elite_frac = row['elite_fraction']
            sel_str = row['selection_strategy']
            cross_str = row['crossover_strategy']
            
            # Create a tuple of hyperparameters (excluding max_gen since it's not in results)
            # Note: We'll need to handle max_gen separately since it's not stored in results
            hyperparams = (pop_size, mut_rate, elite_frac, sel_str, cross_str)
            completed_combinations.add(hyperparams)
    
    # Find which valid combinations haven't been completed yet
    new_combinations = []
    for combo in param_combinations:
        max_gen, pop_size, mut_rate, elite_frac, sel_str, cross_str = combo
        
        # Create hyperparameter tuple (excluding max_gen)
        hyperparams = (pop_size, mut_rate, elite_frac, sel_str, cross_str)
        
        if hyperparams not in completed_combinations:
            new_combinations.append(combo)
    
    total_valid_experiments = len(param_combinations)
    total_all_experiments = len(all_param_combinations)
    remaining_experiments = len(new_combinations)

    if not new_combinations:
        print("All valid experiments have been completed!")
        return

    print(f"\nTotal experiments (all combinations): {total_all_experiments}")
    print(f"Total valid experiments: {total_valid_experiments}")
    print(f"Completed experiments: {total_valid_experiments - remaining_experiments}")
    print(f"Remaining experiments: {remaining_experiments}")

    # Create args list with new experiment indices (starting from the next available index)
    args_list = []
    if not df_results.empty:
        # Find the next available experiment index
        next_experiment_index = int(df_results['experiment_index'].max()) + 1
    else:
        next_experiment_index = 1
    
    for i, combo in enumerate(new_combinations):
        experiment_index = next_experiment_index + i
        args_list.append((df_weighted, N_STATIONS, combo, experiment_index, 
                         experiments_output_file, exploration_tracking_file, selection_strategy, crossover_strategy))
    
    # Run experiments in parallel
    processes = 1
    print(f"\nRunning {remaining_experiments} experiments with {processes} processes")
    try:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(ga_experiment, args_list)
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
    
    # Parameters to try
    selection_strategy = ['tournament']
    crossover_strategy = ['greedy', 'weighted_random', 'top_first']
    POPULATION_SIZES = [25, 50, 100, 200]
    MAX_GENERATIONS = [10_000]  # there is early stopping
    MUTATION_RATES = [0.01, 0.02, 0.05, 0.1, 0.2]
    ELITE_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2]
    N_STATIONS = 60
    STATION_MIN_DISTANCE = 300

    # Load configuration and data
    WEIGHTS = {'population': 0.33, 'education_primary': 0.33, 'unemployment_percentage': 0.33}
    compute_shared_data()  # This sets up our global variables including df
    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), WEIGHTS) # type: ignore

    # Run optimization
    run_all_ga_experiments(df_weighted, N_STATIONS, WEIGHTS, selection_strategy, crossover_strategy)