import sys
import os
import multiprocessing as mp
from functools import partial
import time
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely import wkt  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable  # type: ignore
import contextily as ctx  # type: ignore

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.helper_optimization as oh
import src.optimization.GA.GA as ga
from src.optimization.baseline.algorithm_G import greedy_algorithm
from src.optimization.hyperparameter_optimization.helper_GA_tracker import track_exploration_metrics

def process_weight_combination(
    weight, df, N_STATIONS, initial_population, distance_matrix, id_to_idx, idx_to_id,
    POPULATION_SIZE, GENERATIONS, MUTATION_RATE, ELITE_PROPORTION, 
    CROSSOVER_STRATEGY, SELECTION_STRATEGY, STATION_MIN_DISTANCE
):
    """
    Process a single weight combination.
    
    Args:
        weight (dict): Weight dictionary for this combination.
        df (GeoDataFrame): DataFrame with node attributes.
        N_STATIONS (int): Number of stations.
        initial_population (list): Initial population for GA.
        distance_matrix (ndarray): Distance matrix.
        id_to_idx (dict): Maps node IDs to matrix indices.
        idx_to_id (dict): Maps matrix indices to node IDs.
        POPULATION_SIZE (int): Size of population.
        GENERATIONS (int): Number of generations.
        MUTATION_RATE (float): Mutation rate.
        ELITE_PROPORTION (float): Elite proportion.
        CROSSOVER_STRATEGY (str): Crossover strategy.
        SELECTION_STRATEGY (str): Selection strategy.
        STATION_MIN_DISTANCE (float): Minimum distance between stations. If 0, no distance constraint is applied.
        
    Returns:
        tuple: (best_nodes_greedy, best_scores_greedy, best_nodes_ga, best_score_ga, ga_experiment_str)
    """
    # Compute and normalize scores according to weights
    df_weighted = oh.sum_and_normalize_all_node_scores(df.copy(), weight)
    
    # Run greedy algorithm WITHOUT distance constraint
    best_nodes_greedy, best_score_greedy = greedy_algorithm(
        df=df_weighted,
        column_name='norm_score',
        N_STATIONS=N_STATIONS,
        min_distance_matrix=distance_matrix,
        distance_constraint=(STATION_MIN_DISTANCE > 0),
        id_to_idx=id_to_idx,
        weights=weight
    )
    print(f"Greedy ({STATION_MIN_DISTANCE}m dist.) -> Score: {round(best_score_greedy, 3)} -> {str(weight)}")
    
    # Run genetic algorithm with distance constraint determined by STATION_MIN_DISTANCE
    start_time = time.time()
    best_nodes_ga, best_score_ga, ga_tracking = ga.genetic_algorithm(
        df=df_weighted,
        distance_matrix=distance_matrix,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        N=N_STATIONS,
        population=initial_population.copy(),
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elite_proportion=ELITE_PROPORTION,
        selection_strategy=SELECTION_STRATEGY,
        crossover_strategy=CROSSOVER_STRATEGY,
        distance_constraint=(STATION_MIN_DISTANCE > 0),
        logging_rate=1
    )
    average_scores, generation_scores, generation_times, jaccard_diversities, mutation_rates, generations_population = ga_tracking
    time_taken = round((time.time() - start_time) / 3600, 2) # hours

    print(f"Weight set: {list(weight.items())[:3]}..."\
          f"- Best score (GA with {STATION_MIN_DISTANCE}m dist. constraint) "\
          f"{round(best_score_ga, 3)}, time taken: {time_taken:.2f}h")

    # Save evolution of the GA metrics
    baseline_dir = f'{PR_METAHEURISTICS}/GA/baseline'
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    ga_output_file = f'{baseline_dir}/{N_STATIONS}_stations_{STATION_MIN_DISTANCE}_distance_constraint.csv'
    exploration_tracking_file = f'{baseline_dir}/{N_STATIONS}_stations_{STATION_MIN_DISTANCE}_distance_constraint_exploration.csv'
    
    # Save exploration metrics
    track_exploration_metrics(generations_population, len(df), exploration_tracking_file, weights=weight)
    
    # Create experiment string
    ga_experiment_str = f"pop:{POPULATION_SIZE}, gen:{len(generation_scores)}, "\
        f"mut:{MUTATION_RATE}, eli:{ELITE_PROPORTION}, time:{time_taken:.2f}h"

    # Create df and save
    result = {
        "weights": weight,
        "population_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        "elite_fraction": ELITE_PROPORTION,
        "best_score": best_score_ga,
        "best_solution": best_nodes_ga,
        "best_gen_score": generation_scores,
        "avg_gen_score": average_scores,
        "gen_times": generation_times,
        "jaccard_diversities": jaccard_diversities,
        "mutation_rates": mutation_rates,
    }

    df_results = pd.DataFrame([result])
    df_results.to_csv(ga_output_file, mode='a', header=not os.path.exists(ga_output_file), index=False)
    print(f"{N_STATIONS} stations, {STATION_MIN_DISTANCE}m dist. constraint GA results saved to CSV.")

    return best_nodes_greedy, best_score_greedy, best_nodes_ga, best_score_ga, ga_experiment_str


def plot_all_weight_comparison(
    gdf, 
    N_STATIONS, 
    weights_collection, 
    best_nodes_greedy_collection, 
    best_scores_greedy_collection, 
    best_nodes_ga_collection, 
    best_scores_ga_collection,
    ga_experiment_strs=None,
    STATION_MIN_DISTANCE=0
):
    """
    Plot a comparison of node scores, greedy algorithm, and genetic algorithm results
    for multiple weight combinations.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with node scores.
        N_STATIONS (int): Number of stations.
        weights_collection (list): List of weight dictionaries.
        best_nodes_greedy_collection (list): List of node ID lists selected by greedy algorithm for each weight.
        best_scores_greedy_collection (list): List of score lists for nodes selected by greedy algorithm for each weight.
        best_nodes_ga_collection (list): List of node ID lists selected by genetic algorithm for each weight.
        best_scores_ga_collection (list): List of score lists for nodes selected by genetic algorithm for each weight.
        ga_experiment_strs (list, optional): List of strings describing GA experiment parameters for each weight.
        STATION_MIN_DISTANCE (float): Minimum distance between stations. If 0, no distance constraint is applied.
    """
    # Number of weight combinations to display
    n_weights = len(weights_collection)
    
    # Create a figure with 3 rows (all nodes, greedy, GA) and n_weights columns
    fig = plt.figure(figsize=(6 * n_weights, 15))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, n_weights)
    
    # Load Barcelona boundary
    bcn_boundary = dl.load_bcn_boundary()
    
    # Settings
    markersize = 0.5
    fontsize_title = 9
    nodes_scores_fontsize = 7
    cmap = plt.cm.viridis_r
    
    # For each weight combination
    for w_idx, (weights, best_nodes_greedy, best_scores_greedy, best_nodes_ga, best_scores_ga) in enumerate(
        zip(weights_collection, best_nodes_greedy_collection, best_scores_greedy_collection, 
            best_nodes_ga_collection, best_scores_ga_collection)
    ):
        # Compute scores for this weight combination
        gdf_weighted = oh.sum_and_normalize_all_node_scores(gdf.copy().reset_index(), weights)
        
        # Create axes for this column
        ax_all_nodes = fig.add_subplot(gs[0, w_idx])
        ax_greedy = fig.add_subplot(gs[1, w_idx])
        ax_ga = fig.add_subplot(gs[2, w_idx])
        
        # Row 1: All nodes score map
        scatter = gdf_weighted.plot(
            column='norm_score',
            cmap=cmap,
            ax=ax_all_nodes,
            markersize=markersize,
            legend=False,
            vmin=0, vmax=1)
        
        # Add colorbar inside the plot
        divider = make_axes_locatable(ax_all_nodes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(scatter.collections[0], cax=cax)
        cbar.ax.tick_params(labelsize=8)
        original_min, original_max = gdf_weighted["score"].min(), gdf_weighted["score"].max()
        cbar.ax.set_ylabel(f'Score (min={original_min:.2f}, max={original_max:.2f})', fontsize=8)
        
        # Row 2: Greedy algorithm without distance constraint
        if best_nodes_greedy:
            scatter = gdf_weighted.loc[best_nodes_greedy].plot(
                column='norm_score', 
                ax=ax_greedy, 
                markersize=markersize*10, 
                cmap=cmap, 
                label='BSS stations', 
                vmin=0, vmax=1)
        
        # Row 3: Genetic algorithm with distance constraint determined by STATION_MIN_DISTANCE
        if best_nodes_ga:
            scatter = gdf_weighted.loc[best_nodes_ga].plot(
                column='norm_score', 
                ax=ax_ga, 
                markersize=markersize*10,
                cmap=cmap,
                label='BSS stations',
                vmin=0, vmax=1)
        
        # Add Barcelona boundary to all plots
        bcn_boundary.boundary.plot(ax=ax_all_nodes, edgecolor='black', linewidth=1)
        bcn_boundary.boundary.plot(ax=ax_greedy, edgecolor='black', linewidth=1)
        bcn_boundary.boundary.plot(ax=ax_ga, edgecolor='black', linewidth=1)
        
        # Add best nodes text to the plots
        if best_nodes_greedy:
            nodes_scores_greedy = gdf_weighted.loc[best_nodes_greedy]['norm_score'].head(20)
            nodes_scores_greedy.index.name = ''
            nodes_scores_greedy.name = ''
            
            # Convert Series to string representation
            greedy_text = '\n'.join([f"{idx}: {val:.4f}" for idx, val in nodes_scores_greedy.items()])
            
            ax_greedy.text(
                0.01, 0.99, greedy_text,
                ha='left', va='top', fontsize=nodes_scores_fontsize, color='black',
                transform=ax_greedy.transAxes
            )
        
        if best_nodes_ga:
            nodes_scores_ga = gdf_weighted.loc[best_nodes_ga]['norm_score'].head(20)
            nodes_scores_ga.index.name = ''
            nodes_scores_ga.name = ''
            
            # Convert Series to string representation
            ga_text = '\n'.join([f"{idx}: {val:.4f}" for idx, val in nodes_scores_ga.items()])
            
            ax_ga.text(
                0.01, 0.99, ga_text,
                ha='left', va='top', fontsize=nodes_scores_fontsize, color='black',
                transform=ax_ga.transAxes
            )
        
        # Add basemap to all plots
        ctx.add_basemap(ax_all_nodes, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string(), zoom=12)
        ctx.add_basemap(ax_greedy, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string(), zoom=12)
        ctx.add_basemap(ax_ga, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string(), zoom=12)
        
        # Titles
        weights_str = ', '.join([f'{key}: {value}' for key, value in weights.items() if value != 0])
        if len(weights_str) > 70:
            weights_list = [f'{key}: {value}' for key, value in weights.items() if value != 0]
            weights_str = '\n'.join(weights_list)
        
        greedy_score = sum(best_scores_greedy) if isinstance(best_scores_greedy, list) else best_scores_greedy
        ga_score = sum(best_scores_ga) if isinstance(best_scores_ga, list) else best_scores_ga
        
        ax_all_nodes.set_title(f"{weights_str}", fontsize=fontsize_title)
        ax_greedy.set_title(f"Greedy ({STATION_MIN_DISTANCE}m dist.) -> Score: {greedy_score:.3f}", fontsize=fontsize_title)
        
        if ga_experiment_strs and w_idx < len(ga_experiment_strs):
            ga_title = f"GA ({STATION_MIN_DISTANCE}m dist.) -> Score: {ga_score:.3f}\n{ga_experiment_strs[w_idx]}"
        else:
            ga_title = f"GA ({STATION_MIN_DISTANCE}m dist.) -> Score: {ga_score:.3f}"
            
        ax_ga.set_title(ga_title, fontsize=fontsize_title)
        
        # Turn off axis
        ax_all_nodes.set_axis_off()
        ax_greedy.set_axis_off()
        ax_ga.set_axis_off()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    directory = f'{VISUALIZATIONS}/GA/baseline/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(f'{directory}/multi_weights_{N_STATIONS}_stations_{STATION_MIN_DISTANCE}_distance_constraint.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {directory}/multi_weights_{N_STATIONS}_stations_{STATION_MIN_DISTANCE}_distance_constraint.png")
    plt.show()


def compute_and_plot_all_weights(
    df, N_STATIONS, WEIGHTS_COLLECTION, STATION_MIN_DISTANCE, 
    POPULATION_SIZE, GENERATIONS, MUTATION_RATE, ELITE_PROPORTION,
    CROSSOVER_STRATEGY, SELECTION_STRATEGY, n_processes=None
):
    """
    Main function to run the comparison between greedy and genetic algorithms
    for multiple weight combinations in parallel.
    
    Args:
        df (GeoDataFrame): DataFrame with node attributes.
        N_STATIONS (int): Number of stations.
        WEIGHTS_COLLECTION (list): List of weight dictionaries.
        STATION_MIN_DISTANCE (float): Minimum distance between stations. 
            If 0, no distance constraint is applied.
        POPULATION_SIZE (int): Size of population.
        GENERATIONS (int): Number of generations.
        MUTATION_RATE (float): Mutation rate.
        ELITE_PROPORTION (float): Elite proportion.
        CROSSOVER_STRATEGY (str): Crossover strategy.
        SELECTION_STRATEGY (str): Selection strategy.
        n_processes (int, optional): Number of processes to use. If None, uses all available cores.
    """
    # Compute distance matrix
    distance_matrix, id_to_idx, idx_to_id = oh.compute_distance_matrix(df, STATION_MIN_DISTANCE)
    
    # Generate initial population
    initial_population = oh.generate_initial_population(
        N_STATIONS, POPULATION_SIZE, distance_matrix=distance_matrix, 
        id_to_idx=id_to_idx, distance_constraint=(STATION_MIN_DISTANCE > 0), n_jobs=-1
    )
    
    # Set up multiprocessing
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one core free
    
    print(f"Processing {len(WEIGHTS_COLLECTION)} weight combinations using {n_processes} processes...")
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_weight_combination,
        df=df,
        N_STATIONS=N_STATIONS,
        initial_population=initial_population,
        distance_matrix=distance_matrix,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        POPULATION_SIZE=POPULATION_SIZE,
        GENERATIONS=GENERATIONS,
        MUTATION_RATE=MUTATION_RATE,
        ELITE_PROPORTION=ELITE_PROPORTION,
        CROSSOVER_STRATEGY=CROSSOVER_STRATEGY,
        SELECTION_STRATEGY=SELECTION_STRATEGY,
        STATION_MIN_DISTANCE=STATION_MIN_DISTANCE
    )
    
    # Process weight combinations in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_func, WEIGHTS_COLLECTION)
    
    # Unpack results
    best_nodes_greedy_collection = []
    best_scores_greedy_collection = []
    best_nodes_ga_collection = []
    best_scores_ga_collection = []
    ga_experiment_strs = []
    
    for best_nodes_greedy, best_scores_greedy, best_nodes_ga, best_score_ga, ga_experiment_str in results:
        best_nodes_greedy_collection.append(best_nodes_greedy)
        best_scores_greedy_collection.append(best_scores_greedy)
        best_nodes_ga_collection.append(best_nodes_ga)
        best_scores_ga_collection.append(best_score_ga)
        ga_experiment_strs.append(ga_experiment_str)
    
    # Plot comparison
    plot_all_weight_comparison(
        df, 
        N_STATIONS, 
        WEIGHTS_COLLECTION, 
        best_nodes_greedy_collection, 
        best_scores_greedy_collection, 
        best_nodes_ga_collection, 
        best_scores_ga_collection,
        ga_experiment_strs,
        STATION_MIN_DISTANCE=STATION_MIN_DISTANCE
    )


if __name__ == "__main__":
    POPULATION_SIZE = 200
    GENERATIONS = 10_000
    MUTATION_RATE = 0.1
    ELITE_PROPORTION = 0.05
    SELECTION_STRATEGY = 'tournament'
    CROSSOVER_STRATEGY = 'greedy'
    STATION_MIN_DISTANCE = 300
    N_STATIONS = 100

    # 2. Load node attributes and geometries
    file = f'{PR_NODES}/normalized_node_attributes.csv'
    df = pd.read_csv(file)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=EPSG)

    # 3. Define collections of weights
    weights1 = {'population': 0.33, 'education_primary': 0.33, 'unemployment_percentage': 0.33}
    weights2 = {'population': 0.5, 'education_primary': 0.5}
    weights3 = {'bus_lines': 0.33, 'metro_lines': 0.33, 'tram_lines': 0.33}
    weights4 = {'has_bike_lane': 0.5, 'n_economic_retail': 0.3, 'n_green': 0.2}
    WEIGHTS_COLLECTION = [weights1, weights2, weights3, weights4]

    for N_STATIONS in [50]:
        for STATION_MIN_DISTANCE in [50, 100]:
            # 4. Run the comparison and plot
            print(f"Running for {N_STATIONS} stations and {STATION_MIN_DISTANCE}m distance constraint")
            compute_and_plot_all_weights(
                df, N_STATIONS, WEIGHTS_COLLECTION, STATION_MIN_DISTANCE, POPULATION_SIZE, 
                GENERATIONS, MUTATION_RATE, ELITE_PROPORTION,
                CROSSOVER_STRATEGY, SELECTION_STRATEGY, n_processes=4
            )