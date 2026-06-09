"""
This module provides core utility functions for BSS optimization,
including distance calculations, elevation adjustments, and optimization algorithms.

The module handles:
1. **Elevation Adjustments**: Multiple methods for terrain-aware distance calculations
2. **Distance Computations**: All-pairs shortest path algorithms with elevation considerations
3. **Optimization Utilities**: Helper functions for population-based genetic algorithms
4. **Graph Analysis**: Network analysis computations

Elevation Adjustment Methods:
- **Piecewise**: Empirical cycling model with slope-based penalties/benefits
- **Parkin & Rotheram**: Physics-based model with speed coefficients
- **Empirical**: Real-world cycling behavior modeling

Distance Computation:
- **All-pairs shortest paths**: Complete distance matrix computation
- **Elevation integration**: Computation of equivalent flat distance for elevation-aware routing

Author: Jordi Grau Escolano
"""

import sys
import random
import time
from pathlib import Path
import numpy as np  # type:ignore
from scipy.spatial.distance import cdist  # type:ignore
from joblib import Parallel, delayed  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import contextily as ctx  # type:ignore
import igraph as ig  # type:ignore
import os
import pickle
import networkx as nx  # type:ignore

project_root = Path().resolve().parents[0]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.GA.graph_metric.graph_normalization as gn

np.random.seed(1)
random.seed(1)


# An empirical basis for route choice in cycling = https://sci-hub.box/10.1080/02640410400023282
# https://en.wikipedia.org/wiki/Naismith%27s_rule
# Langmuir's extension of Naismith's rule for walkers
# https://www.scitepress.org/Papers/2018/69370/69370.pdf

################################################################################
########################### ELEVATION ADJUSTMENT ###############################
################################################################################
def adjust_elevation_piecewise(distance, elevation):
    """
    Adjust the weight of an edge based on the elevation.
    """
    if elevation > 0:
        # For positive elevation: add 8 meters per meter of elevation gain
        adjusted_weight = distance + (elevation * 8)
    
    elif elevation < 0:
        # Calculate the slope percentage
        slope_percentage = (elevation / distance) * 100
        drop = abs(elevation)
        
        # For negative elevation: use three zones:
        # 1. Steep declines (< -12%): add 10 min for every 300m descent
        # 2. Gentle declines (-5% to 0%): subtract 10 min for every 300m descent
        # 3. Mid-range (-12% to -5%): no adjustment

        # Convert the "10 minutes per 300m" to an equivalent distance
        # Assuming average cycling speed of 12 km/h (3.33 m/s) -> Cycling into the workshop
        # 10 minutes = 600 seconds; 600 seconds * 3.33 m/s = 2000 meters
        eq_per_m = 2000 / 300 # meters equivalent to 10 minutes of cycling

        if slope_percentage < -12: # steep decline: penalty
            adjusted_weight = distance + (drop * eq_per_m)
        elif slope_percentage >= -5:  # Gentle decline: benefit
            adjusted_weight = distance - (drop * eq_per_m)
        else:  # Mid-range: no adjustment
            adjusted_weight = distance
    else:
        # For zero elevation: keep original weight
        adjusted_weight = distance
    return adjusted_weight


def adjust_elevation_parkin(distance, elevation, clamp_up=True):
    """
    Equivalent flat‐distance (m) based on Parkin & Rotheram (2010) Table 2,
    Coefficients are per unit slope, valid from –7% to +7%.
    """
    # Base parameters (m/s, per unit slope)
    v0_ms     = 6.01     # flat‐ground speed
    b_up_ms   = -40.02   # slowdown per unit uphill slope
    b_down_ms = -23.79   # speedup per unit downhill slope

    # Compute fractional slope
    slope = elevation / distance if distance > 0 else 0.0
    
    # Clamp slope to data range. 
    # Bike have breaks for downhill trips. 
    # Uphill trips are more tricky.
    clamp_up   = 0.10  # Reduces base speed (6.01 m/s) to 3.21 m/s, 11.6 km/h
    clamp_down = -0.10  # Increases base speed (6.01 m/s) to 8.01 m/s, 28.8 km/h
    if clamp_up:
        slope_clamped = max(min(slope,  clamp_up), clamp_down)
    else:
        # Only clamp downhill
        slope_clamped = min(slope, clamp_down)

    # Separate uphill and downhill components
    slope_up   = max(slope_clamped, 0.0)
    slope_down = min(slope_clamped, 0.0)

    # Compute speed contributions
    contrib_up   = b_up_ms   * slope_up
    contrib_down = b_down_ms * slope_down

    # Compute adapted speed
    v_ms = v0_ms + contrib_up + contrib_down
    v_ms = max(v_ms, 0.1)  # minimum speed floor

    # Compute travel time and equivalent flat distance
    t_s     = distance / v_ms
    eq_dist = t_s * v0_ms

    return eq_dist


################################################################################
########################### DISTANCE MATRIX COMPUTATION #######################
################################################################################
def compute_all_pairs_shortest_paths_dijkstra(
    city, nx_graph, weight='weight', use_elevation: str | bool=False, force_recompute=False, root='./',
):
    """
    Compute weighted shortest path distances for all pairs of nodes using Dijkstra's algorithm 
    via igraph's distances() method. Can load pre-computed results or save new computations.

    Parameters:
        city (str): The city of the graph, for saving and loading purposes.
        nx_graph (networkx.Graph): The original weighted NetworkX graph.
        weight (str): The edge attribute name to use as weight.
        use_elevation (str | bool): 'parkin', 'piecewise', or False. Defaults to False.
        force_recompute (bool): Whether to force recomputation even if saved data exists.
        root (str): The root directory of the project.  

    Returns:
        tuple: (distance_matrix, id_to_idx, idx_to_id) where:
            - distance_matrix: numpy.ndarray of shortest path distances
            - id_to_idx: dict mapping node IDs to matrix indices
            - idx_to_id: dict mapping matrix indices to node IDs
    """
    save_dir = f"{root}/{RAW_GRAPH}/"
    matrix_path = f"{save_dir}/distance_matrix.npy" 
    mappings_path = f"{save_dir}/node_mappings.pkl"
    mappings_path_protocol4 = f"{save_dir}/node_mappings_protocol4.pkl"

    if city != 'Barcelona':
        matrix_path = f"{save_dir}/different_cities/distance_matrix_{city}.npy" 
        mappings_path = f"{save_dir}/different_cities/node_mappings_{city}.pkl"
        mappings_path_protocol4 = f"{save_dir}/different_cities/node_mappings_protocol4_{city}.pkl"
    
    # If using elevation, modify the matrix path
    if use_elevation != False:
        matrix_path = matrix_path.replace('.npy', f'_with_elevation_{use_elevation}.npy')
    
    # Check if saved data exists and load it if requested
    if not force_recompute and os.path.exists(matrix_path) and os.path.exists(mappings_path):
        print(f"Loading pre-computed distance matrix and mappings...")
        try:
            # Load distance matrix (uncompressed for faster loading)
            distance_matrix = np.load(matrix_path, mmap_mode='r')  # Memory-mapped for instant loading
            
            # Try loading protocol 5 first, then protocol 4
            try:
                with open(mappings_path, 'rb') as f:
                    mappings = pickle.load(f)
                print("\tSuccessfully loaded mappings with protocol 5")
            except:
                print("\tProtocol 5 failed, trying protocol 4...")
                try:
                    with open(mappings_path_protocol4, 'rb') as f:
                        mappings = pickle.load(f)
                    print("\tSuccessfully loaded mappings with protocol 4")
                except:
                    print("\tFailed to load mappings with protocol 4")
                    raise Exception("Failed to load mappings with protocol 4")
            
            id_to_idx = mappings['id_to_idx']
            idx_to_id = mappings['idx_to_id']
            
            print(f"Successfully loaded distance matrix of shape {distance_matrix.shape}")
            return distance_matrix, id_to_idx, idx_to_id
        
        except Exception as e:
            print(f"Error loading saved data: {str(e)}")
            print("Will recompute distance matrix...")

    # If using elevation, adjust the weights in the graph
    if use_elevation != False:
        print("Adjusting edge weights based on elevation extension...")
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            distance = float(data[weight])
            elevation = data.get('elevation', 0)
            if use_elevation == 'piecewise':
                adjusted_weight = adjust_elevation_piecewise(distance, elevation)
            elif use_elevation == 'parkin':
                adjusted_weight = adjust_elevation_parkin(distance, elevation)
            else:
                raise ValueError(f"Unknown elevation model: {use_elevation}")
            nx_graph.edges[u, v, key][f"{weight}_elevation"] = adjusted_weight

    # Compute the distance matrix
    # Convert the NetworkX graph to an igraph graph
    print("Converting NetworkX graph to igraph...")
    ig_graph = ig.Graph.from_networkx(nx_graph)
    
    # Create dictionaries using the '_nx_name' attribute from igraph vertices
    print("Creating id-to-idx and idx-to-id dictionaries...")
    id_to_idx = {}
    idx_to_id = {}
    for v in ig_graph.vs:
        idx = v.index
        node_id = v["_nx_name"]
        id_to_idx[node_id] = idx
        idx_to_id[idx] = node_id
    
    # Compute all pairs shortest path distances
    print(f"\nComputing all-pairs shortest paths for {len(ig_graph.vs)} nodes...")
    start_time = time.time()
    if use_elevation != False:
        distance_matrix = np.array(ig_graph.distances(weights=f"{weight}_elevation"))
    else:
        distance_matrix = np.array(ig_graph.distances(weights=weight))
    end_time = time.time()
    print(f"Time taken to compute distance matrix: {end_time - start_time:.2f} seconds")

    # Save the results for future use
    print(f"Saving distance matrix to {save_dir}")
    try:
        # Save distance matrix
        np.save(matrix_path, distance_matrix)  # Faster to load than compressed
        
        # Save mappings in both protocol 4 and 5
        mappings = {
            'id_to_idx': id_to_idx,
            'idx_to_id': idx_to_id
        }
        
        # Save with protocol 5
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save with protocol 4
        with open(mappings_path_protocol4, 'wb') as f:
            pickle.dump(mappings, f, protocol=4)
        
        print("Save successful")
    except Exception as e:
        print(f"Error saving data: {str(e)}")
    
    return distance_matrix, id_to_idx, idx_to_id
    

################################################################################
########################### INITIAL POPULATION GENERATION ######################
################################################################################
def generate_one_initial_solution(
    N, distance_matrix, min_station_distance, idx_to_id, id_to_idx, required_nodes=None, seed=None,
    max_attempts=50,
):
    """
    Generate ONE initial solution of N nodes using vectorized operations.
    
    Args:
        N: Number of nodes to select
        distance_matrix: Matrix of distances in meters
        min_station_distance: Minimum distance required between stations in meters
        idx_to_id: Dictionary mapping matrix indices to node IDs
        id_to_idx: Dictionary mapping node IDs to matrix indices
        required_nodes: List of node IDs that must be included in the solution (no distance constraints between them)
        seed: Random seed for reproducibility
        max_attempts: Maximum number of attempts to generate a valid solution
    
    Returns:
        list: Selected node IDs that form a valid solution
        
    Note:
        - Required nodes don't need to maintain distance constraints between themselves
        - However, any new node must maintain distance constraints with ALL previously selected nodes
          (both required and non-required)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_nodes = len(distance_matrix)
    
    # Convert required_nodes to indices
    required_indices = []
    if required_nodes:
        if len(required_nodes) > N:
            raise ValueError("More required nodes than N")
        required_indices = [id_to_idx[node] for node in required_nodes]
    
    for attempt in range(max_attempts):
        # Start with required nodes
        selected_indices = required_indices.copy()
        
        # If no required nodes, start with a random node
        if not selected_indices:
            all_candidates = np.arange(num_nodes)
            np.random.shuffle(all_candidates)
            selected_indices = [all_candidates[0]]
        
        while len(selected_indices) < N:
            # Create a mask of valid nodes using matrix operations
            valid_mask = np.ones(num_nodes, dtype=bool)
            
            # A new node must maintain distance constraints with ALL previously selected nodes
            for sel_idx in selected_indices:
                valid_mask &= (distance_matrix[sel_idx] >= min_station_distance) & \
                            (distance_matrix[:, sel_idx] >= min_station_distance)
            
            # Remove already selected nodes
            valid_mask[selected_indices] = False
            
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                break
            
            # Among valid nodes, choose one randomly
            selected_indices.append(np.random.choice(valid_indices))
            
        
        if len(selected_indices) == N:
            return [idx_to_id[idx] for idx in selected_indices]
    
    print(f"Warning: Could only find {len(selected_indices)} nodes after {max_attempts} attempts")
    return [idx_to_id[idx] for idx in selected_indices]


def generate_initial_population(N, population_size, distance_matrix, id_to_idx, min_station_distance, required_nodes=None, n_jobs=-1):
    """
    Generate multiple initial solutions in parallel. 
    Only keep those that have exactly N nodes.
    
    Args:
        N: Number of nodes to select
        population_size: Size of the population to generate
        distance_matrix: Matrix of distances in meters
        id_to_idx: Dictionary mapping node IDs to matrix indices
        min_station_distance: Minimum distance required between stations in meters
        required_nodes: List of node IDs that must be included in each solution
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
    """
    # Generate different seeds for each parallel execution
    seeds = list(range(population_size))
    random.shuffle(seeds)
    
    # Create idx_to_id once for all parallel executions
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

    population = Parallel(n_jobs=n_jobs)(
        delayed(generate_one_initial_solution)(
            N, distance_matrix, min_station_distance, idx_to_id, id_to_idx, required_nodes, seed
        )
        for seed in seeds
    )

    return [sol for sol in population if sol is not None and len(sol) == N]


def validate_solution(
        solution, distance_matrix, id_to_idx, idx_to_id, N, min_station_distance, required_nodes=None,
):
    """
    Vectorized validation that a solution has N distinct nodes and validates distance constraints
    for non-required nodes.
    
    Args:
        solution: List of node IDs
        distance_matrix: Distance matrix in meters
        id_to_idx: Dictionary mapping node IDs to matrix indices
        N: Expected number of nodes
        min_station_distance: Minimum required distance between stations in meters
        required_nodes: List of node IDs that are required (no distance constraints between them)
    
    Returns:
        bool: True if solution is valid, False otherwise
        
    Note:
        Performs two checks for non-required nodes:
        1. They must maintain minimum distance between themselves
        2. They must maintain minimum distance with required nodes
        Required nodes don't need to maintain any distance constraints between themselves.
    """
    # Basic validation
    if len(solution) != N:
        print(f"Solution length is not N: {len(solution)} != {N}")
        return False
    
    if len(set(solution)) != N:
        print(f"Solution contains duplicate nodes")
        return False
        
    # If no required nodes specified, treat all nodes as non-required
    if required_nodes is None:
        required_nodes = []
    
    # Get non-required nodes
    non_required_nodes = [node for node in solution if node not in required_nodes]
    
    # If all nodes are required, solution is valid
    if not non_required_nodes:
        return True
        
    # Convert nodes to indices
    non_required_indices = [id_to_idx[node] for node in non_required_nodes]
    required_indices = [id_to_idx[node] for node in required_nodes if node in solution]
    
    # Check 1: Non-required nodes must maintain distance between themselves
    if len(non_required_indices) >= 2:
        submatrix = distance_matrix[np.ix_(non_required_indices, non_required_indices)]
        mask = ~np.eye(len(non_required_indices), dtype=bool)
        if np.any(submatrix[mask] < min_station_distance):
            print("\nInvalid distances found between non-required nodes:")
            for i, node1_idx in enumerate(non_required_indices):
                for j, node2_idx in enumerate(non_required_indices):
                    if i != j and distance_matrix[node1_idx, node2_idx] < min_station_distance:
                        node1, node2 = idx_to_id[node1_idx], idx_to_id[node2_idx]
                        dist = distance_matrix[node1_idx, node2_idx]
                        print(f"Distance from {node1} to {node2}: {dist:.2f} < {min_station_distance}")
            return False
    
    # Check 2: Non-required nodes must maintain distance with required nodes
    if required_indices:
        for non_req_idx in non_required_indices:
            for req_idx in required_indices:
                dist_to = distance_matrix[non_req_idx, req_idx]
                dist_from = distance_matrix[req_idx, non_req_idx]
                if dist_to < min_station_distance or dist_from < min_station_distance:
                    print("\nInvalid distances found between non-required and required nodes:")
                    non_req_node = idx_to_id[non_req_idx]
                    req_node = idx_to_id[req_idx]
                    if dist_to < min_station_distance:
                        print(f"Distance from {non_req_node} to {req_node}: {dist_to:.2f} < {min_station_distance}")
                    if dist_from < min_station_distance:
                        print(f"Distance from {req_node} to {non_req_node}: {dist_from:.2f} < {min_station_distance}")
                    return False
    
    return True


################################################################################
########################### NODE NORMALIZATION #################################
################################################################################
def sum_and_normalize_all_node_scores(df, weights):
    """
    Calculate node scores for solutions in 'pop'.
    `df` must contain each node's variables, keyed by 'node_id'.
    """
    df['score'] = sum(weights[var] * df[var] for var in weights.keys())
    score_min, score_max = df["score"].min(), df["score"].max()
    df["norm_score"] = (df["score"] - score_min) / (score_max - score_min)
    if 'node_id' in df.columns:
        df = df.set_index('node_id')

    # Drop columns that weights are 0
    df = df.drop(columns=[var for var in weights.keys() if weights[var] == 0])
    return df


################################################################################
##################### POPULATION SCORE COMPUTATION #############################
################################################################################
def calculate_node_scores(population, df):
    """
    Calculate fitness scores for solutions in 'pop'.
    `df` must contain each node's variables, keyed by 'node_id'.
    """
    node_scores = df['norm_score'].to_dict()
    scores = np.array([sum(node_scores[n] for n in individual) for individual in population])

    return scores


def calculate_graph_scores(population, distance_matrix, id_to_idx, alpha=0.5, metric_bounds=None):
    """
    Calculate graph scores for solutions in population using parallel processing.
    
    Args:
        population: List of solutions
        distance_matrix: Distance matrix
        id_to_idx: Mapping from node IDs to matrix indices
        alpha: Weight between dispersion and accessibility (0-1)
        metric_bounds: Pre-computed (dispersion_bounds, accessibility_bounds) tuple

    
    Returns:
        numpy.ndarray: Array of composite scores for each solution
    """
    # Compute bounds if not provided
    dispersion_bounds, accessibility_bounds = metric_bounds

    # Convert all nodes ids to idxs
    population_idxs = [[id_to_idx[node] for node in solution] for solution in population]
    
    results = [
        gn.evaluate_normalize_and_invert_stations_set(
            solution_idxs, distance_matrix, alpha, dispersion_bounds, accessibility_bounds
        ) 
        for solution_idxs in population_idxs
    ]

    composite_scores = [result[2] for result in results]
    
    return composite_scores


def calculate_population_scores(population, df, distance_matrix, 
                                id_to_idx, graph_score=False, alpha=0.5, metric_bounds=None,
                                score_combination='multiply', penalty_power=2):
    """
    Calculate fitness scores for solutions in 'pop'.
    
    Args:
        population: List of solutions
        df: DataFrame with node scores
        distance_matrix: Distance matrix
        id_to_idx: Mapping from node IDs to matrix indices
        graph_score: Whether to include graph metrics
        alpha: Weight between dispersion and accessibility (0-1)
        metric_bounds: Pre-computed (dispersion_bounds, accessibility_bounds) tuple
        score_combination: How to combine node and graph scores ('multiply', 'exponential', 'power_penalty')
        penalty_power: Power to use for penalizing bad graph scores (default 2)
    """
    node_scores = calculate_node_scores(population, df)
    if graph_score:
        if metric_bounds is None:
            raise ValueError("Metric bounds must be provided if graph_score is True")
        graph_scores = calculate_graph_scores(
            population, distance_matrix, id_to_idx, alpha, metric_bounds
        )
        
        # Different ways to combine scores
        if score_combination == 'multiply':
            scores = node_scores * graph_scores
        elif score_combination == 'exponential':
            scores = np.power(node_scores, graph_scores)
        elif score_combination == 'power_penalty':
            # Penalize bad graph scores more heavily
            penalized_graph_scores = np.power(graph_scores, penalty_power)
            scores = node_scores * penalized_graph_scores
        else:
            raise ValueError(f"Unknown score combination method: {score_combination}")
    else:
        scores = node_scores
    return scores


################################################################################
########################### POPULATION DIVERSITY ###############################
################################################################################
def jaccard_distance(sol1, sol2):
    """
    Computes the Jaccard distance between two solutions interpreted as sets.
    sol1 and sol2 can be lists; we'll convert them to sets internally.
    """
    # Convert to sets only if they're not already sets
    set1 = sol1 if isinstance(sol1, set) else set(sol1)
    set2 = sol2 if isinstance(sol2, set) else set(sol2)
    
    intersection_size = len(set1 & set2)
    union_size = len(set1) + len(set2) - intersection_size  # More efficient than len(set1 | set2)
    
    if union_size == 0:
        return 0.0  # both sets empty => distance 0
    jaccard_similarity = intersection_size / union_size
    return 1.0 - jaccard_similarity


def average_jaccard_diversity(population):
    """
    Computes the average pairwise Jaccard distance among all individuals.
    Each individual is a list (but treated as a set for this calculation).
    """
    if len(population) < 2:
        return 0.0
    
    # Convert all solutions to sets once
    population_sets = [set(sol) for sol in population]
    
    total_distance = 0.0
    count = 0
    
    # Use numpy for faster computation
    n = len(population_sets)
    for i in range(n):
        for j in range(i+1, n):
            total_distance += jaccard_distance(population_sets[i], population_sets[j])
            count += 1

    return total_distance / count


################################################################################
########################### PLOTTING ###########################################
################################################################################
def plot_stations(best_nodes, gdf_nodes, title, filename, root='.', save=False, show=False):
    # Settings
    cmap = plt.cm.viridis_r
    markersize = 0.5*15    
    score_col = 'norm_score'
    best_nodes_scores = gdf_nodes.loc[best_nodes][score_col].apply(lambda x: round(x, 4))
    title_fontsize = 8
    
    # Load barcelona boundary
    bcn_boundary = dl.load_bcn_boundary()
    
    # Plot
    fig, axs = plt.subplots(1,2, figsize=(12, 12), gridspec_kw={'width_ratios': [0.1, 1]})
    axs = axs.flatten()

    # Plot all nodes
    scatter = gdf_nodes.loc[best_nodes].plot(
        column=score_col,
        cmap=cmap,
        ax=axs[1],
        markersize=markersize,
        legend=False,
        vmin=0, vmax=1)

    # Colorbar in a separate subplot
    cbar = plt.colorbar(scatter.collections[0], ax=axs[0], shrink=1.3)
    cbar.ax.tick_params(labelsize=8, labelleft=True, labelright=False)
    original_min, original_max = round(gdf_nodes['score'].min(),4), round(gdf_nodes['score'].max(),4)
    bar_str = f'Normalized score\n(originally min={original_min}, max={original_max})'
    cbar.ax.set_ylabel(bar_str, fontsize=8, rotation=90, labelpad=7)
    cbar.ax.yaxis.set_label_position('left')    
    
    # Add barcelona boundary
    bcn_boundary.boundary.plot(ax=axs[1], edgecolor='black', linewidth=1)

    # Add best nodes text to the plot
    best_nodes_scores.index.name = ''
    best_nodes_scores.name = ''
    axs[1].text(
        0.01, 0.99, best_nodes_scores.sort_values(ascending=False).to_string(),
        ha='left', va='top', fontsize=7, color='black',
        transform=axs[1].transAxes
    )

    # Add basemap
    ctx.add_basemap(
        axs[1], source=ctx.providers.CartoDB.Positron, crs=gdf_nodes.crs.to_string(), zoom=12
    )
    
    axs[1].set_title(title, fontsize=title_fontsize)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    
    # Save figure
    if save:
        plt.savefig(f"{root}/{filename}", bbox_inches='tight', dpi=300)
        plt.close()
    if show:
        plt.show()
        plt.close()