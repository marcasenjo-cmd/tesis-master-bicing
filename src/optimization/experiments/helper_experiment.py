"""
This module provides utility functions for running and managing optimization experiments
in the BSS optimization pipeline. It handles data loading, experiment setup, and result 
management for various optimization scenarios.

The module handles:
1. **Data Loading**: Node scores, graph data, and distance matrices
2. **Experiment Setup**: Configuration loading and parameter management
3. **Result Management**: Loading and processing optimization results

Key Features:
- **Data Integration**: Automatic loading of all required datasets
- **Elevation Support**: Optional elevation-aware distance calculations

Data Loading:
- **Node Attributes**: Normalized node scores and characteristics
- **Network Graphs**: Road network with optional elevation data
- **Distance Matrices**: All-pairs shortest path computations

Author: Jordi Grau Escolano
"""

import os
import sys
from pathlib import Path
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely import wkt  # type: ignore

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.assign_to_nodes.utils.class_node_assigner as cna
import src.optimization.helper_optimization as ho
import src.optimization.GA.graph_metric.graph_normalization as gn


################################################################################
########################### DATA LOADING #######################################
################################################################################
def _load_node_scores(root='./'):
    file = f'{root}/{PR_NODES}/normalized_node_attributes.csv'
    df = pd.read_csv(file)

    if 'geometry' not in df.columns:
        raise KeyError("The 'geometry' column (containing WKT strings) was not found in the CSV file.")
    
    
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=25831)
    return df


def _load_graph(location, use_elevation: str | bool = False, root='./'):
    BUFFER_SIZE= 300
    if use_elevation != False:
        use_elevation_bool = True
    else:   
        use_elevation_bool = False
    node_assigner =  cna.NodeAttributesAssigner(
        location, graph_path=RAW_GRAPH, crs=25831, buffer_size=BUFFER_SIZE, root=root, use_elevation=use_elevation_bool)
    G = node_assigner.G
    return G


def load_data(city=None, use_elevation: str | bool = False, root='./'):
    """
    Load and prepare data needed for optimization experiments.

    Args:
        city (str): City name. If None, the city is Barcelona.
        use_elevation (str | bool): Whether to use elevation in the distance matrix ('parkin', 'piecewise', or False). Defaults to False.
        root (str): Root directory path, defaults to './'

    Returns:
        tuple: Contains:
            - df (GeoDataFrame): Node scores and attributes
            - G (networkx.Graph): Road network graph
            - distance_matrix (numpy.ndarray): All-pairs shortest path distances
            - id_to_idx (dict): Maps node IDs to matrix indices
            - idx_to_id (dict): Maps matrix indices to node IDs  
            - distance_matrix_binary (numpy.ndarray): Binary matrix indicating if nodes are beyond min distance
            - STATION_MIN_DISTANCE (float): Minimum allowed distance between stations
    """
    # Load configuration
    location = 'Barcelona, Barcelona, Catalunya, España'
    STATION_MIN_DISTANCE = 300

    BUFFER_SIZE= 300
    G = _load_graph(location, use_elevation, root)

    # Node scores
    df = _load_node_scores(root)

    # Check for missing nodes
    graph_nodes = set(G.nodes())
    df_nodes = set(df['node_id'])
    missing_in_graph = df_nodes - graph_nodes
    missing_in_df = graph_nodes - df_nodes

    if missing_in_df or missing_in_graph:
        print("Debugging info:")
        print(f"\tNumber of nodes in graph: {len(G.nodes())}")
        print(f"\tNumber of nodes in DataFrame: {len(df)}")
        print(f"\tFirst few node IDs in graph: {list(G.nodes())[:5]}")
        print(f"\tFirst few node IDs in DataFrame: {df['node_id'].iloc[:5].tolist()}")
    
    if missing_in_graph:
        print(f"\n\tWarning: {len(missing_in_graph)} nodes in DataFrame not found in graph")
        print(f"\tFirst few missing nodes: {list(missing_in_graph)[:5]}")
    if missing_in_df:
        print(f"\n\tWarning: {len(missing_in_df)} nodes in graph not found in DataFrame")
        print(f"\tFirst few missing nodes: {list(missing_in_df)[:5]}")
    
    # Distance matrix
    city = location.split(',')[0]
    distance_matrix, id_to_idx, idx_to_id = ho.compute_all_pairs_shortest_paths_dijkstra(
        city, G, weight='weight', root=root, use_elevation=use_elevation)
    distance_matrix = np.array(distance_matrix)

    # Check for missing mappings
    missing_mappings = [node for node in df['node_id'] if node not in id_to_idx]

    if missing_mappings:
        print(f"\nDebugging ID mappings:")
        print(f"\tNumber of nodes in distance matrix: {len(id_to_idx)}")
        print(f"\tFirst few ID mappings: {dict(list(id_to_idx.items())[:5])}")
    
        print(f"\n\tWarning: {len(missing_mappings)} nodes missing from id_to_idx mapping")
        print(f"\tFirst few missing mappings: {missing_mappings[:5]}")

    # Binary distance matrix
    distance_matrix_binary = (distance_matrix > STATION_MIN_DISTANCE).astype('int8')
    return df, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE


################################################################################
########################### EXPERIMENT MANAGEMENT ##############################
################################################################################
def load_existing_experiments(results_file):
    """Load existing experiments from the master CSV file"""
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        return df
    return pd.DataFrame()


def save_experiment(result_df, results_file, experiment_idx):
    """Save experiment results to the master CSV file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Add execution_id to the results
    result_df['experiment_idx'] = experiment_idx
    
    try:
        if os.path.exists(results_file):
            master_df = pd.read_csv(results_file)
            master_df = pd.concat([master_df, result_df], ignore_index=True)
            print(f"Appending to existing file: {results_file}")
        else:
            print(f"Creating new file: {results_file}")
            master_df = result_df
        
        # Ensure execution_id is the first column
        cols = ['experiment_idx'] + [col for col in master_df.columns if col != 'experiment_idx']
        master_df = master_df[cols]
        master_df.to_csv(results_file, index=False)
        print(f"Successfully saved experiment {experiment_idx}")
        return True
    except Exception as e:
        print(f"Error saving to {results_file}: {e}")
        return False


################################################################################
########################### WEIGHT VECTOR GENERATION ##########################
################################################################################
def generate_weight_vectors(variables, n_samples, seed=None):
    """
    Generate random weight combinations for statistical testing of GA vs MILP.
    
    Args:
        variables: List of variable names to generate weights for
        n_samples: Number of weight combinations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries, where each dictionary maps variable names to their weights.
        Weights sum to 1 for each combination.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n_vars = len(variables)
    weight_combinations = []
    
    for _ in range(n_samples):
        # Generate random weights from uniform distribution
        weights = np.random.random(n_vars)
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        # Create dictionary mapping variables to weights
        weight_dict = {var: float(w) for var, w in zip(variables, weights)}
        weight_combinations.append(weight_dict)
    
    return weight_combinations



################################################################################
########################### GRAPH MARKERS ######################################
################################################################################
def mark_graph(G, S):
    """
    Mark each node in the graph with True in the 'selected' attribute.
    
    Args:
        G (networkx.Graph): The input graph
        S (set): Set of node IDs to mark as selected
        
    Returns:
        networkx.Graph: Graph with updated node attributes
    """
    for node in G.nodes():
        G.nodes[node]['selected'] = node in S
    return G


def randomly_mark_graph(G, N):
    """
    Marks N random nodes in the graph with an attribute 'selected' set to True.
    
    Args:
        G (networkx.Graph): The input graph
        N (int): Number of nodes to mark
        
    Returns:
        tuple: (networkx.Graph, set) - Graph with updated attributes and set of selected nodes
    """
    if N > len(G):
        raise ValueError("N cannot be greater than the number of nodes in the graph.")
    
    selected_nodes = set(random.sample(list(G.nodes()), N))
    return mark_graph(G, selected_nodes), selected_nodes


################################################################################
########################### METRIC BOUNDS COMPUTATION ##########################
################################################################################
def compute_metric_bounds_and_get_nodes(G, distance_matrix, id_to_idx, idx_to_id, n_stations, station_min_distance):
    """
    Compute dispersion and accessibility bounds for a given number of stations.
    
    Args:
        G (networkx.Graph): The graph
        distance_matrix (np.ndarray): Distance matrix
        id_to_idx (dict): Node ID to matrix index mapping
        idx_to_id (dict): Matrix index to node ID mapping
        n_stations (int): Number of stations
        station_min_distance (float): Minimum station distance
        
    Returns:
        tuple: (dispersion_bounds, accessibility_bounds, dispersion_nodes, accessibility_nodes)
    """
    # Compute dispersion bounds
    min_disp, min_disp_nodes = gn.min_dispersion_bound(n_stations, station_min_distance)
    max_disp, max_disp_nodes = gn.max_dispersion_metric(n_stations, distance_matrix, station_min_distance, idx_to_id)
    
    # Compute accessibility bounds
    min_acc, min_acc_nodes = gn.min_accessibility_bound_kmeans(G, n_stations, distance_matrix, id_to_idx)
    max_acc, max_acc_nodes = gn.max_accessibility_bound(n_stations, distance_matrix, idx_to_id, station_min_distance)
    
    # Return bounds and nodes
    disp_bounds = (min_disp, max_disp)
    disp_nodes = (min_disp_nodes, max_disp_nodes)
    acc_bounds = (min_acc, max_acc)
    acc_nodes = (min_acc_nodes, max_acc_nodes)

    return disp_bounds, acc_bounds, disp_nodes, acc_nodes