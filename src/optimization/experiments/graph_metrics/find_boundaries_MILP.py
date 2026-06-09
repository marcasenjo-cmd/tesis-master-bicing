"""
Here the min and max dispersion and accessibility metrics are tried to be computed using 
mixed integer linear programming.

The problem is too big for 32GB of RAM.
"""


import sys
from pathlib import Path
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
from shapely import wkt # type: ignore
import networkx as nx  # type: ignore
import igraph as ig  # type: ignore
import pulp  # type: ignore
import time
import os
import gc  # Garbage collection
import random
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix  # For sparse matrices

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *

# Define fallback paths if not already defined in paths.py
if 'RESULTS_DIR' not in globals():
    RESULTS_DIR = os.path.join(project_root, 'data', 'processed', 'experiments', 'graph_metric')

import src.data_loader as dl
import src.assign_to_nodes.utils.class_node_assigner as cna


def compute_shortest_paths_igraph(G, weight='weight', max_nodes=300):
    """
    Compute all-pairs shortest paths using igraph's shortest_paths_dijkstra function.
    Uses integer indices instead of original node IDs to save memory.
    
    Args:
        G (networkx.Graph): Input NetworkX graph
        weight (str): Edge attribute to use as weight
        max_nodes (int): Maximum number of nodes to use (None for all nodes)
    
    Returns:
        tuple: (distances, idx_to_id) - Distance matrix and mapping
    """
    # Sample nodes if graph is too large
    if max_nodes is not None and len(G.nodes()) > max_nodes:
        print(f"Graph is large ({len(G.nodes())} nodes). Sampling {max_nodes} nodes for optimization.")
        all_nodes = list(G.nodes())
        sampled_nodes = random.sample(all_nodes, max_nodes)
        # Create a subgraph with only sampled nodes
        G = nx.subgraph(G, sampled_nodes)
        del all_nodes, sampled_nodes
        print(f"Created subgraph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Create mappings between original node IDs and integer indices
    nodes = list(G.nodes())
    id_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_id = {i: node for node, i in id_to_idx.items()}
    
    # Convert NetworkX graph to igraph directly with integer indices
    ig_graph = ig.Graph(directed=G.is_directed())
    ig_graph.add_vertices(len(nodes))
    
    # Add edges with weights using integer indices
    edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get(weight, 1.0) for u, v in G.edges()]
    
    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = weights
    
    print(f"Computing all-pairs shortest paths...")
    start_time = time.time()
    
    # Compute shortest paths using igraph's shortest_paths method
    # This returns a matrix of distances between integer indices
    shortest_paths = ig_graph.shortest_paths_dijkstra(weights="weight")
    elapsed = time.time() - start_time
    print(f"Shortest paths computation completed in {elapsed:.2f} seconds")
    
    # Convert to a more memory-efficient format - a numpy array
    distances = np.array(shortest_paths)
    
    # Clean up
    del shortest_paths, ig_graph
    gc.collect()
    
    return distances, idx_to_id


def solve_lp_problem(prob, problem_name):
    """Helper function to solve LP problems with timing and error handling"""
    try:
        print(f"Using Gurobi for {problem_name}")
        solver = pulp.GUROBI_CMD(msg=False)
    except:
        print(f"Using CBC for {problem_name}")
        solver = pulp.PULP_CBC_CMD(msg=False)
    
    solve_start = time.time()
    prob.solve(solver)
    solve_time = time.time() - solve_start
    
    if prob.status != pulp.LpStatusOptimal:
        raise ValueError(f"Could not find optimal solution for {problem_name}")
    
    return solve_time


def dispersion_lp(distances, idx_to_id, N, maximize=True):
    """
    Find the optimal dispersion value using linear programming.
    Works with integer indices and numpy arrays.
    
    Args:
        distances (numpy.ndarray): Distance matrix
        idx_to_id (dict): Mapping from indices to original node IDs
        N (int): Number of nodes to select
        maximize (bool): If True, maximize dispersion; if False, minimize dispersion
    
    Returns:
        tuple: (optimal_value, selected_nodes, solve_time)
    """
    # Create the model
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    problem_name = "MaxDispersion" if maximize else "MinDispersion"
    prob = pulp.LpProblem(problem_name, sense)
    
    # Use integer indices for variables to save memory
    n_nodes = distances.shape[0]
    
    # Decision variables - only create node variables
    nodes_vars = {}
    for i in range(n_nodes):
        nodes_vars[i] = pulp.LpVariable(f"node_{i}", cat='Binary')
    
    # Create edge variables and objective terms directly
    edges_vars = {}
    objective_terms = []
    
    # Process all nodes at once - simpler and not necessarily more memory intensive
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = distances[i, j]
            if dist > 0 and dist < float('inf'):  # Only create variables for valid distances
                var_name = f"edge_{i}_{j}"
                edges_vars[i, j] = pulp.LpVariable(var_name, cat='Binary')
                objective_terms.append(dist * edges_vars[i, j])
                
                # Add constraints immediately
                prob += edges_vars[i, j] <= nodes_vars[i]
                prob += edges_vars[i, j] <= nodes_vars[j]
                prob += edges_vars[i, j] >= nodes_vars[i] + nodes_vars[j] - 1
    
    # Objective function
    prob += pulp.lpSum(objective_terms)
    
    # Constraint: select exactly N nodes
    prob += pulp.lpSum(nodes_vars[i] for i in range(n_nodes)) == N
    
    # Solve the problem
    solve_time = solve_lp_problem(prob, problem_name)
    
    # Get selected indices
    selected_indices = [i for i in range(n_nodes) if nodes_vars[i].value() > 0.5]
    
    # Convert back to original node IDs
    selected_nodes = [idx_to_id[i] for i in selected_indices]
    
    # Calculate actual dispersion value
    optimal_value = 0
    for i in selected_indices:
        for j in selected_indices:
            if i < j:
                optimal_value += distances[i, j]
    
    # Clean up
    del prob, nodes_vars, edges_vars, objective_terms
    gc.collect()
    
    return optimal_value, selected_nodes, solve_time


def accessibility_lp(distances, idx_to_id, N, maximize=False):
    """
    Find the optimal accessibility value using linear programming.
    Works with integer indices and numpy arrays.
    
    Args:
        distances (numpy.ndarray): Distance matrix
        idx_to_id (dict): Mapping from indices to original node IDs
        N (int): Number of nodes to select
        maximize (bool): If True, maximize accessibility (worse coverage); 
                         if False, minimize accessibility (better coverage)
    
    Returns:
        tuple: (optimal_value, selected_nodes, solve_time)
    """
    # Create the model
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    problem_name = "MaxAccessibility" if maximize else "MinAccessibility"
    prob = pulp.LpProblem(problem_name, sense)
    
    # Use integer indices for variables to save memory
    n_nodes = distances.shape[0]
    
    # Decision variables - only create what's needed
    station_vars = {}
    for i in range(n_nodes):
        station_vars[i] = pulp.LpVariable(f"station_{i}", cat='Binary')
    
    min_dist = {}
    closest = {}
    
    # Create all variables at once
    for j in range(n_nodes):
        min_dist[j] = pulp.LpVariable(f"min_dist_{j}", lowBound=0)
        
        # Get all non-zero distances to this node directly from numpy array
        potential_stations = [i for i in range(n_nodes) if i != j and distances[i, j] > 0 and distances[i, j] < float('inf')]
        
        # Create closest variables only for non-zero distances
        for i in potential_stations:
            closest[i, j] = pulp.LpVariable(f"closest_{i}_{j}", cat='Binary')
        
        # Constraint: each node must have exactly one closest station
        if potential_stations:
            prob += pulp.lpSum(closest[i, j] for i in potential_stations) == 1
        
        # Constraint: a node can only be assigned to a selected station
        for i in potential_stations:
            prob += closest[i, j] <= station_vars[i]
        
        # Set minimum distances using big-M method
        M = 1e6
        for i in potential_stations:
            distance = distances[i, j]
            prob += min_dist[j] >= distance - M * (1 - closest[i, j])
            prob += min_dist[j] <= distance + M * (1 - closest[i, j])
    
    # Objective function
    prob += pulp.lpSum(min_dist[j] for j in range(n_nodes))
    
    # Constraint: select exactly N stations
    prob += pulp.lpSum(station_vars[i] for i in range(n_nodes)) == N
    
    # Solve the problem
    solve_time = solve_lp_problem(prob, problem_name)
    
    # Get selected indices
    selected_indices = [i for i in range(n_nodes) if station_vars[i].value() > 0.5]
    
    # Convert back to original node IDs
    selected_nodes = [idx_to_id[i] for i in selected_indices]
    
    # Calculate actual accessibility value
    optimal_value = 0
    for j in range(n_nodes):
        # Find minimum distance from node j to any selected station
        min_distance = float('inf')
        for i in selected_indices:
            if i != j:
                dist = distances[i, j]
                if dist > 0 and dist < float('inf'):
                    min_distance = min(min_distance, dist)
        if min_distance < float('inf'):
            optimal_value += min_distance
    
    # Clean up
    del prob, station_vars, min_dist, closest
    gc.collect()
    
    return optimal_value, selected_nodes, solve_time


def run_optimization(func, args, metric_name, problem_type, optimization_type):
    """Generic wrapper function for running optimizations with timing"""
    print(f"Computing {metric_name}...")
    start_time = time.time()
    result = func(*args)
    elapsed = time.time() - start_time
    print(f"{metric_name} computation completed in {elapsed:.2f} seconds")
    return (metric_name, problem_type, optimization_type, result, elapsed)


def save_results_to_csv(results, N_stations, output_dir=None):
    """Save optimization results to a CSV file"""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"LP_graph_metric_{N_stations}_stations.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for CSV
    data = []
    
    # Check if results is empty
    if not results:
        print("Warning: No results to save!")
        return filepath
    
    for metric_name, (problem_type, optimization_type, result, total_time) in results.items():
        try:
            # Unpack the result tuple safely
            if isinstance(result, tuple) and len(result) == 3:
                optimal_value, selected_nodes, solve_time = result
                
                # Ensure selected_nodes is a list
                if not isinstance(selected_nodes, list):
                    selected_nodes = []
                
                data.append({
                    'n_stations': N_stations,
                    'metric': problem_type,
                    'optimization': optimization_type,
                    'optimal_value': optimal_value,
                    'solve_time': solve_time,
                    'total_time': total_time,
                    'selected_nodes': ','.join(map(str, selected_nodes))
                })
            else:
                print(f"Warning: Invalid result format for {metric_name}")
        except Exception as e:
            print(f"Error processing results for {metric_name}: {e}")
    
    # Only create DataFrame and save if we have data
    if data:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    else:
        print("No valid results to save!")
    
    return filepath


def print_results(results):
    """Print formatted results to console"""
    print("\n===== OPTIMIZATION RESULTS =====")
    
    for metric_name, (_, _, (optimal_value, selected_nodes, solve_time), total_time) in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Optimal value: {optimal_value}")
        print(f"  Solve time: {solve_time:.2f} seconds")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Selected nodes: {selected_nodes[:5]}... (total: {len(selected_nodes)})")
    
    print("\n")


if __name__ == "__main__":
    start_total = time.time()

    # 1. Load configuration
    STATION_MIN_DISTANCE = 300
    N_STATIONS = 100
    EPSG = 25831
    location = 'Barcelona, Barcelona, Catalunya, España'

    # 2. Load graph
    node_assigner = cna.NodeAttributesAssigner(
        location, graph_path=RAW_GRAPH, crs=EPSG, buffer_size=STATION_MIN_DISTANCE)
    G = node_assigner.G
    G_und = G.to_undirected()
    for u, v, data in G_und.edges(data=True):
        data['weight'] = data['length']

    # 3. Load data
    file = f'{PR_NODES}/normalized_node_attributes.csv'
    df = pd.read_csv(file)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=EPSG)

    # 4. Compute all-pairs shortest paths with igraph using integer indices
    distances, idx_to_id = compute_shortest_paths_igraph(G_und, weight='weight')
    
    # Free memory
    del node_assigner, G, G_und, df
    gc.collect()
    
    # 5. Run one optimization at a time and free memory between runs
    results = {}
    
    try:
        print("\nRunning maximum dispersion...")
        start_time = time.time()
        result = dispersion_lp(distances, idx_to_id, N_STATIONS, True)
        elapsed = time.time() - start_time
        results["maximum dispersion"] = ("dispersion", "maximize", result, elapsed)
        gc.collect()
    except Exception as e:
        print(f"Maximum dispersion failed: {e}")
    
    # try:
    #     print("\nRunning minimum dispersion...")
    #     start_time = time.time()
    #     result = dispersion_lp(distances, id_to_idx, idx_to_id, N_STATIONS, False)
    #     elapsed = time.time() - start_time
    #     results["minimum dispersion"] = ("dispersion", "minimize", result, elapsed)
    #     gc.collect()
    # except Exception as e:
    #     print(f"Minimum dispersion failed: {e}")
    
    # try:
    #     print("\nRunning minimum accessibility...")
    #     start_time = time.time()
    #     result = accessibility_lp(distances, id_to_idx, idx_to_id, N_STATIONS, False)
    #     elapsed = time.time() - start_time
    #     results["minimum accessibility"] = ("accessibility", "minimize", result, elapsed)
    #     gc.collect()
    # except Exception as e:
    #     print(f"Minimum accessibility failed: {e}")
    
    # try:
    #     print("\nRunning maximum accessibility...")
    #     start_time = time.time()
    #     result = accessibility_lp(distances, id_to_idx, idx_to_id, N_STATIONS, True)
    #     elapsed = time.time() - start_time
    #     results["maximum accessibility"] = ("accessibility", "maximize", result, elapsed)
    #     gc.collect()
    # except Exception as e:
    #     print(f"Maximum accessibility failed: {e}")
    
    # 6. Print and save results
    print_results(results)
    
    # Debug the results structure
    print("\nDEBUG: Results structure:")
    for key, value in results.items():
        print(f"  {key}: {type(value)}")
        if isinstance(value, tuple):
            print(f"    Length: {len(value)}")
            for i, item in enumerate(value):
                print(f"    Item {i}: {type(item)}")
    
    elapsed_total = time.time() - start_total
    print(f"Total execution time: {elapsed_total:.2f} seconds")
    
    csv_path = save_results_to_csv(results, N_STATIONS)
    print(f"Results saved to CSV: {csv_path}")

    