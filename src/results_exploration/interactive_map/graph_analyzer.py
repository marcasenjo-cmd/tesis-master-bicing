"""
Graph analyzer module for network analysis and path calculations.
"""

import random
from itertools import combinations
import networkx as nx

def calculate_paths_and_distances(solution, G, distance_matrix, id_to_idx, idx_to_id):
    """
    Calculate shortest paths and distances between all node pairs.
    
    Args:
        solution (list): List of node IDs in the solution
        G (networkx.Graph): NetworkX graph
        distance_matrix (numpy.ndarray): Matrix of distances between nodes
        id_to_idx (dict): Dictionary mapping node IDs to indices
        idx_to_id (dict): Dictionary mapping indices to node IDs
        
    Returns:
        tuple: (dict of paths, dict of distances)
    """
    all_paths = {}
    all_distances = {}
    
    for node1_id in solution:
        for node2_id in solution:
            if node1_id == node2_id:
                continue
                
            try:
                # Get indices for the distance matrix
                node1_idx = id_to_idx[node1_id]
                node2_idx = id_to_idx[node2_id]
            except KeyError:
                # Handle case where we might have indices instead of IDs
                node1_idx = node1_id
                node2_idx = node2_id
                node1_id = idx_to_id[node1_idx]
                node2_id = idx_to_id[node2_idx]
            
            try:
                # Calculate shortest path
                path = nx.shortest_path(G, node1_id, node2_id, weight='weight')
                all_paths[(node1_id, node2_id)] = path
                all_distances[(node1_id, node2_id)] = distance_matrix[node1_idx][node2_idx]
            except nx.NetworkXNoPath:
                print(f"No path found between nodes {node1_id} and {node2_id}")
                continue
                
    return all_paths, all_distances

def generate_pair_colors(solution, seed=None):
    """
    Generate random colors for the edges between each node pair.
    
    Args:
        solution (list): List of node IDs in the solution
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        dict: Mapping of node pairs to colors
    """
    if seed is not None:
        random.seed(seed)
        
    colors = {}
    for node1_id, node2_id in combinations(solution, 2):
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        colors[(node1_id, node2_id)] = color
        colors[(node2_id, node1_id)] = color
    return colors


def get_path_coordinates(path, df_weighted_osm):
    """
    Get coordinates for a path.
    
    Args:
        path (list): List of node IDs in the path
        df_weighted_osm (GeoDataFrame): DataFrame with node data
        
    Returns:
        list: List of [lat, lon] coordinates
    """
    coords = []
    for node in path:
        try:
            node_geom = df_weighted_osm.loc[node]['geometry']
            coords.append([node_geom.y, node_geom.x])
        except (KeyError, TypeError) as e:
            print(f"Error getting coordinates for node {node}: {e}")
            continue
    return coords 