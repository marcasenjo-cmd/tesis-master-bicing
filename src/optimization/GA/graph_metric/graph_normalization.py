import sys
import numpy as np  # type: ignore
import networkx as nx  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

# Import from the new metrics file instead
from src.optimization.GA.graph_metric.metrics import dispersion_metric, accessibility_metric


def _find_closest_nodes(first_node, N, distance_matrix, min_station_distance):
    """
    Finds the N closest nodes to the given node that respect minimum distance constraints.
    """
    num_nodes = len(distance_matrix)
    selected_nodes_idx = [first_node]
    
    while len(selected_nodes_idx) < N:
        # Convert to numpy array for faster indexing
        selected_arr = np.array(selected_nodes_idx)
        
        # Create mask for candidates (exclude already selected nodes)
        candidate_mask = np.ones(num_nodes, dtype=bool)
        candidate_mask[selected_arr] = False
        
        # Get valid candidates using vectorized operations
        valid_mask = candidate_mask.copy()
        for sel_idx in selected_arr:
            # Check both directions at once
            valid_mask &= (distance_matrix[sel_idx] >= min_station_distance) & \
                         (distance_matrix[:, sel_idx] >= min_station_distance)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            break
        
        # Find closest valid candidate to first node
        valid_distances = distance_matrix[first_node][valid_indices]
        closest_idx = valid_indices[np.argmin(valid_distances)]
        selected_nodes_idx.append(closest_idx)
    
    return selected_nodes_idx


def _node_idx_with_max_eccentricity(distance_matrix):
    """
    Returns the node with the maximum eccentricity in the directed graph G. 
    Distance is measured as the sum of outgoing and incoming distances.

    Args:
        distance_matrix: numpy array of distances between nodes
    
    Returns:
        tuple: (node, eccentricity) where node is the node with maximum eccentricity
               and eccentricity is its eccentricity value. For directed graphs,
               considers both incoming and outgoing distances.
    """
    # Sum both incoming and outgoing distances for each node
    out_eccentricities = np.sum(distance_matrix, axis=1)
    in_eccentricities = np.sum(distance_matrix, axis=0)
    total_eccentricities = out_eccentricities + in_eccentricities
    
    max_ecc_idx = int(np.argmax(total_eccentricities))
    return max_ecc_idx, total_eccentricities[max_ecc_idx]


def _greedy_max_dispersion(N, distance_matrix, min_station_distance):
    """
    Optimized greedy selection to maximize dispersion for a directed graph while
    respecting minimum distance constraints between all selected nodes.
    
    Args:
        G: NetworkX graph
        N: Number of nodes to select
        distance_matrix: Matrix containing distances in meters between nodes
        min_distance: Minimum required distance between any pair of selected nodes in meters
    """
    num_nodes = len(distance_matrix)
    
    # Use the node with max eccentricity as initial node
    initial_node_idx, _ = _node_idx_with_max_eccentricity(distance_matrix=distance_matrix)
    
    # Initialize arrays to track cumulative distances
    outgoing_sums = distance_matrix[initial_node_idx].copy()
    incoming_sums = distance_matrix[:, initial_node_idx].copy()
    
    # Track selected nodes
    selected_idx = [initial_node_idx]
    
    # Track available nodes
    available = np.ones(num_nodes, dtype=bool)
    available[initial_node_idx] = False
    
    # Select remaining N-1 nodes
    while len(selected_idx) < N:
        # Get nodes that satisfy minimum distance constraint with ALL selected nodes
        valid_mask = np.ones(num_nodes, dtype=bool)
        for sel_idx in selected_idx:
            # Check both directions since it's a directed graph
            # distance_matrix contains actual distances in meters
            valid_mask &= (distance_matrix[sel_idx] >= min_station_distance) & (distance_matrix[:, sel_idx] >= min_station_distance)
        
        # Combine with availability mask
        valid_mask &= available
        
        # Total distances for valid nodes
        total_distances = np.where(
            valid_mask,
            outgoing_sums + incoming_sums,
            0  # invalid nodes or already selected nodes get 0 distance
        )
        
        # If no valid nodes remain, break
        if np.max(total_distances) == 0:
            print(f"Warning: Could not find more valid nodes. Selected {len(selected_idx)}/{N} nodes.")
            break
        
        # Find best candidate
        best_idx = np.argmax(total_distances)
        
        # Update selected nodes
        selected_idx.append(best_idx)
        
        # Mark as unavailable
        available[best_idx] = False
        
        # Update cumulative distances
        outgoing_sums += distance_matrix[best_idx]
        incoming_sums += distance_matrix[:, best_idx]
    
    return selected_idx


def min_dispersion_bound(N, min_station_distance):
    """
    Computes theoretical minimum bound for the dispersion metric while respecting distance constraints.
    The dispersion metric is the sum of pairwise shortest path distances between stations.
    For a directed graph, this includes both A→B and B→A distances. Assumes fully bidirectional edges.
    
    Args:
        N: Number of stations to place
        min_station_distance: Minimum distance between stations
        
    Returns:
        tuple: (min_disp_bound, min_disp_nodes)
            min_disp_bound: Minimum value for dispersion metric
            min_disp_nodes: List of nodes that achieve this minimum value
    """
    min_disp_bound = (N * (N - 1) / 2) * min_station_distance * 2
    return min_disp_bound, []


def max_dispersion_metric(N, distance_matrix, min_station_distance, idx_to_id):
    """
    Approximates the maximum dispersion metric for N stations. Uses a greedy 
    dispersion algorithm to select N nodes that approximately maximize the sum 
    of pairwise distances between them.
    
    Args:
        N: Number of stations to select.
        distance_matrix: Distance matrix of the graph.
        min_station_distance: Minimum distance between stations.
        idx_to_id: Dictionary mapping matrix indices to node IDs.
    
    Returns:
        tuple: (max_disp_metric, selected_nodes)
            max_disp_metric: Approximated maximum dispersion metric value
            selected_nodes: List of selected node IDs
    """
    # Get indices of selected nodes
    selected_indices = _greedy_max_dispersion(N, distance_matrix, min_station_distance)
    
    # Calculate metric using indices
    max_disp_metric = dispersion_metric(selected_indices, distance_matrix)
    
    # Convert indices to node IDs for return value
    selected_nodes = [idx_to_id[idx] for idx in selected_indices]
    
    return max_disp_metric, selected_nodes


def min_accessibility_bound_kmeans(G, N, distance_matrix, id_to_idx):
    """
    Uses k-means clustering on the node positions of graph G to select k representative nodes,
    then computes the accessibility metric for these nodes. 
    
    Parameters:
        G: A NetworkX graph with node attributes 'x' and 'y' (projected coordinates).
        N: The number of nodes to select.
        distance_matrix: Distance matrix of the graph.
        id_to_idx: Dictionary mapping node IDs to matrix indices.
    
    Returns:
        tuple: (min_accessibility, selected_nodes)
            min_accessibility: The accessibility metric value
            selected_nodes: List of selected node IDs
    """
    # Get positions and node list.
    nodes = list(G.nodes())
    pos = []
    for node in nodes:
        node_data = G.nodes[node]
        if 'x' in node_data and 'y' in node_data:
            pos.append([node_data['x'], node_data['y']])
        else:
            raise ValueError("Graph nodes must have 'x' and 'y' attributes for k-means clustering.")
    pos = np.array(pos)
    
    # Run k-means clustering.
    kmeans = KMeans(n_clusters=N, random_state=42).fit(pos)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # For each cluster, select the node closest to the centroid.
    selected_nodes_idx = []
    for cluster in range(N):
        indices = np.where(labels == cluster)[0]
        cluster_nodes_idx = [nodes[i] for i in indices]
        cluster_positions = pos[indices]
        centroid = centroids[cluster]
        
        # Compute Euclidean distances from each node in the cluster to the centroid.
        dists = np.linalg.norm(cluster_positions - centroid, axis=1)
        min_idx = np.argmin(dists)
        selected_node_idx = cluster_nodes_idx[min_idx]
        selected_nodes_idx.append(selected_node_idx)

    # Convert selected node IDs to indices before calculating metric
    selected_indices = [id_to_idx[node_id] for node_id in selected_nodes_idx]
    
    # Calculate metric using indices
    min_accessibility = accessibility_metric(selected_indices, distance_matrix)
    
    return min_accessibility, selected_nodes_idx


def max_accessibility_bound(N, distance_matrix, idx_to_id, min_station_distance):
    """
    Compute an upper bound for the accessibility metric by selecting N nodes
    that respect minimum distance constraints and maximize accessibility.
    """
    max_ecc_node_idx, _ = _node_idx_with_max_eccentricity(distance_matrix)
    
    # Get indices of selected nodes
    selected_indices = _find_closest_nodes(max_ecc_node_idx, N, distance_matrix, min_station_distance)
    
    # Calculate metric using indices
    accessibility = accessibility_metric(selected_indices, distance_matrix)
    
    # Convert indices to node IDs for return value
    selected_nodes = [idx_to_id[idx] for idx in selected_indices]
    
    return accessibility, selected_nodes


def normalize_values(value, min_val, max_val):
    """
    Normalize a value to the [0, 1] range given min and max.
    """
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def evaluate_normalize_and_invert_stations_set(selected_indices, distance_matrix, 
                                               alpha, dispersion_bounds, accessibility_bounds):
    """
    Evaluates station placement quality by combining accessibility and dispersion metrics.
    
    This function takes raw accessibility and dispersion metrics for a set of stations, 
    normalizes them to a [0,1] range by the provided bounds, and then inverts them
    (1 - normalized_value) so that higher values indicate better performance. 
    Then, a weighted composite score based on the alpha parameter is computed.
    
    The accessibility metric measures how well stations cover the network by summing shortest-path
    distances from each node to its nearest station. The dispersion metric sums pairwise distances
    between stations to measure their spread.
    
    Args:
        selected_indices (list): List of node indices selected as stations.
        distance_matrix (numpy.ndarray): Distance matrix of the graph.
        alpha (float): Trade-off parameter between 0 and 1. Higher values prioritize 
            accessibility (alpha=1 considers only accessibility), while lower values prioritize
            dispersion (alpha=0 considers only dispersion).
        dispersion_bounds (tuple): (min, max) bounds for normalizing the dispersion metric.
        accessibility_bounds (tuple): (min, max) bounds for normalizing the accessibility metric.
    
    Returns:
        tuple: containing:
            inverted_dispersion (float): Inverted normalized dispersion. 
                Higher means closer stations.
            inverted_accessibility (float): Inverted normalized accessibility. 
                Higher means stations are more accessible.
            composite_score (float): Weighted combination of inverted metrics based on alpha
    """
    # Compute the raw metrics (lower is better)
    raw_access = accessibility_metric(selected_indices, distance_matrix)
    raw_disp = dispersion_metric(selected_indices, distance_matrix)
    
    # Normalize the metrics to [0,1]
    disp_min, disp_max = dispersion_bounds
    access_min, access_max = accessibility_bounds
    if disp_max == disp_min:
        norm_disp = 0
    else:
        norm_disp = (raw_disp - disp_min) / (disp_max - disp_min)
        
    if access_max == access_min:
        norm_access = 0
    else:
        norm_access = (raw_access - access_min) / (access_max - access_min)

    # Invert the metrics
    inverted_disp = 1 - norm_disp
    inverted_access = 1 - norm_access
    
    # Composite score
    composite_score = alpha * inverted_disp + (1 - alpha) * inverted_access
    
    return inverted_disp, inverted_access, composite_score