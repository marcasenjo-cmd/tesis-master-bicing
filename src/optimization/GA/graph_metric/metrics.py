import numba  # type: ignore
import numpy as np  # type: ignore



def dispersion_metric(selected_indices, distance_matrix):
    """
    Computes the dispersion metric for a set of stations using their indices in the distance matrix.

    Args:
        selected_indices: List of node indices selected as stations.
        distance_matrix: Distance matrix of the graph.

    Returns:
        float: Sum of pairwise shortest-path distances between stations.
    """
    # Extract the submatrix for just the stations
    station_distances = distance_matrix[np.ix_(selected_indices, selected_indices)]
    
    # Get total pairwise distances considering both directions
    # Exclude diagonal (self-distances) using a mask
    mask = ~np.eye(len(selected_indices), dtype=bool)
    return station_distances[mask].sum()


@numba.njit(parallel=True, fastmath=True)
def accessibility_metric(selected_indices, distance_matrix):
    """
    Computes the accessibility metric for a set of stations using their indices in the distance matrix.

    Args:
        selected_indices: List of node indices selected as stations.
        distance_matrix: Distance matrix of the graph.
    """
    n_nodes = distance_matrix.shape[0]
    result = 0.0
    
    # Use a boolean array for faster membership testing
    is_station = np.zeros(n_nodes, dtype=numba.boolean)
    for idx in selected_indices:
        is_station[idx] = True
    
    # Parallel loop for computing min distances
    for i in numba.prange(n_nodes):
        if not is_station[i]:
            min_dist = np.inf
            for j in selected_indices:
                dist = distance_matrix[i, j] + distance_matrix[j, i]
                if dist < min_dist:
                    min_dist = dist
            result += min_dist
    
    return result
