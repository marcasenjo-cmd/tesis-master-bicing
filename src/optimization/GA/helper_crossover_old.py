import random
import numpy as np  # type: ignore
import heapq
from numba import jit  # type: ignore
from src.optimization.GA.graph_metric.graph_normalization import evaluate_normalize_and_invert_stations_set

@jit(nopython=True)
def _check_feasibility_numba(candidate_idx, solution_indices, distance_matrix, station_min_distance):
    """
    Numba-optimized helper for feasibility checking
    """
    if len(solution_indices) == 0:
        return True
        
    # Check both directions
    for idx in solution_indices:
        if distance_matrix[candidate_idx, idx] < station_min_distance:
            return False
        if distance_matrix[idx, candidate_idx] < station_min_distance:
            return False
    return True

def _is_feasible(candidate, solution_indices, distance_matrix, id_to_idx, station_min_distance):
    """
    Check if candidate node maintains minimum distance constraint with all nodes in solution
    for a directed graph.
    
    Args:
        candidate: Node ID to check
        solution_indices: List of matrix indices of already selected nodes
        distance_matrix: Distance matrix in meters
        id_to_idx: Dictionary mapping node IDs to matrix indices
        station_min_distance: Minimum required distance between stations in meters
    
    Returns:
        bool: True if candidate maintains distance constraints with all nodes in solution
    """
    if candidate not in id_to_idx:
        raise KeyError(f"Candidate {candidate} not found in id_to_idx.")
    
    if not solution_indices:
        return True
        
    c_idx = id_to_idx[candidate]
    
    # Use numba-optimized function for the actual distance checking
    return _check_feasibility_numba(c_idx, np.array(solution_indices), distance_matrix, station_min_distance)


def _get_combined_scores(candidates, solution_indices, node_scores, distance_matrix, id_to_idx, 
                        graph_score=False, alpha=0.5, metric_bounds=None):
    """
    Calculation of node scores * graph scores for multiple candidates.
    """
    # Get node scores for all candidates
    node_score_values = np.array([node_scores[node] for node in candidates])
    
    if not graph_score:
        return node_score_values
        
    # Calculate graph scores for all candidates
    graph_scores = np.zeros(len(candidates))
    for i, node in enumerate(candidates):
        current_solution = solution_indices + [id_to_idx[node]]
        _, _, graph_scores[i] = evaluate_normalize_and_invert_stations_set(
            current_solution, distance_matrix, alpha, *metric_bounds
        )
    
    return node_score_values * graph_scores

def _crossover_greedy(N, candidates, node_scores, distance_matrix, id_to_idx, station_min_distance=0, 
                     graph_score=False, alpha=0.5, metric_bounds=None):
    """
    Fully greedy: sort candidates by descending combined score and pick feasible nodes in order.
    """
    if station_min_distance > 0:
        new_solution = []
        solution_indices = []
        remaining_candidates = np.array(candidates)
        
        while len(remaining_candidates) > 0 and len(new_solution) < N:
            # Get indices of feasible candidates
            feasible_mask = np.array([
                _is_feasible(node, solution_indices, distance_matrix, id_to_idx, station_min_distance)
                for node in remaining_candidates
            ])
            feasible_candidates = remaining_candidates[feasible_mask]
            
            if len(feasible_candidates) == 0:
                break
                
            # Calculate combined scores for all feasible candidates
            scores = _get_combined_scores(
                feasible_candidates, solution_indices, node_scores, 
                distance_matrix, id_to_idx, graph_score, alpha, metric_bounds
            )
            
            # Get best scoring node
            best_idx = np.argmax(scores)
            best_node = feasible_candidates[best_idx]
            
            new_solution.append(best_node)
            solution_indices.append(id_to_idx[best_node])
            remaining_candidates = remaining_candidates[remaining_candidates != best_node]
            
        return new_solution
    else:
        # For unconstrained case, use vectorized score calculation
        scores = _get_combined_scores(
            candidates, [], node_scores, distance_matrix, 
            id_to_idx, graph_score, alpha, metric_bounds
        )
        # Get top N nodes by score
        top_indices = np.argpartition(scores, -N)[-N:]
        return [candidates[i] for i in top_indices]


def _crossover_weighted_random(N, candidates, node_scores, distance_matrix, id_to_idx, station_min_distance=0,
                             graph_score=False, alpha=0.5, metric_bounds=None):
    """
    Weighted-random approach using combined scores for probabilities.
    """
    new_solution = []
    solution_indices = [] 
    remaining_candidates = np.array(candidates)

    while len(remaining_candidates) > 0 and len(new_solution) < N:
        # Get indices of feasible candidates
        feasible_mask = np.array([
            _is_feasible(node, solution_indices, distance_matrix, id_to_idx, station_min_distance)
            for node in remaining_candidates
        ])
        feasible_candidates = remaining_candidates[feasible_mask]
        
        if len(feasible_candidates) == 0:
            break
            
        # Calculate combined scores for all feasible candidates
        scores = _get_combined_scores(
            feasible_candidates, solution_indices, node_scores, 
            distance_matrix, id_to_idx, graph_score, alpha, metric_bounds
        )
        
        # Convert to probabilities
        total = scores.sum()
        if total <= 0:
            break
            
        probs = scores / total
        chosen_idx = np.random.choice(len(feasible_candidates), p=probs)
        chosen_node = feasible_candidates[chosen_idx]
        
        new_solution.append(chosen_node)
        solution_indices.append(id_to_idx[chosen_node])
        remaining_candidates = remaining_candidates[remaining_candidates != chosen_node]

    return new_solution


def _crossover_top_first(N, candidates, node_scores, distance_matrix, id_to_idx, station_min_distance=0):
    """
    'Top-first' approach:
      1. Sort candidates by descending score.
      2. Shuffle the top quarter (K).
      3. Pick from top quarter, then from the remainder, until we have N or run out.
    
    Args:
        N: Number of nodes to select
        candidates: List of candidate node IDs
        node_scores: Dictionary mapping node IDs to scores
        distance_matrix: Distance matrix in kilometers
        id_to_idx: Dictionary mapping node IDs to matrix indices
        station_min_distance: Minimum required distance between stations in kilometers
    
    Returns:
        list: List of selected node IDs
    """
    new_solution = []
    solution_indices = []
    
    # Sort candidates by descending score
    sorted_nodes = sorted(candidates, key=lambda node: node_scores[node], reverse=True)

    # Shuffle the top K
    K = max(1, len(sorted_nodes) // 4)
    top_subset = sorted_nodes[:K]
    random.shuffle(top_subset)

    # Pick from the shuffled top half
    for node in top_subset:
        if station_min_distance > 0:
            if _is_feasible(node, solution_indices, distance_matrix, id_to_idx, station_min_distance):
                new_solution.append(node)
                solution_indices.append(id_to_idx[node])
        else:
            new_solution.append(node)
            solution_indices.append(id_to_idx[node])

        if len(new_solution) == N:
            return new_solution

    # If still not full, pick from the remainder
    remaining = [node for node in sorted_nodes[K:] if node not in new_solution]
    random.shuffle(remaining)
    for node in remaining:
        if station_min_distance > 0:
            if _is_feasible(node, solution_indices, distance_matrix, id_to_idx, station_min_distance):
                new_solution.append(node)
                solution_indices.append(id_to_idx[node])
        else:
            new_solution.append(node)
            solution_indices.append(id_to_idx[node])

        if len(new_solution) == N:
            return new_solution

    return new_solution


def fallback_fill(N, partial_solution, candidates, distance_matrix, id_to_idx, station_min_distance=0):
    """
    Fallback strategy: shuffle all candidates, try to fill up to N.
    Keeps already chosen nodes from partial_solution.
    
    Args:
        N: Target number of nodes
        partial_solution: List of already selected node IDs
        candidates: List of candidate node IDs to choose from
        distance_matrix: Distance matrix in meters
        id_to_idx: Dictionary mapping node IDs to matrix indices
        station_min_distance: Minimum required distance between stations in meters
    
    Returns:
        list: Solution with N nodes (if possible) respecting distance constraints
    """
    new_solution = partial_solution[:]
    solution_indices = [id_to_idx[node] for node in new_solution]
    random.shuffle(candidates)

    for node in candidates:
        if node not in new_solution:
            if station_min_distance > 0:
                if _is_feasible(node, solution_indices, distance_matrix, id_to_idx, station_min_distance):
                    new_solution.append(node)
                    solution_indices.append(id_to_idx[node])
            else:
                new_solution.append(node)
                solution_indices.append(id_to_idx[node])

            if len(new_solution) == N:
                break
                
    return new_solution