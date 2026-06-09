import random
from numba import jit, prange
import numpy as np
from src.optimization.GA.graph_metric.graph_normalization import evaluate_normalize_and_invert_stations_set


@jit(nopython=True)
def _check_feasibility_numba(candidate_idx, solution_indices, distance_matrix, station_min_distance):
    """
    Numba‐optimized helper for checking a single candidate_idx against solution_indices.
    """
    if solution_indices.shape[0] == 0:
        return True

    for idx in solution_indices:
        if distance_matrix[candidate_idx, idx] < station_min_distance:
            return False
        if distance_matrix[idx, candidate_idx] < station_min_distance:
            return False
    return True

def _is_feasible(candidate, solution_indices, distance_matrix, id_to_idx, station_min_distance):
    """
    Check if `candidate` (node ID) is feasible given an array/list of selected indices.
    """
    if candidate not in id_to_idx:
        raise KeyError(f"Candidate {candidate} not found in id_to_idx.")
    
    # Handle both lists and numpy arrays
    if hasattr(solution_indices, 'shape'):
        # It's a numpy array
        if solution_indices.shape[0] == 0:
            return True
        solution_arr = solution_indices
    else:
        # It's a list
        if len(solution_indices) == 0:
            return True
        solution_indices = np.array(solution_indices, dtype=np.int64)

    c_idx = id_to_idx[candidate]
    return _check_feasibility_numba(c_idx, solution_indices, distance_matrix, station_min_distance)


@jit(nopython=True, parallel=True)
def batch_is_feasible(
    rem_idx_arr,          # shape (R,), dtype=int64 ― remaining candidate‐indices
    sol_idx_arr,          # shape (sol_size,), dtype=int64 ― currently selected indices
    distance_matrix,      # shape (V, V), dtype=float64
    station_min_distance  # float64 or int64
):
    """
    Return a Boolean array mask of length R: for each rem_idx_arr[i],
    True if it is feasible vs. sol_idx_arr under the distance constraint.
    """
    R = rem_idx_arr.shape[0]
    out = np.empty((R,), dtype=np.bool_)
    sol_size = sol_idx_arr.shape[0]

    for i in prange(R):
        c_idx = rem_idx_arr[i]

        # If the solution is empty, everything is feasible
        if sol_size == 0:
            out[i] = True
            continue

        feasible = True
        for j in range(sol_size):
            s_idx = sol_idx_arr[j]
            if distance_matrix[c_idx, s_idx] < station_min_distance or \
               distance_matrix[s_idx, c_idx] < station_min_distance:
                feasible = False
                break

        out[i] = feasible

    return out


def _get_combined_scores(candidates, solution_indices, node_scores, distance_matrix, id_to_idx,
                         graph_score=False, alpha=0.5, metric_bounds=None):
    """
    Calculation of node‐scores * graph‐scores for multiple candidates.
    (UNCHANGED from your original code.)
    """
    # Get node scores for all candidates
    node_score_values = np.array([node_scores[node] for node in candidates])

    if not graph_score:
        return node_score_values

    # Calculate graph scores for all candidates
    graph_scores = np.zeros(len(candidates))
    for i, node in enumerate(candidates):
        current_solution = np.concatenate([solution_indices, np.array([id_to_idx[node]])])
        _, _, graph_scores[i] = evaluate_normalize_and_invert_stations_set(
            current_solution, distance_matrix, alpha, *metric_bounds
        )

    return node_score_values * graph_scores

def _crossover_greedy(
    N,
    candidates,           # Python list of node‐IDs
    node_scores,          # Python dict { node_ID: score }
    distance_matrix,      # NumPy array (V × V)
    id_to_idx,            # dict { node_ID → idx (0…V−1) }
    idx_to_id,            # dict { idx → node_ID }
    station_min_distance=0,
    graph_score=False,
    alpha=0.5,
    metric_bounds=None,
):
    """
    Fully‐greedy: pick up to N nodes from `candidates`, subject to a minimum‐distance
    constraint and optional graph‐score. Uses one batch feasibility check per iteration,
    then calls the original _get_combined_scores().
    """
    # 2) Create a flat array of all candidate‐indices and an 'alive' mask
    cand_idx_arr = np.array([id_to_idx[node] for node in candidates], dtype=np.int64)
    alive_mask   = np.ones(cand_idx_arr.shape[0], dtype=np.bool_)  # True = still available

    # 3) Pre‐allocate a fixed‐size array to hold selected indices; track sol_size
    solution_indices = np.empty((N,), dtype=np.int64)
    sol_size = 0

    # 4) Unpack metric_bounds
    if metric_bounds is None:
        lower_bound, upper_bound = 0.0, 1.0
    else:
        lower_bound, upper_bound = metric_bounds

    # 5) If unconstrained + no graph_score, pick top‐N by node_scores directly
    if station_min_distance == 0 and not graph_score:
        # Build an array of all candidate IDs in order of cand_idx_arr
        # (we want the same ordering as cand_idx_arr, so use `candidates` directly)
        c_id_arr = np.array(candidates, dtype=object)
        # Gather scores in the same order
        scores = np.array([node_scores[nid] for nid in c_id_arr], dtype=np.float64)
        # If there are ≤ N candidates, just return all
        if scores.shape[0] <= N:
            return list(c_id_arr)
        # Otherwise, pick the top‐N positions (unsorted)
        top_pos = np.argpartition(scores, -N)[-N:]
        top_ids = c_id_arr[top_pos]
        return list(top_ids)

    # 6) Otherwise, do up to N iterations, each with one batch‐feasibility check
    while sol_size < N:
        # 6a) Which positions in cand_idx_arr are still alive?
        rem_positions = np.where(alive_mask)[0]  # shape (R,)
        if rem_positions.size == 0:
            break  # no more candidates

        # 6b) Map those positions to actual node‐indices
        rem_idx_arr = cand_idx_arr[rem_positions]  # shape (R,)

        # 6c) Run batch feasibility (returns a Boolean mask of length R)
        feasible_mask = batch_is_feasible(
            rem_idx_arr,
            solution_indices[:sol_size],  # slice of current solution
            distance_matrix,
            station_min_distance
        )

        # 6d) If none are feasible, we’re done
        if not np.any(feasible_mask):
            break

        # 6e) Filter rem_positions by feasible_mask to get feasible_positions
        feasible_positions = rem_positions[feasible_mask]  # indices into cand_idx_arr

        # 6f) Build a Python list of feasible candidate IDs, to feed into _get_combined_scores()
        feasible_candidates = [
            idx_to_id[int(cand_idx_arr[pos])]
            for pos in feasible_positions
        ]

        # 6g) Call the original _get_combined_scores() on those feasible IDs
        scores = _get_combined_scores(
            feasible_candidates,
            solution_indices[:sol_size],  # these are node‐indices from previous picks
            node_scores,
            distance_matrix,
            id_to_idx,
            graph_score,
            alpha,
            metric_bounds
        )
        # `scores` is a 1D numpy array of length len(feasible_candidates)

        # 6h) Pick the best one
        best_local = np.argmax(scores)
        best_node = feasible_candidates[best_local]

        # 6i) Convert best_node (ID) → its integer index in [0..V−1], then find its position in cand_idx_arr
        best_c_idx = id_to_idx[best_node]
        best_pos   = feasible_positions[best_local]  # index into cand_idx_arr

        # 6j) Add best_c_idx to our solution_indices array
        solution_indices[sol_size] = best_c_idx
        sol_size += 1

        # 6k) Mark that candidate as removed
        alive_mask[best_pos] = False

    # 7) Convert selected indices back to node‐IDs:
    return [idx_to_id[int(solution_indices[i])] for i in range(sol_size)]


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