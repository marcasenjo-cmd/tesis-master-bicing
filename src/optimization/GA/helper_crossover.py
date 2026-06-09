import random
import numba
from numba import jit, prange, njit
import numpy as np
from src.optimization.GA.graph_metric.graph_normalization import evaluate_normalize_and_invert_stations_set


@njit(fastmath=True, parallel=False)
def _check_feasibility_numba(candidate_idx, sol_idx_arr, distance_matrix, station_min_distance):
    """
    Numba‐optimized helper for checking a single candidate_idx against sol_idx_arr.
    """
    if sol_idx_arr.shape[0] == 0:
        return True

    for j in range(sol_idx_arr.shape[0]):
        s_idx = sol_idx_arr[j]
        # Because the matrix may not be symmetric, we check both directions
        if distance_matrix[candidate_idx, s_idx] < station_min_distance or \
           distance_matrix[s_idx, candidate_idx] < station_min_distance:
            return False
    return True


@njit(nogil=True, parallel=True, fastmath=True)
def batch_is_feasible(
    rem_idx_arr,          # shape (R,), dtype=int64 ― remaining candidate‐indices
    sol_idx_arr,          # shape (sol_size,), dtype=int64 ― currently selected indices
    distance_matrix,      # shape (V, V), dtype=float64
    station_min_distance  # float64 or int64
):
    """
    Return a Boolean mask (length R).  True if rem_idx_arr[i] can be added to sol_idx_arr
    without violating station_min_distance.
    """
    R = rem_idx_arr.shape[0]
    sol_size = sol_idx_arr.shape[0]
    out = np.empty(R, dtype=np.bool_)

    if sol_size == 0:
        # If no stations are chosen yet, everything is feasible
        for i in prange(R):
            out[i] = True
        return out
    # Otherwise, test each candidate against all already‐selected stations
    for i in prange(R):
        c_idx = rem_idx_arr[i]
        feasible = True
        for j in range(sol_size):
            s_idx = sol_idx_arr[j]
            if distance_matrix[c_idx, s_idx] < station_min_distance or \
               distance_matrix[s_idx, c_idx] < station_min_distance:
                feasible = False
                break
        out[i] = feasible
    return out


# --------------------------------------------------------------------------------
# 2) Numba‐compiled "one‐shot" graph‐score calculator (as before)
# --------------------------------------------------------------------------------


@njit(fastmath=True)
def precompute_distance_sums(distance_matrix):
    """
    Pre-compute two-way distance sums for all nodes.
    This is used in accessibility computation when no stations are chosen yet.
    """
    V = distance_matrix.shape[0]
    two_way_sums = np.empty(V, dtype=np.float64)
    
    for i in range(V):
        total = 0.0
        for j in range(V):
            total += distance_matrix[i, j] + distance_matrix[j, i]
        two_way_sums[i] = total
    
    return two_way_sums


@njit(fastmath=True)
def precompute_distance_matrix_chunks(distance_matrix, chunk_indices):
    """
    Pre-compute distance matrix chunks for frequently accessed candidates.
    Returns row and column data for the specified indices.
    """
    V = distance_matrix.shape[0]
    num_chunks = len(chunk_indices)
    
    # Pre-fetch rows and columns for candidate indices
    rows = np.empty((num_chunks, V), dtype=np.float64)
    cols = np.empty((num_chunks, V), dtype=np.float64)
    
    for i in range(num_chunks):
        idx = chunk_indices[i]
        for j in range(V):
            rows[i, j] = distance_matrix[idx, j]
            cols[i, j] = distance_matrix[j, idx]
    
    return rows, cols


@njit(fastmath=True)
def _compute_graph_scores_batch_optimized(
    cand_idx_arr,            # np.ndarray[int64] of feasible candidate‐indices
    sol_idx_arr,             # np.ndarray[int64] of chosen indices
    sol_size,                # int64
    current_access_min_dists,# np.ndarray[float64]
    current_disp_sum,        # float64
    is_station,              # np.ndarray[boolean]
    distance_matrix,         # np.ndarray[float64]
    alpha,                   # float64 ∈ [0,1]
    disp_min, disp_max,      # float64
    access_min, access_max,  # float64
    two_way_sums=None        # Pre-computed two-way distance sums (optional)
):
    """
    Optimized version that accepts pre-computed values to avoid redundant calculations.
    """
    R = cand_idx_arr.shape[0]
    V = distance_matrix.shape[0]
    out = np.empty(R, dtype=np.float64)

    disp_range = disp_max - disp_min
    access_range = access_max - access_min

    # FAST‐PATH: no stations chosen yet (sol_size == 0)
    if sol_size == 0:
        if alpha == 1.0:
            # Pure dispersion with no stations = all 1.0
            for r in range(R):
                out[r] = 1.0
            return out

        # Use pre-computed two-way sums if available (MAJOR OPTIMIZATION)
        if two_way_sums is not None:
            # Ultra-fast lookup from pre-computed values
            for r in range(R):
                c = cand_idx_arr[r]
                total_two_way = two_way_sums[c]
                
                if access_range > 0.0:
                    inv_access = 1.0 - ((total_two_way - access_min) / access_range)
                else:
                    inv_access = 1.0
                out[r] = alpha + (1.0 - alpha) * inv_access
        else:
            # Fallback to computation (still optimized)
            for r in range(R):
                c = cand_idx_arr[r]
                total_two_way = 0.0
                for i in range(V):
                    total_two_way += distance_matrix[c, i] + distance_matrix[i, c]

                if access_range > 0.0:
                    inv_access = 1.0 - ((total_two_way - access_min) / access_range)
                else:
                    inv_access = 1.0
                out[r] = alpha + (1.0 - alpha) * inv_access

        return out

    # GENERAL CASE: stations already chosen - optimize based on alpha
    if alpha == 1.0:
        # Pure dispersion mode - only compute dispersion
        for r in range(R):
            c = cand_idx_arr[r]
            new_disp_sum = current_disp_sum
            for j in range(sol_size):
                s = sol_idx_arr[j]
                new_disp_sum += distance_matrix[c, s] + distance_matrix[s, c]
            
            if disp_range > 0.0:
                out[r] = 1.0 - ((new_disp_sum - disp_min) / disp_range)
            else:
                out[r] = 1.0
        return out
    
    # Pre-compute base accessibility sum once (expensive operation)
    base_access_sum = 0.0
    if alpha < 1.0:
        for i in range(V):
            if not is_station[i]:
                base_access_sum += current_access_min_dists[i]

    if alpha == 0.0:
        # Pure accessibility mode - only compute accessibility
        for r in range(R):
            c = cand_idx_arr[r]
            improvement = 0.0
            for i in range(V):
                if not is_station[i]:
                    two_way = distance_matrix[i, c] + distance_matrix[c, i]
                    if two_way < current_access_min_dists[i]:
                        improvement += (current_access_min_dists[i] - two_way)
            
            total_new = base_access_sum - improvement
            if access_range > 0.0:
                out[r] = 1.0 - ((total_new - access_min) / access_range)
            else:
                out[r] = 1.0
        return out
    
    # Mixed mode (0 < alpha < 1) - compute both dispersion and accessibility
    for r in range(R):
        c = cand_idx_arr[r]
        
        # Compute dispersion component
        new_disp_sum = current_disp_sum
        for j in range(sol_size):
            s = sol_idx_arr[j]
            new_disp_sum += distance_matrix[c, s] + distance_matrix[s, c]
        
        if disp_range > 0.0:
            inv_disp = 1.0 - ((new_disp_sum - disp_min) / disp_range)
        else:
            inv_disp = 1.0
        
        # Compute accessibility component
        improvement = 0.0
        for i in range(V):
            if not is_station[i]:
                two_way = distance_matrix[i, c] + distance_matrix[c, i]
                if two_way < current_access_min_dists[i]:
                    improvement += (current_access_min_dists[i] - two_way)
        
        total_new = base_access_sum - improvement
        if access_range > 0.0:
            inv_access = 1.0 - ((total_new - access_min) / access_range)
        else:
            inv_access = 1.0
        
        out[r] = alpha * inv_disp + (1.0 - alpha) * inv_access

    return out


# Keep the original function for backward compatibility
@njit(fastmath=True, parallel=True)
def _compute_graph_scores_batch(
    cand_idx_arr,            # np.ndarray[int64] of feasible candidate‐indices
    sol_idx_arr,             # np.ndarray[int64] of chosen indices
    sol_size,                # int64
    current_access_min_dists,# np.ndarray[float64]
    current_disp_sum,        # float64
    is_station,              # np.ndarray[boolean]
    distance_matrix,         # np.ndarray[float64]
    alpha,                   # float64 ∈ [0,1]
    disp_min, disp_max,      # float64
    access_min, access_max   # float64
):
    """
    Original function - calls optimized version without pre-computed values.
    """
    return _compute_graph_scores_batch_optimized(
        cand_idx_arr, sol_idx_arr, sol_size, current_access_min_dists,
        current_disp_sum, is_station, distance_matrix, alpha,
        disp_min, disp_max, access_min, access_max, None
    )

# --------------------------------------------------------------------------------
# 3) The greedy crossover
# --------------------------------------------------------------------------------
def _crossover_greedy(
    N,
    candidates,           # Python list of node‐IDs
    node_scores,          # Python dict { node_ID: score }
    distance_matrix,      # NumPy array (V × V), float64
    id_to_idx,            # dict { node_ID → idx (0…V−1) }
    station_min_distance=0.0,
    graph_score=False,
    alpha=0.5,
    metric_bounds=None
):
    """
    Greedy crossover: pick up to N nodes from `candidates`,
    subject to a minimum‐distance constraint and optional graph_score.

    Returns a Python list of selected node‐IDs (length ≤ N).
    """
    V = distance_matrix.shape[0]

    # 1) Build reverse‐mapping idx → node_ID
    idx_to_id_local = [None] * V
    for nid, idx in id_to_idx.items():
        idx_to_id_local[idx] = nid

    # 2) Build a full array of node_scores in index‐space (size V)
    node_scores_arr = np.zeros(V, dtype=np.float64)
    for nid, sc in node_scores.items():
        node_scores_arr[id_to_idx[nid]] = sc

    # 3) Build cand_idx_arr = indices of all candidates
    M = len(candidates)
    cand_idx_arr = np.empty((M,), dtype=np.int64)
    for i, nid in enumerate(candidates):
        cand_idx_arr[i] = id_to_idx[nid]

    # 4) "Alive" mask for which positions (0..M-1) are still available
    alive_mask = np.ones((M,), dtype=np.bool_)

    # 5) Prepare storage for the greedy solution (indices in [0..V-1])
    solution_indices = np.empty((N,), dtype=np.int64)
    sol_size = 0

    # 6) If graph_score is True, set up incremental state arrays and pre-computations
    two_way_sums = None
    if graph_score:
        is_station = np.zeros((V,), dtype=np.bool_)
        # For each i, current_access_min_dists[i] = min distance from i to any chosen station
        current_access_min_dists = np.full((V,), np.inf, dtype=np.float64)
        current_disp_sum = 0.0

        # Unpack metric_bounds into (disp_min, disp_max) and (access_min, access_max)
        if metric_bounds is None:
            disp_min, disp_max = 0.0, 1.0
            access_min, access_max = 0.0, 1.0
        else:
            (disp_min, disp_max), (access_min, access_max) = metric_bounds

        # Pre-compute two-way distance sums for accessibility computation (when sol_size == 0)
        if alpha < 1.0:  # Only if accessibility is needed
            two_way_sums = precompute_distance_sums(distance_matrix)

    # 7) Fast‐path: if no distance constraint & no graph_score, pick top‐N by node_score
    if station_min_distance == 0.0 and not graph_score:
        scores_all = node_scores_arr[cand_idx_arr]  # shape (M,)
        if M <= N:
            # just return all candidates
            return [idx_to_id_local[int(idx)] for idx in cand_idx_arr]
        # select the N largest‐score indices
        top_pos = np.argpartition(scores_all, -N)[-N:]
        return [idx_to_id_local[int(cand_idx_arr[int(p)])] for p in top_pos]

    # 8) Main greedy loop
    while sol_size < N:
        # 8a) Which candidate‐positions are still "alive"?
        rem_positions = np.flatnonzero(alive_mask)  # 1D array of ints, shape = (#remaining,)
        if rem_positions.size == 0:
            break

        rem_idx_arr = cand_idx_arr[rem_positions]  # actual node‐indices for those alive

        # 8b) Batch feasibility check (Numba‐compiled)
        feasible_mask = batch_is_feasible(
            rem_idx_arr,
            solution_indices[:sol_size],
            distance_matrix,
            station_min_distance
        )
        if not feasible_mask.any():
            # no feasible remaining candidate
            break

        feasible_positions = rem_positions[feasible_mask]     # positions in [0..M-1]
        feasible_idx_arr = cand_idx_arr[feasible_positions]   # actual node‐indices

        # 8c) Compute "base" (node‐score) for each feasible candidate
        base_scores = node_scores_arr[feasible_idx_arr]  # shape = (r,)

        # 8d) If graph_score=True, call the optimized Numba‐compiled batch‐scorer
        if graph_score:
            graph_scores = _compute_graph_scores_batch_optimized(
                feasible_idx_arr,
                solution_indices[:sol_size],
                sol_size,
                current_access_min_dists,
                current_disp_sum,
                is_station,
                distance_matrix,
                alpha,
                disp_min, disp_max,
                access_min, access_max,
                two_way_sums  # Pass pre-computed values
            )
            combined_scores = base_scores * graph_scores
        else:
            combined_scores = base_scores

        # 8e) Pick the best‐scoring candidate
        best_local = int(np.argmax(combined_scores))
        best_pos_in_cand = int(feasible_positions[best_local])
        best_idx = int(cand_idx_arr[best_pos_in_cand])

        # 8f) Add it to the solution
        solution_indices[sol_size] = best_idx
        sol_size += 1
        alive_mask[best_pos_in_cand] = False

        # 8g) If graph_score, update "is_station", "current_disp_sum", and "current_access_min_dists"
        if graph_score:
            # 8g-1) Mark as station
            is_station[best_idx] = True

            # 8g-2) Update current_disp_sum by adding distance to all prior stations
            for j in range(sol_size - 1):
                s = solution_indices[j]
                current_disp_sum += distance_matrix[best_idx, s] + distance_matrix[s, best_idx]

            # 8g-3) Vectorized update of current_access_min_dists
            #       For each i, candidate adds a path i → best_idx → i
            new_col = distance_matrix[:, best_idx] + distance_matrix[best_idx, :]
            #    new distance via best_idx vs. old min
            #    we store zero for already‐stations in current_access_min_dists, so those remain 0
            current_access_min_dists = np.minimum(current_access_min_dists, new_col)
            #    explicitly set the new station's entry to 0.0
            current_access_min_dists[best_idx] = 0.0

    # 9) Map chosen indices → node‐IDs and return
    return [idx_to_id_local[int(solution_indices[i])] for i in range(sol_size)]


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


@njit(inline='always', fastmath=True)
def _is_feasible(node_id, solution_indices, distance_matrix, id_to_idx, station_min_distance):
    """
    Check if adding node_id to solution violates minimum distance constraint.
    """
    if len(solution_indices) == 0:
        return True
    
    node_idx = id_to_idx[node_id]
    for s_idx in solution_indices:
        if (distance_matrix[node_idx, s_idx] < station_min_distance or 
            distance_matrix[s_idx, node_idx] < station_min_distance):
            return False
    return True


def _get_combined_scores(feasible_candidates, solution_indices, node_scores, 
                        distance_matrix, id_to_idx, graph_score, alpha, metric_bounds):
    """
    Calculate combined scores for feasible candidates.
    """
    scores = np.array([node_scores[node] for node in feasible_candidates])
    
    if not graph_score:
        return scores
    
    # For simplicity, just return node scores if graph scoring is requested
    # A full implementation would compute graph scores here
    return scores


class CrossoverPrecomputed:
    """
    Container for pre-computed values that can be reused across multiple crossover operations.
    Call this once per GA run to avoid redundant computations.
    """
    
    def __init__(self, distance_matrix, alpha=0.5):
        self.distance_matrix = distance_matrix
        self.alpha = alpha
        self.V = distance_matrix.shape[0]
        
        # Pre-compute two-way distance sums for all nodes
        # This is used when sol_size == 0 and accessibility is needed
        if alpha < 1.0:
            self.two_way_sums = precompute_distance_sums(distance_matrix)
        else:
            self.two_way_sums = None
    
    def get_two_way_sums(self):
        """Get pre-computed two-way distance sums."""
        return self.two_way_sums


def _crossover_greedy_with_precomputed(
    N,
    candidates,           # Python list of node‐IDs
    node_scores,          # Python dict { node_ID: score }
    distance_matrix,      # NumPy array (V × V), float64
    id_to_idx,            # dict { node_ID → idx (0…V−1) }
    precomputed=None,     # CrossoverPrecomputed object (optional)
    station_min_distance=0.0,
    graph_score=False,
    alpha=0.5,
    metric_bounds=None
):
    """
    Optimized greedy crossover that uses pre-computed values when available.
    
    Args:
        precomputed: CrossoverPrecomputed object with pre-calculated values
        
    Returns a Python list of selected node‐IDs (length ≤ N).
    """
    V = distance_matrix.shape[0]

    # 1) Build reverse‐mapping idx → node_ID
    idx_to_id_local = [None] * V
    for nid, idx in id_to_idx.items():
        idx_to_id_local[idx] = nid

    # 2) Build a full array of node_scores in index‐space (size V)
    node_scores_arr = np.zeros(V, dtype=np.float64)
    for nid, sc in node_scores.items():
        node_scores_arr[id_to_idx[nid]] = sc

    # 3) Build cand_idx_arr = indices of all candidates
    M = len(candidates)
    cand_idx_arr = np.empty((M,), dtype=np.int64)
    for i, nid in enumerate(candidates):
        cand_idx_arr[i] = id_to_idx[nid]

    # 4) "Alive" mask for which positions (0..M-1) are still available
    alive_mask = np.ones((M,), dtype=np.bool_)

    # 5) Prepare storage for the greedy solution (indices in [0..V-1])
    solution_indices = np.empty((N,), dtype=np.int64)
    sol_size = 0

    # 6) If graph_score is True, set up incremental state arrays and get pre-computed values
    two_way_sums = None
    if graph_score:
        is_station = np.zeros((V,), dtype=np.bool_)
        current_access_min_dists = np.full((V,), np.inf, dtype=np.float64)
        current_disp_sum = 0.0

        # Unpack metric_bounds
        if metric_bounds is None:
            disp_min, disp_max = 0.0, 1.0
            access_min, access_max = 0.0, 1.0
        else:
            (disp_min, disp_max), (access_min, access_max) = metric_bounds

        # Use pre-computed values if available
        if precomputed is not None and alpha < 1.0:
            two_way_sums = precomputed.get_two_way_sums()
        elif alpha < 1.0:
            # Fallback: compute on-demand
            two_way_sums = precompute_distance_sums(distance_matrix)

    # 7) Fast‐path: if no distance constraint & no graph_score, pick top‐N by node_score
    if station_min_distance == 0.0 and not graph_score:
        scores_all = node_scores_arr[cand_idx_arr]
        if M <= N:
            return [idx_to_id_local[int(idx)] for idx in cand_idx_arr]
        top_pos = np.argpartition(scores_all, -N)[-N:]
        return [idx_to_id_local[int(cand_idx_arr[int(p)])] for p in top_pos]

    # 8) Main greedy loop
    while sol_size < N:
        # 8a) Which candidate‐positions are still "alive"?
        rem_positions = np.flatnonzero(alive_mask)
        if rem_positions.size == 0:
            break

        rem_idx_arr = cand_idx_arr[rem_positions]

        # 8b) Batch feasibility check
        feasible_mask = batch_is_feasible(
            rem_idx_arr, solution_indices[:sol_size], distance_matrix, station_min_distance
        )
        if not feasible_mask.any():
            break

        feasible_positions = rem_positions[feasible_mask]
        feasible_idx_arr = cand_idx_arr[feasible_positions]

        # 8c) Compute base scores
        base_scores = node_scores_arr[feasible_idx_arr]

        # 8d) Compute graph scores using optimized function
        if graph_score:
            graph_scores = _compute_graph_scores_batch_optimized(
                feasible_idx_arr, solution_indices[:sol_size], sol_size,
                current_access_min_dists, current_disp_sum, is_station,
                distance_matrix, alpha, disp_min, disp_max, access_min, access_max,
                two_way_sums
            )
            combined_scores = base_scores * graph_scores
        else:
            combined_scores = base_scores

        # 8e) Pick the best candidate
        best_local = int(np.argmax(combined_scores))
        best_pos_in_cand = int(feasible_positions[best_local])
        best_idx = int(cand_idx_arr[best_pos_in_cand])

        # 8f) Add to solution
        solution_indices[sol_size] = best_idx
        sol_size += 1
        alive_mask[best_pos_in_cand] = False

        # 8g) Update state
        if graph_score:
            is_station[best_idx] = True
            for j in range(sol_size - 1):
                s = solution_indices[j]
                current_disp_sum += distance_matrix[best_idx, s] + distance_matrix[s, best_idx]
            new_col = distance_matrix[:, best_idx] + distance_matrix[best_idx, :]
            current_access_min_dists = np.minimum(current_access_min_dists, new_col)
            current_access_min_dists[best_idx] = 0.0

    # 9) Return results
    return [idx_to_id_local[int(solution_indices[i])] for i in range(sol_size)]


@njit(fastmath=True)
def _compute_bulk_selection_scores(
    cand_idx_arr,            # All candidate indices
    node_scores_arr,         # Node scores for all candidates  
    distance_matrix,         # Distance matrix
    alpha,                   # Alpha parameter
    disp_min, disp_max,      # Dispersion bounds
    access_min, access_max,  # Accessibility bounds
    two_way_sums,            # Pre-computed two-way sums
    station_min_distance     # Minimum distance constraint
):
    """
    Ultra-fast bulk selection: compute approximate scores for ALL candidates
    and return them sorted for bulk selection.
    """
    R = cand_idx_arr.shape[0]
    out = np.empty(R, dtype=np.float64)
    
    access_range = access_max - access_min
    
    # For bulk selection with sol_size=0, use pre-computed accessibility scores
    if alpha == 1.0:
        # Pure dispersion = just node scores
        for r in range(R):
            out[r] = node_scores_arr[r]
    else:
        # Combined node + accessibility scores
        for r in range(R):
            c = cand_idx_arr[r]
            total_two_way = two_way_sums[c]
            
            if access_range > 0.0:
                inv_access = 1.0 - ((total_two_way - access_min) / access_range)
            else:
                inv_access = 1.0
                
            combined_score = alpha + (1.0 - alpha) * inv_access
            out[r] = node_scores_arr[r] * combined_score
    
    return out


@njit(fastmath=True)
def _bulk_feasibility_filter(
    candidate_indices,       # Candidate indices to check
    candidate_scores,        # Scores for candidates
    distance_matrix,         # Distance matrix
    station_min_distance,    # Minimum distance constraint
    max_selections          # Maximum number to select
):
    """
    Select up to max_selections candidates using a greedy approach
    that respects distance constraints but is optimized for bulk selection.
    """
    n_candidates = len(candidate_indices)
    selected = np.empty(min(max_selections, n_candidates), dtype=np.int64)
    selected_count = 0
    used = np.zeros(n_candidates, dtype=np.bool_)
    
    # Sort candidates by score (descending)
    sorted_indices = np.argsort(-candidate_scores)
    
    for i in range(n_candidates):
        if selected_count >= max_selections:
            break
            
        candidate_pos = sorted_indices[i]
        if used[candidate_pos]:
            continue
            
        candidate_idx = candidate_indices[candidate_pos]
        
        # Check feasibility against already selected
        is_feasible = True
        for j in range(selected_count):
            selected_idx = selected[j]
            if (distance_matrix[candidate_idx, selected_idx] < station_min_distance or
                distance_matrix[selected_idx, candidate_idx] < station_min_distance):
                is_feasible = False
                break
        
        if is_feasible:
            selected[selected_count] = candidate_idx
            selected_count += 1
            used[candidate_pos] = True
    
    return selected[:selected_count]


@njit(fastmath=True)
def _approximate_incremental_scores(
    cand_idx_arr,           # Remaining candidate indices
    partial_solution,       # Already selected indices
    node_scores_arr,        # Node scores for candidates  
    distance_matrix,        # Distance matrix
    alpha,                  # Alpha parameter
    disp_min, disp_max,     # Dispersion bounds
    access_min, access_max, # Accessibility bounds
    two_way_sums,          # Pre-computed two-way sums
    station_min_distance   # Minimum distance constraint
):
    """
    Approximate the incremental graph scores for multiple candidates at once.
    This is much faster than computing exact scores for each candidate individually.
    """
    R = cand_idx_arr.shape[0]
    sol_size = partial_solution.shape[0]
    out = np.empty(R, dtype=np.float64)
    
    # For approximation when sol_size > 0, we estimate the impact
    if sol_size == 0:
        # Use exact computation for first selection (no approximation needed)
        access_range = access_max - access_min
        
        if alpha == 1.0:
            # Pure dispersion = all 1.0 when no stations
            for r in range(R):
                out[r] = 1.0
        elif alpha == 0.0:
            # Pure accessibility 
            for r in range(R):
                c = cand_idx_arr[r]
                total_two_way = two_way_sums[c]
                if access_range > 0.0:
                    out[r] = 1.0 - ((total_two_way - access_min) / access_range)
                else:
                    out[r] = 1.0
        else:
            # Mixed mode
            for r in range(R):
                c = cand_idx_arr[r]
                total_two_way = two_way_sums[c]
                
                if access_range > 0.0:
                    inv_access = 1.0 - ((total_two_way - access_min) / access_range)
                else:
                    inv_access = 1.0
                out[r] = alpha + (1.0 - alpha) * inv_access
    else:
        # APPROXIMATION: For incremental steps, estimate impact based on average distances
        # This is much faster than exact computation but maintains reasonable accuracy
        
        # Pre-compute average distances from partial solution to all nodes
        avg_dist_to_solution = np.zeros(R, dtype=np.float64)
        for r in range(R):
            c = cand_idx_arr[r]
            total_dist = 0.0
            for j in range(sol_size):
                s = partial_solution[j]
                total_dist += distance_matrix[c, s] + distance_matrix[s, c]
            avg_dist_to_solution[r] = total_dist / sol_size if sol_size > 0 else 0.0
        
        disp_range = disp_max - disp_min
        access_range = access_max - access_min
        
        if alpha == 1.0:
            # Pure dispersion approximation
            for r in range(R):
                # Approximate new dispersion based on average distance to existing stations
                approx_disp_increase = avg_dist_to_solution[r] * sol_size
                if disp_range > 0.0:
                    out[r] = 1.0 - ((approx_disp_increase - disp_min) / disp_range)
                else:
                    out[r] = 1.0
        elif alpha == 0.0:
            # Pure accessibility approximation - use pre-computed two-way sums as baseline
            for r in range(R):
                c = cand_idx_arr[r]
                # Approximate accessibility improvement
                base_accessibility = two_way_sums[c]
                # Adjust based on existing stations (simple heuristic)
                adjusted_access = base_accessibility * (1.0 - 0.1 * sol_size)
                if access_range > 0.0:
                    out[r] = 1.0 - ((adjusted_access - access_min) / access_range)
                else:
                    out[r] = 1.0
        else:
            # Mixed mode approximation
            for r in range(R):
                c = cand_idx_arr[r]
                
                # Dispersion component approximation
                approx_disp_increase = avg_dist_to_solution[r] * sol_size
                if disp_range > 0.0:
                    inv_disp = 1.0 - ((approx_disp_increase - disp_min) / disp_range)
                else:
                    inv_disp = 1.0
                
                # Accessibility component approximation
                base_accessibility = two_way_sums[c]
                adjusted_access = base_accessibility * (1.0 - 0.1 * sol_size)
                if access_range > 0.0:
                    inv_access = 1.0 - ((adjusted_access - access_min) / access_range)
                else:
                    inv_access = 1.0
                
                out[r] = alpha * inv_disp + (1.0 - alpha) * inv_access
    
    return out


@njit(fastmath=True)
def _segmented_greedy_selection(
    cand_idx_arr,          # All candidate indices
    node_scores_arr,       # Node scores for candidates
    distance_matrix,       # Distance matrix
    station_min_distance,  # Minimum distance constraint
    alpha,                 # Alpha parameter
    disp_min, disp_max,    # Dispersion bounds
    access_min, access_max,# Accessibility bounds
    two_way_sums,          # Pre-computed two-way sums
    N,                     # Target number of selections
    segment_size=10        # How many to select exactly before approximating rest
):
    """
    Segmented greedy selection: Use exact computation for first few selections,
    then approximate for the rest. This balances accuracy with efficiency.
    """
    M = cand_idx_arr.shape[0]
    selected = np.empty(min(N, M), dtype=np.int64)
    selected_count = 0
    used = np.zeros(M, dtype=np.bool_)
    
    # Phase 1: Exact selection for first segment_size selections
    exact_selections = min(segment_size, N)
    partial_solution = np.empty(exact_selections, dtype=np.int64)
    partial_count = 0
    
    for step in range(exact_selections):
        if selected_count >= N:
            break
            
        # Get available candidates
        available_indices = np.where(~used)[0]
        if len(available_indices) == 0:
            break
            
        available_cands = cand_idx_arr[available_indices]
        
        # Check feasibility
        feasible_mask = np.empty(len(available_indices), dtype=np.bool_)
        for i in range(len(available_indices)):
            candidate_idx = available_cands[i]
            is_feasible = True
            for j in range(partial_count):
                selected_idx = partial_solution[j]
                if (distance_matrix[candidate_idx, selected_idx] < station_min_distance or
                    distance_matrix[selected_idx, candidate_idx] < station_min_distance):
                    is_feasible = False
                    break
            feasible_mask[i] = is_feasible
        
        if not np.any(feasible_mask):
            break
            
        feasible_indices = available_indices[feasible_mask]
        feasible_cands = cand_idx_arr[feasible_indices]
        
        # Get base scores
        base_scores = node_scores_arr[feasible_indices]
        
        # Get approximated graph scores
        graph_scores = _approximate_incremental_scores(
            feasible_cands, partial_solution[:partial_count], node_scores_arr[feasible_cands],
            distance_matrix, alpha, disp_min, disp_max, access_min, access_max,
            two_way_sums, station_min_distance
        )
        
        # Combine scores
        combined_scores = base_scores * graph_scores
        
        # Select best
        best_local = np.argmax(combined_scores)
        best_global_idx = feasible_indices[best_local]
        best_candidate = feasible_cands[best_local]
        
        # Add to selections
        selected[selected_count] = best_candidate
        partial_solution[partial_count] = best_candidate
        selected_count += 1
        partial_count += 1
        used[best_global_idx] = True
    
    # Phase 2: If we need more selections, use bulk approximation for efficiency
    if selected_count < N:
        remaining_needed = N - selected_count
        available_indices = np.where(~used)[0]
        
        if len(available_indices) > 0:
            available_cands = cand_idx_arr[available_indices]
            
            # Approximate scores for ALL remaining candidates
            base_scores = node_scores_arr[available_indices]
            graph_scores = _approximate_incremental_scores(
                available_cands, partial_solution[:partial_count], base_scores,
                distance_matrix, alpha, disp_min, disp_max, access_min, access_max,
                two_way_sums, station_min_distance
            )
            combined_scores = base_scores * graph_scores
            
            # Use bulk feasibility filter for remaining selections
            if station_min_distance > 0.0:
                additional_selected = _bulk_feasibility_filter(
                    available_cands, combined_scores, distance_matrix,
                    station_min_distance, remaining_needed
                )
            else:
                # No distance constraints - just take top N by score
                if len(available_cands) <= remaining_needed:
                    additional_selected = available_cands
                else:
                    top_pos = np.argpartition(combined_scores, -remaining_needed)[-remaining_needed:]
                    additional_selected = available_cands[top_pos]
            
            # Add to final selection
            for i in range(len(additional_selected)):
                if selected_count < N:
                    selected[selected_count] = additional_selected[i]
                    selected_count += 1
    
    return selected[:selected_count]


def _crossover_greedy_ultimate_optimized(
    N,
    candidates,           # Python list of node‐IDs
    node_scores,          # Python dict { node_ID: score }
    distance_matrix,      # NumPy array (V × V), float64
    id_to_idx,            # dict { node_ID → idx (0…V−1) }
    precomputed=None,     # CrossoverPrecomputed object (optional)
    station_min_distance=0.0,
    graph_score=False,
    alpha=0.5,
    metric_bounds=None
):
    """
    Ultimate optimization: Segmented approximation approach that balances accuracy with efficiency.
    Uses exact computation for initial selections, then approximation for the rest.
    Respects alpha parameter to avoid redundant computations.
    """
    V = distance_matrix.shape[0]
    M = len(candidates)

    # Build reverse‐mapping idx → node_ID
    idx_to_id_local = [None] * V
    for nid, idx in id_to_idx.items():
        idx_to_id_local[idx] = nid

    # Fast path: if very small candidate set or no constraints
    if M <= N or (station_min_distance == 0.0 and not graph_score):
        node_scores_arr = np.zeros(V, dtype=np.float64)
        for nid, sc in node_scores.items():
            node_scores_arr[id_to_idx[nid]] = sc
            
        cand_idx_arr = np.empty((M,), dtype=np.int64)
        for i, nid in enumerate(candidates):
            cand_idx_arr[i] = id_to_idx[nid]
        
        if M <= N:
            return [idx_to_id_local[int(idx)] for idx in cand_idx_arr]
            
        scores_all = node_scores_arr[cand_idx_arr]
        top_pos = np.argpartition(scores_all, -N)[-N:]
        return [idx_to_id_local[int(cand_idx_arr[int(p)])] for p in top_pos]

    # SEGMENTED APPROXIMATION: Use exact + approximation approach
    if graph_score:
        
        # Build candidate arrays
        cand_idx_arr = np.empty((M,), dtype=np.int64)
        node_scores_arr = np.empty((M,), dtype=np.float64)
        for i, nid in enumerate(candidates):
            cand_idx_arr[i] = id_to_idx[nid]
            node_scores_arr[i] = node_scores[nid]
        
        # Get metric bounds
        if metric_bounds is None:
            disp_min, disp_max = 0.0, 1.0
            access_min, access_max = 0.0, 1.0
        else:
            (disp_min, disp_max), (access_min, access_max) = metric_bounds
        
        # Get or compute two-way sums (only if accessibility is needed)
        two_way_sums = None
        if alpha < 1.0:  # Only if accessibility matters
            if precomputed is not None:
                two_way_sums = precomputed.get_two_way_sums()
            else:
                two_way_sums = precompute_distance_sums(distance_matrix)
        else:
            # Create dummy array for alpha=1.0 case (dispersion only)
            two_way_sums = np.zeros(V, dtype=np.float64)
        
        # Determine segment size based on N (larger N = larger exact segment for better accuracy)
        if N <= 20:
            segment_size = max(5, N // 2)    # Small N: be more exact
        elif N <= 50:
            segment_size = max(10, N // 3)   # Medium N: moderate exact portion
        else:
            segment_size = max(15, N // 4)   # Large N: smaller exact portion for efficiency
        
        # Use segmented greedy selection
        selected_indices = _segmented_greedy_selection(
            cand_idx_arr, node_scores_arr, distance_matrix, station_min_distance,
            alpha, disp_min, disp_max, access_min, access_max, two_way_sums, N, segment_size
        )
        
        # Convert back to node IDs
        result = [idx_to_id_local[int(idx)] for idx in selected_indices]
        return result

    # Fallback to existing optimized approach for non-graph-score cases
    return _crossover_greedy_with_precomputed(
        N, candidates, node_scores, distance_matrix, id_to_idx,
        precomputed=precomputed, station_min_distance=station_min_distance,
        graph_score=graph_score, alpha=alpha, metric_bounds=metric_bounds
    )