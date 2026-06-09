import numpy as np  # type: ignore
import numba as nb  # type: ignore

np.random.seed(1)

@nb.njit(cache=True)
def _init_rng_state():
    """Initialize random number generator state"""
    return np.random.randint(0, 2**31, size=4)

@nb.njit(cache=True)
def _random_int(rng_state, low, high):
    """Fast random integer generator with internal state"""
    # Simple xorshift algorithm
    x = rng_state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    rng_state[0] = x
    return low + (x % (high - low))

@nb.njit(cache=True)
def _shuffle_array(arr, length, rng_state):
    """Shuffle the first 'length' elements of an array in-place"""
    for i in range(length):
        j = _random_int(rng_state, 0, length)
        arr[i], arr[j] = arr[j], arr[i]

@nb.njit(cache=True)
def _find_compatible_candidates_vectorized(solution, positions_to_mutate, 
                                           n_positions, distance_matrix, min_station_distance):
    """
    Find compatible candidates that maintain minimum distance with all unchanged nodes.
    
    Args:
        solution: Current solution (indices)
        positions_to_mutate: Positions that will be mutated
        n_positions: Number of positions to mutate
        distance_matrix: Distance matrix in kilometers
        min_station_distance: Minimum required distance between stations
        
    Returns:
        Boolean mask of compatible candidates
    """
    solution_length = len(solution)
    n_nodes = distance_matrix.shape[0]
    
    # Pre-allocate arrays
    compatible_candidates = np.ones(n_nodes, dtype=np.bool_)
    unchanged_mask = np.ones(solution_length, dtype=np.bool_)
    
    # Mark positions that will be mutated
    for i in range(n_positions):
        unchanged_mask[positions_to_mutate[i]] = False
    
    # Process all unchanged positions
    for i in range(solution_length):
        if unchanged_mask[i]:
            node = solution[i]
            # Check distances in both directions (for directed graph)
            compatible_candidates &= (distance_matrix[:, node] >= min_station_distance) # To node
            compatible_candidates &= (distance_matrix[node, :] >= min_station_distance) # From node
    
    # Exclude nodes already in solution
    for node in solution:
        compatible_candidates[node] = False
        
    return compatible_candidates

@nb.njit(cache=True)
def _mutate_solution_vectorized(solution, mutation_rate, distance_matrix, 
                                min_station_distance, max_replacements, required_nodes_indices):
    """
    Mutate a solution while respecting distance constraints.
    
    Args:
        solution: Current solution (indices)
        mutation_rate: Probability of mutation per node
        distance_matrix: Distance matrix in kilometers
        min_station_distance: Minimum required distance between stations
        max_replacements: Maximum number of nodes to replace
        required_nodes_indices: Array of required node indices that must be in the solution
        
    Returns:
        Mutated solution (indices)
    """
    
    solution_length = len(solution)
    n_nodes = distance_matrix.shape[0]
    
    # Pre-allocate arrays
    result = solution.copy()
    used_nodes = np.zeros(n_nodes, dtype=np.bool_)
    
    # Separate required and non-required nodes
    n_required = len(required_nodes_indices) if required_nodes_indices is not None else 0
    non_required_nodes = np.zeros(solution_length - n_required, dtype=np.int32)
    non_required_idx = 0
    
    # Mark used nodes and separate required from non-required
    for node in solution:
        used_nodes[node] = True
        if required_nodes_indices is None or node not in required_nodes_indices:
            non_required_nodes[non_required_idx] = node
            non_required_idx += 1
    
    # Will store which positions we'll change
    positions_to_mutate = np.zeros(len(non_required_nodes), dtype=np.int32)
    
    # Determine mutation positions
    n_mutations = 0  # Counter for how many positions we'll mutate
    rng_state = _init_rng_state() # Initialize random number generator state
    mutation_threshold = int(mutation_rate * 100)
    
    # Pre-generate random positions
    all_positions = np.arange(len(non_required_nodes), dtype=np.int32)
    _shuffle_array(all_positions, len(non_required_nodes), rng_state)
    
    # Count mutations
    for i in range(len(non_required_nodes)):
        if _random_int(rng_state, 0, 100) < mutation_threshold:
            if n_mutations < max_replacements:
                positions_to_mutate[n_mutations] = all_positions[i]
                n_mutations += 1
    
    if n_mutations == 0:
        return result
    
    if min_station_distance > 0:
        # With distance constraint
        # Find compatible candidates
        compatible_candidates = _find_compatible_candidates_vectorized(
            non_required_nodes, positions_to_mutate, n_mutations, distance_matrix, min_station_distance
        )
        mutated_positions = np.zeros(n_mutations, dtype=np.int32)
        n_mutated = 0
        
        # Pre-allocate arrays for valid candidates
        valid_candidates = np.zeros(n_nodes, dtype=np.int32)
        
        for i in range(n_mutations):
            pos = positions_to_mutate[i]
            old_node = non_required_nodes[pos]
            n_valid = 0
            
            # Find valid candidates
            for node in range(n_nodes):
                if not (compatible_candidates[node] and not used_nodes[node]):
                    continue
                
                # Check compatibility with already mutated positions
                if n_mutated > 0:
                    is_compatible = True
                    for j in range(n_mutated):
                        mut_pos = mutated_positions[j]
                        mut_node = non_required_nodes[mut_pos]
                        # Check both directions for directed graph
                        if (distance_matrix[node, mut_node] < min_station_distance or 
                            distance_matrix[mut_node, node] < min_station_distance):
                            is_compatible = False
                            break
                    if not is_compatible:
                        continue
                
                valid_candidates[n_valid] = node
                n_valid += 1
            
            if n_valid > 0:
                idx = _random_int(rng_state, 0, n_valid)
                new_node = valid_candidates[idx]
                non_required_nodes[pos] = new_node
                used_nodes[old_node] = False
                used_nodes[new_node] = True
                mutated_positions[n_mutated] = pos
                n_mutated += 1
    else:
        # Optimize unconstrained case
        unused_nodes = np.zeros(n_nodes, dtype=np.int32)
        for i in range(n_mutations):
            pos = positions_to_mutate[i]
            old_node = non_required_nodes[pos]
            
            # Find unused nodes
            n_unused = 0
            for node in range(n_nodes):
                if not used_nodes[node]:
                    unused_nodes[n_unused] = node
                    n_unused += 1
            
            if n_unused > 0:
                idx = _random_int(rng_state, 0, n_unused)
                new_node = unused_nodes[idx]
                non_required_nodes[pos] = new_node
                used_nodes[old_node] = False
                used_nodes[new_node] = True
    
    # Combine required and non-required nodes in the result
    if required_nodes_indices is not None:
        for i, node in enumerate(required_nodes_indices):
            result[i] = node
        for i, node in enumerate(non_required_nodes):
            result[i + n_required] = node
    else:
        result = non_required_nodes
    
    return result

@nb.njit(cache=True, parallel=True, fastmath=True)
def _mutate_population(solutions_as_indices, mutation_rate, distance_matrix,
                                  min_station_distance, max_replacements, required_nodes_indices):
    """
    Mutate multiple solutions in parallel.
    
    Args:
        solutions_as_indices: Array of solutions (indices)
        mutation_rate: Probability of mutation per node
        distance_matrix: Distance matrix in kilometers
        min_station_distance: Minimum required distance between stations
        max_replacements: Maximum number of nodes to replace
        required_nodes_indices: Array of required node indices that must be in the solution
        
    Returns:
        Array of mutated solutions
    """
    n_solutions = solutions_as_indices.shape[0]
    mutated_solutions = np.zeros_like(solutions_as_indices)
    
    for i in nb.prange(n_solutions):
        mutated_solutions[i] = _mutate_solution_vectorized(
            solutions_as_indices[i],
            mutation_rate,
            distance_matrix,
            min_station_distance,
            max_replacements,
            required_nodes_indices,
        )
    
    return mutated_solutions


def mutate_population(solutions, mutation_rate, distance_matrix, id_to_idx, idx_to_id,
           max_replacements=None, station_min_distance=0, batch_size=32, required_nodes=None):
    """
    Mutate a list of solutions using optimized vectorized operations.
    
    Args:
        solutions: A list of of solutions (with node IDs)
        mutation_rate: Probability of mutation per node
        distance_matrix: Distance matrix in meters
        id_to_idx: Maps node IDs to matrix indices
        idx_to_id: Maps matrix indices to node IDs
        max_replacements: Maximum number of nodes to replace per solution
        station_min_distance: Minimum required distance between stations in meters
        batch_size: Number of solutions to process in each batch of the population
        required_nodes: List of node IDs that must be included in the solution
        
    Returns:
        List of mutated solutions
    """
    if not solutions:
        return []
        
    n_solutions = len(solutions)
    solution_length = len(solutions[0])
    
    # Convert required nodes to indices
    required_nodes_indices = None
    if required_nodes is not None:
        required_nodes_indices = np.array([id_to_idx[node] for node in required_nodes], dtype=np.int32)
    
    # Process in batches
    result = []
    for i in range(0, n_solutions, batch_size):
        batch = solutions[i:min(i+batch_size, n_solutions)]
        
        # Convert batch to indices
        batch_indices = np.zeros((len(batch), solution_length), dtype=np.int32)
        for j, solution in enumerate(batch):
            for k, node_id in enumerate(solution):
                batch_indices[j, k] = id_to_idx[node_id]
        
        # Apply mutation
        mutated_indices = _mutate_population(
            batch_indices,
            mutation_rate,
            distance_matrix,
            station_min_distance,
            max_replacements if max_replacements is not None else solution_length,
            required_nodes_indices
        )
        
        # Convert back to node IDs
        for j in range(len(batch)):
            mutated_solution = [idx_to_id[idx] for idx in mutated_indices[j]]
            result.append(mutated_solution)
    
    return result