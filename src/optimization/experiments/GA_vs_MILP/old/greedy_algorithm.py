import src.optimization.helper_optimization as oh


def greedy_algorithm(df, column_name, N_STATIONS, min_distance_matrix=None, distance_constraint=False, id_to_idx=None, weights=None):
    """
    Finds the nodes with the highest score, optionally considering distance constraints.
    When min_distance_matrix is provided, applies distance constraints between stations.
    
    Args:
        df (pd.DataFrame): DataFrame containing node attributes.
        column_name (str): Name of the column to sort by.
        N_STATIONS (int): Number of nodes to select.
        min_distance_matrix (ndarray, optional): Binary distance matrix (1 if distance >= min_distance, 0 otherwise).
        id_to_idx (dict, optional): Dictionary mapping node IDs to matrix indices.
        weights (dict, optional): Dictionary with variable names as keys and their weights as values.
                                Required if using distance constraints.

    Returns:
        tuple: (selected_nodes, scores)
            - selected_nodes (list): List of selected node IDs
            - scores (list/float): If using distance constraints, list of individual scores;
                                 otherwise, total sum score
    """
    if 'node_id' in df.columns:
        df = df.set_index('node_id')
    # Case 1: No distance constraints (min_distance_matrix is None)
    if distance_constraint == False:
        best_nodes = df.sort_values(by=column_name, ascending=False).head(N_STATIONS)
        return best_nodes.index.tolist(), sum(best_nodes[column_name].values)
    
    # Case 2: With distance constraints
    selected_nodes = []
    remaining_nodes = df.index.tolist()
    all_scores = []

    for _ in range(N_STATIONS):
        if not remaining_nodes:
            print("No more valid nodes available. Stopping early.")
            break

        # Calculate scores for the remaining nodes
        df = oh.sum_and_normalize_all_node_scores(df, weights)

        # Filter valid candidates based on the minimum distance constraint
        valid_candidates = []
        for candidate in remaining_nodes:
            c_idx = id_to_idx[candidate]
            if all(
                min_distance_matrix[c_idx, id_to_idx[selected]] == 1
                for selected in selected_nodes):
                valid_candidates.append(candidate)

        if not valid_candidates:
            print("No valid candidates satisfying the distance constraint. Stopping early.")
            break

        # Select the node with the highest score
        best_node = df[df.index.isin(valid_candidates)].sort_values(
            by='norm_score', ascending=False).iloc[0].name
        selected_nodes.append(int(best_node))

        # Update total score
        all_scores.append(df[df.index == best_node]['norm_score'].values[0])

        # Remove the selected node from the remaining nodes
        remaining_nodes.remove(best_node)

    # Validate the solution
    if len(selected_nodes) != N_STATIONS:
        raise ValueError(
            f"Could only select {len(selected_nodes)} valid nodes out of {N_STATIONS} requested. "
            "Try reducing N_STATIONS or the minimum distance constraint."
        )

    return selected_nodes, sum(all_scores)