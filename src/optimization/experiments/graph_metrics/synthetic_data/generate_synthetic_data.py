import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

def assign_uniform(G):
    """Uniform random in [0,1]."""
    return {u: np.random.rand() for u in G.nodes()}

def assign_dist_center(G, center=(429030.279, 4583487.198)):
    """normalized Euclidean distance from center."""
    cx, cy = center
    # Get bounds of graph nodes to normalize distances
    xs = [data['x'] for _, data in G.nodes(data=True)]
    ys = [data['y'] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Maximum possible distance from center to any corner
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    max_d = max(np.hypot(x - cx, y - cy) for x, y in corners)
    
    vals = {}
    for u, data in G.nodes(data=True):
        x, y = data['x'], data['y']  # Access x,y directly instead of using pos
        d = np.hypot(x - cx, y - cy)
        vals[u] = 1 - min(d / max_d, 1)
    return vals

def assign_inv_dist_center(G, center=(429030.279, 4583487.198)):
    """inverse normalized Euclidean distance from center."""
    cx, cy = center
    # Get bounds of graph nodes to normalize distances
    xs = [data['x'] for _, data in G.nodes(data=True)]
    ys = [data['y'] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Maximum possible distance from center to any corner
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    max_d = max(np.hypot(x - cx, y - cy) for x, y in corners)
    
    vals = {}
    for u, data in G.nodes(data=True):
        x, y = data['x'], data['y']  # Access x,y directly instead of using pos
        d = np.hypot(x - cx, y - cy)
        vals[u] = min(d / max_d, 1)  # Removed the 1- to invert the values
    return vals
    

def assign_community_plateaus(G):
    """Assign a plateau value per community in [0,1]."""
    comms = list(greedy_modularity_communities(G))
    if len(comms) == 1:
        return {u: 0.5 for u in G.nodes()}
    vals = {}
    for i, comm in enumerate(comms):
        v = i / (len(comms) - 1)
        for u in comm:
            vals[u] = v
    return vals

# ------------------------------------------------------------------------------

def apply_strategies(nodes_gdf, G):
    """
    Apply the named strategy to G and nodes_gdf,
    storing the result under attribute/column 'demand'.
    """
    strategies = {
        'uniform': assign_uniform,
        'distance_center': assign_dist_center,
        'inv_distance_center': assign_inv_dist_center,
        'community': assign_community_plateaus,
    }

    # Validate that all nodes in G are in nodes_gdf and vice versa
    g_nodes = set(G.nodes())
    df_nodes = set(nodes_gdf['node_id'])
    if g_nodes != df_nodes:
        missing_in_df = g_nodes - df_nodes
        missing_in_g = df_nodes - g_nodes
        if missing_in_df:
            print(f"Nodes in graph but not in DataFrame: {len(missing_in_df)}")
        if missing_in_g:
            print(f"Nodes in DataFrame but not in graph: {len(missing_in_g)}")

    for strategy_name in strategies.keys():
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'")
        # compute the new values
        values = strategies[strategy_name](G)
        
        # Validate that all nodes got values
        if set(values.keys()) != g_nodes:
            missing_values = g_nodes - set(values.keys())
            raise ValueError(f"Strategy {strategy_name} did not assign values to all nodes: {len(missing_values)}")
            
        nx.set_node_attributes(G, values, name=strategy_name)
        # Map values using node_id instead of index
        nodes_gdf[strategy_name] = nodes_gdf['node_id'].map(values)
        
        # Verify no NaN values were created
        if nodes_gdf[strategy_name].isna().any():
            nan_nodes = nodes_gdf[nodes_gdf[strategy_name].isna()]['node_id'].tolist()
            raise ValueError(f"NaN values created for strategy {strategy_name} at nodes: {nan_nodes}")
            
    return nodes_gdf, G
