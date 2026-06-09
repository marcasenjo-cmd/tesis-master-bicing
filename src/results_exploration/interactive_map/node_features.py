"""
Node features module for visualizing nodes on the interactive map.
"""

import folium

def add_nodes_to_map(m, solution, df_weighted_osm, df_raw, all_distances, node_id_to_idx, colormap, weights):
    """
    Add nodes to the map.
    
    Args:
        m (folium.Map): The map to add nodes to
        solution (list): List of node IDs in the solution
        df_weighted_osm (GeoDataFrame): DataFrame with node data
        df_raw (DataFrame): DataFrame with raw node attributes
        all_distances (dict): Dictionary of distances between nodes
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        colormap: Colormap for node coloring
        weights (dict): Dictionary of weights for node attributes
        
    Returns:
        folium.FeatureGroup: Feature group containing all nodes
    """
    nodes_group = folium.FeatureGroup(name='Nodes')
    
    for node_id in solution:
        try:
            node = df_weighted_osm.loc[node_id]
            connected_nodes = get_connected_nodes_info(node_id, solution, all_distances, node_id_to_idx)
            popup_html = create_node_popup(node_id, node, df_weighted_osm, df_raw, connected_nodes, node_id_to_idx, weights)
            
            add_node_to_map(nodes_group, node, popup_html, colormap, node_id_to_idx)
            
        except Exception as e:
            print(f"Error adding node {node_id} to map: {e}")
            continue
    
    nodes_group.add_to(m)
    return nodes_group

def add_node_to_map(nodes_group, node, popup_html, colormap, node_id_to_idx):
    """
    Add a single node to the map.
    
    Args:
        nodes_group (folium.FeatureGroup): The feature group to add the node to
        node (Series): Node data
        popup_html (str): HTML content for the popup
        colormap: Colormap for node coloring
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
    """
    score = node.get('score', 0)
    color = colormap(score) if colormap else '#3388ff'
    
    # Create a circle marker for the node
    circle = folium.CircleMarker(
        location=[node.geometry.y, node.geometry.x],
        radius=10,
        weight=2,
        color='#000000',
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(popup_html, max_width=450),
        tooltip=f"Node {node_id_to_idx[node.name]:,}",
        class_name=f"node-marker",
        attr={"data-node-id": str(node.name)}
    )
    
    circle.add_to(nodes_group)

def get_connected_nodes_info(node_id, solution, all_distances, node_id_to_idx):
    """
    Get information about nodes connected to a given node.
    
    Args:
        node_id (str/int): ID of the node
        solution (list): List of all node IDs in the solution
        all_distances (dict): Dictionary of distances between nodes
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        
    Returns:
        list: List of strings with connection information
    """
    connected_nodes = []
    for other_id in solution:
        if other_id == node_id:
            continue
            
        try:
            # Try both directions in case of missing keys
            if (node_id, other_id) in all_distances and (other_id, node_id) in all_distances:
                dist_to = all_distances[(node_id, other_id)]
                dist_from = all_distances[(other_id, node_id)]
            else:
                # One or both directions might be missing
                dist_to = all_distances.get((node_id, other_id), all_distances.get((other_id, node_id), 0))
                dist_from = all_distances.get((other_id, node_id), all_distances.get((node_id, other_id), 0))
                
            connected_nodes.append(
                f'{node_id_to_idx[other_id]:,}: {dist_to:,.0f}m // {dist_from:,.0f}m'
            )
        except Exception as e:
            print(f"Error getting connection info between nodes {node_id} and {other_id}: {e}")
            continue
            
    return connected_nodes

def create_node_popup(node_id, node, df_weighted_osm, df_raw, connected_nodes, node_id_to_idx, weights):
    """
    Create popup HTML for a node.
    
    Args:
        node_id (str/int): ID of the node
        node (Series): Node data row from DataFrame
        df_weighted_osm (GeoDataFrame): DataFrame with node weighted data
        df_raw (DataFrame): DataFrame with raw node attributes
        connected_nodes (list): List of strings with connection information
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        weights (dict): Dictionary of weights for node attributes
        
    Returns:
        str: HTML content for the popup
    """
    try:
        # Get the weights that are not zero
        non_zero_weights = {k: v for k, v in weights.items() if v != 0}
        
        # Get the node's values from df_raw
        node_raw_values = df_raw[df_raw['node_id'] == node_id].iloc[0]
        node_weighted_values = df_weighted_osm[df_weighted_osm.index == node_id].iloc[0]
        
        # Create weight information string
        weight_info = []
        for var, weight in non_zero_weights.items():
            if var in node_raw_values and var in node_weighted_values:
                raw_value = node_raw_values[var]
                normalized_value = node_weighted_values[var]
                weight_info.append(f"{var} ({weight:.3f}): {raw_value:,.1f} -> {normalized_value:.1f}")
        
        return f'''
        <div style="font-family: Arial, sans-serif; font-size: 11px;">
            <h4 style="font-size: 13px; margin: 5px 0;">Node {node_id_to_idx[node_id]:,} (Score: {node.norm_score:.3f})</h4>
            <h5 style="font-size: 12px; margin: 5px 0;">Weight Values:</h5>
            <ul style="margin: 3px 0; padding-left: 15px;">
                {''.join([f'<li style="margin: 2px 0;">{info}</li>' for info in weight_info])}
            </ul>
            <h5 style="font-size: 12px; margin: 5px 0;">Connected to/from:</h5>
            <ul style="margin: 3px 0; padding-left: 15px;">
                {''.join([f'<li style="margin: 2px 0;">{conn}</li>' for conn in connected_nodes])}
            </ul>
            <div style="margin-top: 5px;">
                <button style="font-size: 10px; padding: 2px 4px;" onclick="toggleNodePaths({node_id_to_idx[node_id]})">Toggle All</button>
                <button style="font-size: 10px; padding: 2px 4px;" onclick="toggleIncomingPaths({node_id_to_idx[node_id]})">Incoming</button>
                <button style="font-size: 10px; padding: 2px 4px;" onclick="toggleOutgoingPaths({node_id_to_idx[node_id]})">Outgoing</button>
            </div>
        </div>
        '''
    except Exception as e:
        print(f"Error creating popup for node {node_id}: {e}")
        return f'<div style="font-size: 11px;">Error creating popup for node {node_id}</div>'