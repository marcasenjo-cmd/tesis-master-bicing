"""
Edge features module for visualizing edges on the interactive map.
"""

from itertools import combinations
import folium
from .graph_analyzer import get_path_coordinates

def add_edges_to_map(m, solution, df_weighted_osm, all_paths, all_distances, colors, node_id_to_idx):
    """
    Add edges between nodes to the map.
    
    Args:
        m (folium.Map): The map to add edges to
        solution (list): List of node IDs
        df_weighted_osm (GeoDataFrame): DataFrame with node data
        all_paths (dict): Dictionary of path lists between nodes
        all_distances (dict): Dictionary of distances between nodes
        colors (dict): Dictionary of colors for node pairs
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        
    Returns:
        folium.FeatureGroup: The group containing all edges
    """
    edges_group = folium.FeatureGroup(name='Edges')
    edges_group.add_to(m)
    
    for node1_id, node2_id in combinations(solution, 2):
        add_edge_pair(
            edges_group, 
            node1_id, 
            node2_id, 
            df_weighted_osm, 
            all_paths, 
            all_distances, 
            colors, 
            node_id_to_idx
        )
    
    return edges_group

def add_edge_pair(edges_group, node1_id, node2_id, df_weighted_osm, all_paths, all_distances, colors, node_id_to_idx):
    """
    Add a pair of directed edges between two nodes.
    
    Args:
        edges_group (folium.FeatureGroup): The feature group to add edges to
        node1_id, node2_id (str/int): IDs of the two nodes
        df_weighted_osm (GeoDataFrame): DataFrame with node data
        all_paths (dict): Dictionary of path lists between nodes
        all_distances (dict): Dictionary of distances between nodes
        colors (dict): Dictionary of colors for node pairs
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
    """
    # Check if paths exist for both directions
    if (node1_id, node2_id) not in all_paths or (node2_id, node1_id) not in all_paths:
        return
    
    # Add forward edge
    add_single_edge(
        edges_group,
        node1_id, 
        node2_id,
        df_weighted_osm,
        all_paths[(node1_id, node2_id)],
        all_distances[(node1_id, node2_id)],
        colors[(node1_id, node2_id)],
        node_id_to_idx,
        is_forward=True
    )
    
    # Add reverse edge
    add_single_edge(
        edges_group,
        node2_id, 
        node1_id,
        df_weighted_osm,
        all_paths[(node2_id, node1_id)],
        all_distances[(node2_id, node1_id)],
        colors[(node2_id, node1_id)],
        node_id_to_idx,
        is_forward=False
    )

def add_single_edge(edges_group, from_id, to_id, df_weighted_osm, path, distance, color, node_id_to_idx, is_forward=True):
    """
    Add a single directed edge to the map.
    
    Args:
        edges_group (folium.FeatureGroup): The feature group to add the edge to
        from_id, to_id (str/int): IDs of source and target nodes
        df_weighted_osm (GeoDataFrame): DataFrame with node data
        path (list): List of node IDs in the path
        distance (float): Distance between nodes
        color (str): Hex color code for the edge
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        is_forward (bool): Whether this is a forward edge (affects styling)
    """
    # Get coordinates for the path
    path_coords = get_path_coordinates(path, df_weighted_osm)
    
    # Create popup HTML
    popup_html = create_edge_popup(from_id, to_id, distance, node_id_to_idx)
    
    # Define CSS class for edge toggling
    class_str = f'edge source_{node_id_to_idx[from_id]} target_{node_id_to_idx[to_id]}'
    
    # Define styling options based on direction
    weight = 1.5 if is_forward else 2
    dash_array = None if is_forward else '8'
    
    # Create the polyline
    folium.PolyLine(
        locations=path_coords,
        color=color,
        weight=weight,
        opacity=1,
        dash_array=dash_array,
        popup=folium.Popup(popup_html, max_width=300),
        class_name=class_str
    ).add_to(edges_group)

def create_edge_popup(from_id, to_id, distance, node_id_to_idx):
    """
    Create popup HTML for an edge.
    
    Args:
        from_id, to_id (str/int): IDs of source and target nodes
        distance (float): Distance between nodes
        node_id_to_idx (dict): Dictionary mapping node IDs to indices
        
    Returns:
        str: HTML content for the popup
    """
    return f'''
    <div style="font-family: Arial, sans-serif; font-size: 11px;">
        <h4 style="font-size: 13px; margin: 5px 0;">Path Information</h4>
        <p style="margin: 3px 0;">From Node {node_id_to_idx[from_id]} to Node {node_id_to_idx[to_id]}: {distance:.1f}m</p>
    </div>
    '''