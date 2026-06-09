"""
This module loads or downloads a bike graph for a given location and prepares the nodes 
as a GeoDataFrame. It also fetches and integrates altitude data for all nodes and computes 
altitude differences for each directed edge.

Author: Jordi Grau Escolano
"""

import sys
import os
from pathlib import Path
import pickle
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

file_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(file_dir, '../..'))

from paths import *
from src.assign_to_nodes.utils.assign_altitude import get_nodes_altitude
from src.assign_to_nodes.utils.fix_directed_graph import fix_directed_graph


def add_altitude_and_elevation_attributes(G, root):
    """Add altitude and elevation attributes to the graph."""
    G = get_nodes_altitude(G, root)
    for u, v, key, data in G.edges(data=True, keys=True):
        G.edges[u, v, key]['elevation'] = round(G.nodes[v]['altitude'] - G.nodes[u]['altitude'], 2)
    return G


def download_fix_altitude_and_save_graph(location, graph_file, crs, root, graph_path):
    """Helper function to download a graph, add edges on nodes with only one directed edge,
    project the graph and save in both pickle (protocol 5) and GraphML formats. 
    Also computes the altitude and elevation attributes for each edge.

    Args:
        location (str): The location to download the graph from
        graph_file (str): The path to save the graph with protocol 5
        crs (int): The CRS of the graph
        root (str): The root directory of the project
        graph_path (str): The path to save the graph

    Returns:
        nx.MultiDiGraph: The fixed graph
    """
    # Download graph first
    G = ox.graph_from_place(location, network_type='bike', simplify=True, retain_all=False)
    G = ox.project_graph(G)
    
    # Create temporary nodes_gdf for fix_directed_graph
    temp_nodes_gdf, _ = ox.graph_to_gdfs(G)
    temp_nodes_gdf = temp_nodes_gdf.to_crs(crs)
    
    # Fix the graph
    G = fix_directed_graph(G, temp_nodes_gdf, root)

    # Convert edge weights to float
    for u, v, key, data in G.edges(data=True, keys=True):
        if 'weight' in data:
            G.edges[u, v, key]['weight'] = float(data['weight'])
    
    # Save the graph without elevation first (regular version)
    base_name = f"bike_graph" if 'Barcelona' in location else f"bike_graph_{location}"
    regular_graph_file = f"{root}/{graph_path}/{base_name}.pickle"
    graphml_file = f"{root}/{graph_path}/{base_name}.graphml"
    
    if graph_path:
        Path(f"{root}{graph_path}").mkdir(parents=True, exist_ok=True)
        
        # Save as pickle with protocol 5
        try:
            print(f"Saving graph to {graph_file} with protocol 5")
            with open(graph_file, "wb") as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving with protocol 5: {e}")
        
        # Save as GraphML
        graphml_file = f"{root}{graph_path}/bike_graph.graphml"
        try:
            print(f"Saving graph as GraphML to {graphml_file}")
            ox.save_graphml(G, graphml_file)
        except Exception as e:
            print(f"Error saving graph as GraphML: {e}")
    
    # Get altitude and elevation data and integrate it into the graph
    G = add_altitude_and_elevation_attributes(G, root)
    
    # Save the elevation version with '_elevation' suffix
    elevation_graph_file = f"{root}/{graph_path}/{base_name}_elevation.pickle"
    elevation_graphml_file = f"{root}/{graph_path}/{base_name}_elevation.graphml"
    
    if graph_path:        
        # Save elevation version as pickle with protocol 5
        try:
            print(f"Saving elevation graph to {elevation_graph_file} with protocol 5")
            with open(elevation_graph_file, "wb") as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving elevation graph with protocol 5: {e}")
        
        # Save elevation version as GraphML
        try:
            print(f"Saving elevation graph as GraphML to {elevation_graphml_file}")
            ox.save_graphml(G, elevation_graphml_file)
        except Exception as e:
            print(f"Error saving elevation graph as GraphML: {e}")
            
    return G


def load_or_download_graph(location: str, graph_path: str, crs: int, root: str, use_elevation: bool = False):
    """
    Graph was created on 2025-03-19.
    Loads or downloads a bike graph for a given location and prepares the nodes 
    as a GeoDataFrame. Uses the following priority list for loading:
    1. bike_graph.pickle (protocol 5) or bike_graph_elevation.pickle if use_elevation=True
    2. bike_graph.graphml or bike_graph_elevation.graphml if use_elevation=True
    3. bike_graph_notebook_processed_2025-03-19.graphml
    
    Also fetches and integrates altitude data for all nodes and computes altitude
    differences for each directed edge.

    Args:
        location (str): The location to download the graph from
        graph_path (str): Path to the graph directory
        crs (int): The CRS of the graph
        root (str): The root directory of the project
        use_elevation (bool): Whether to load a graph with elevation data

    Returns:
        G (nx.MultiDiGraph): The graph
        nodes_gdf (gpd.GeoDataFrame): The nodes as a GeoDataFrame
    """
    base_name = f"bike_graph" if 'Barcelona' in location else f"bike_graph_{location}"
    if use_elevation:
        file_name = f"{base_name}_elevation"
    else:
        file_name = base_name
    
    graph_file = f"{root}/{graph_path}/{file_name}.pickle"
    graphml_file = f"{root}/{graph_path}/{file_name}.graphml"
    plot_file = f"{root}/{VISUALIZATIONS}/{file_name}.png"
    
    # Try to load an existing graph file
    G = None
    
    # Priority 1: Try protocol 5 pickle
    if os.path.exists(graph_file):
        try:
            with open(graph_file, "rb") as f:
                G = pickle.load(f)
            print(f"\tSuccessfully loaded graph {file_name} with protocol 5")
        except Exception as e:
            print(f"\tError loading with protocol 5: {e}")
    
    # Priority 2: Try GraphML
    if G is None and os.path.exists(graphml_file):
        try:
            G = ox.load_graphml(graphml_file)
            # Convert node IDs to integers if they're strings
            G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
            print(f"\tSuccessfully loaded graph from GraphML")
        except Exception as e:
            print(f"\tError loading from GraphML: {e}")

    
    # If still no graph, try to download a new one
    if G is None:
        print(f"\tNo existing graph {graph_file} could be loaded in {root}/{graph_path}/" 
              f"\n\tThere is : {os.listdir(f'{root}/{graph_path}')}."
              f"\n\tDownloading bike graph for location: {location}")
        G = download_fix_altitude_and_save_graph(location, graph_file, crs, root, graph_path)

    # Convert graph nodes to a GeoDataFrame
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    nodes_gdf = nodes_gdf.to_crs(crs)
    nodes_gdf = nodes_gdf[['geometry']]
    nodes_gdf.index.name = 'node_id'
    nodes_gdf.index = nodes_gdf.index.astype(int)

    # Plot graph with boundary
    if not os.path.exists(plot_file):
        # Download OSM bcn boundary
        bcn_boundary = ox.geocode_to_gdf(location)
        bcn_boundary = bcn_boundary.to_crs(crs)
        
        fig, ax = plt.subplots(figsize=(20, 20))
        bcn_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf.plot(ax=ax, linewidth=0.5, color='gray', label='Edges')
        nodes_gdf.plot(ax=ax, markersize=0.5, label='Nodes')
        ax.axis('off')
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
    
    return G, nodes_gdf


def save_node_attributes(node_attributes: pd.DataFrame, attr_file: str):
    """
    Saves the node attributes to a CSV file.

    Args:
        node_attributes (pd.DataFrame): DataFrame containing node attributes.
        attr_file (str): Path where the file should be saved.
    """
    if not node_attributes.empty:
        node_attributes.to_csv(attr_file)
        print(f"Node attributes saved to {attr_file}")
    else:
        print("No node attributes to save.")


def load_node_attributes(attr_file: str, nodes_index) -> pd.DataFrame:
    """
    Loads the node attributes from a CSV file if available.

    Args:
        attr_file (str): Path to the stored file.
        nodes_index (Index or list): The node IDs to ensure alignment.

    Returns:
        pd.DataFrame: The loaded node attributes DataFrame.
    """
    if os.path.exists(f"{attr_file}"):
        print(f"Loading existing node attributes from {attr_file}")
        return pd.read_csv(f"{attr_file}", index_col=0)
    else:
        dir_path = os.path.dirname(attr_file)
        print(f"\tNo existing attributes found in {attr_file}."
              f"\n\tThere is these files: {os.listdir(dir_path) if os.path.exists(dir_path) else 'Directory does not exist'}."
              f"\n\tInitializing an empty DataFrame.")
        return pd.DataFrame({'node_id': nodes_index}).set_index('node_id')
