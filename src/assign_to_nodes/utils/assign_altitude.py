"""
This module fetches altitude data for each node in the graph using the OpenTopoData API.

Author: Jordi Grau Escolano
"""

import os
import sys
import time 
import requests 
import matplotlib.pyplot as plt  # type:ignore
import pandas as pd
import osmnx as ox

notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '../../..'))
sys.path.append(project_root)

import src.data_loader as dl
from paths import *

def get_altitude_batch(coordinates, batch_size=100, api_url=\
                       "https://api.opentopodata.org/v1/eudem25m"):
    """
    Get the altitude for a list of coordinates using the OpenTopoData API in batches.
    Edudem25m: 25 metre resolution and a vertical accuracy of ± 7m RMSE.

    Args:
        coordinates (list of tuple): A list of (latitude, longitude) pairs.
        batch_size (int): The number of coordinates to process per API request.
        api_url (str): The API endpoint for retrieving altitude data.

    Returns:
        list: A list of altitude values corresponding to the input coordinates.
    """
    results = []
    for i in range(0, len(coordinates), batch_size):
        batch = coordinates[i:i + batch_size]
        query = "|".join(f"{lat},{lon}" for lat, lon in batch)
        response = requests.get(f"{api_url}?locations={query}")
        if response.status_code == 200:
            data = response.json()
            results.extend([res.get("elevation", None) for res in data.get(\
                "results", [])])
        else:
            results.extend([None] * len(batch))
        time.sleep(1)
    return results


def plot_graph_altitude(nodes_alt, plot_file):
    
    bcn_boundary = dl.load_bcn_boundary()
    
    fig, ax = plt.subplots(figsize=(20,20))
    bcn_boundary.boundary.plot(ax=ax)
    nodes_alt.plot(
        column='altitude',
        cmap='Blues',
        ax=ax,
        legend=True,
        markersize=5)
    
    ax.axis('off')
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


def get_nodes_altitude(G, root='.'):
    """
    Fetches altitude data for each node in the graph using the OpenTopoData API.

    Args:
        G (nx.MultiDiGraph): NetworkX graph to add altitude attributes to.
        root (str, optional): Root directory path prefix for saving visualization.
            Defaults to empty string.

    Returns:
        nx.MultiDiGraph: Graph with altitude data integrated into node attributes.
    """
    # Convert graph to GeoDataFrame temporarily to get coordinates
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    nodes_4326 = nodes_gdf.to_crs(epsg=4326)
    
    # Extract coordinates (latitude, longitude)
    coordinates = [(point.y, point.x) for point in nodes_4326.geometry]

    # Fetch altitude data for the node points
    altitudes = get_altitude_batch(coordinates)
    
    # Ensure response length matches expected
    if len(altitudes) != len(coordinates):
        print(f"Warning: Expected {len(coordinates)} altitudes, but got {len(altitudes)}.")
        altitudes = [alt if alt is not None else -9999 for alt in altitudes]  # Replace None with default

    # Add altitude attributes to graph nodes
    for node_id, altitude in zip(G.nodes(), altitudes):
        G.nodes[node_id]['altitude'] = altitude

    # Retry for missing altitudes
    missing_nodes = [node_id for node_id, alt in zip(G.nodes(), altitudes) if pd.isna(alt)]
    if missing_nodes:
        missing_coordinates = [(G.nodes[node_id]['y'], G.nodes[node_id]['x']) for node_id in missing_nodes]
        retry_altitudes = get_altitude_batch(missing_coordinates)
        for node_id, altitude in zip(missing_nodes, retry_altitudes):
            G.nodes[node_id]['altitude'] = altitude

    # Fill remaining NaNs with 0, they correspond to the points close to the sea
    for node_id in G.nodes():
        if pd.isna(G.nodes[node_id].get('altitude')):
            G.nodes[node_id]['altitude'] = 0.0
        else:
            G.nodes[node_id]['altitude'] = round(float(G.nodes[node_id]['altitude']), 2)

    # # Plot
    # plot_file = f'{root}/{VISUALIZATIONS}/graph_altitude.png'
    # if not os.path.exists(plot_file):
    #     # Convert to GeoDataFrame for plotting
    #     nodes_gdf, _ = ox.graph_to_gdfs(G)
    #     plot_graph_altitude(nodes_gdf, plot_file)

    return G