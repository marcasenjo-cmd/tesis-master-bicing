import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *

location = 'Barcelona, Barcelona, Catalunya, España'
bcn_boundary = ox.geocode_to_gdf(location)
G = ox.graph_from_place(location, network_type='bike', simplify=True, retain_all=False)
G = ox.project_graph(G)
G_und = G.to_undirected()

# Assign distances as edge weights.
for u, v, data in G_und.edges(data=True):
    data['weight'] = data['length']

def timestamped_print(message):
    """Prints a message with the current time."""
    print(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def compute_centrality_for_all(graph, output_file):
    """
    Computes closeness, betweenness, degree, and percolation centrality for all nodes in a graph.
    Saves the results to a CSV file.
    
    The degree centrality for a node v is the fraction of nodes it is connected to.
    Closeness centrality of a node u is the reciprocal of the average shortest path
        distance to u over all n-1 reachable nodes.
    Betweenness centrality of a node v is the sum of the fraction of all-pairs 
        shortest paths that pass through v.
    Percolation centrality measures the influence of a node over the flow of information
        in a network.

    Parameters:
    - graph: networkx.Graph or networkx.DiGraph
    - output_file: str, path to save the results
    """

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        existing_centralities = df.columns
    else:
        df = pd.DataFrame()
        existing_centralities = []

    results = {}
    node_ids = []
    timestamped_print("Starting centrality computation")

    if "Closeness" not in existing_centralities:
        timestamped_print("Computing closeness centrality...")
        closeness = nx.closeness_centrality(graph, distance='weight')
        results["Closeness"] = list(closeness.values())
        if node_ids == []:
            node_ids = list(closeness.keys())
    
    if "Betweenness" not in existing_centralities:
        timestamped_print("Computing betweenness centrality...")
        betweenness = nx.betweenness_centrality(graph, weight='weight', normalized=True) # normalized by 2/((n-1)(n-2))
        results["Betweenness"] = list(betweenness.values())
        if node_ids == []:
            node_ids = list(betweenness.keys())
    
    if "Degree" not in existing_centralities:
        timestamped_print("Computing degree centrality...")
        degree = nx.degree_centrality(graph)
        results["Degree"] = list(degree.values())
        if node_ids == []:
            node_ids = list(degree.keys())
        
    if "Percolations" not in existing_centralities:
        timestamped_print("Computing percolations centrality...")
        percolations = nx.percolation_centrality(graph, weight='weight')
        results["Percolations"] = list(percolations.values())
        if node_ids == []:
            node_ids = list(percolations.keys())
    
    if 'Eccentricity' not in existing_centralities:
        timestamped_print("Computing eccentricity...")
        eccentricity = nx.eccentricity(graph, weight='weight')
        results["Eccentricity"] = list(eccentricity.values())
        if node_ids == []:
            node_ids = list(eccentricity.keys())

    # Add the new columns to the existing df
    df['node_id'] = node_ids
    for key, value in results.items():
        df[key] = value

    df.to_csv(output_file, header=True, index=True)
    timestamped_print(f"Centrality metrics for all nodes saved to {output_file}")


compute_centrality_for_all(G_und, f"{PR_NODES}/all_nodes_centrality.csv")