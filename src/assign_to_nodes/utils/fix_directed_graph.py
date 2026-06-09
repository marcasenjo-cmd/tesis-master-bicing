"""
This module complements a directed graph by making sure that it is a strongly connected graph.

Author: Jordi Grau Escolano
"""

import sys
import os
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Define project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl

def categorize_nodes_by_connectivity(G):
    """
    Categorize nodes based on their connectivity patterns.
    
    Args:
        G (networkx.DiGraph): Input directed graph
        
    Returns:
        dict: Dictionary with lists of nodes in each category:
            - 'no_connections': Nodes with no incoming or outgoing edges
            - 'only_incoming': Nodes with only incoming edges
            - 'only_outgoing': Nodes with only outgoing edges
            - 'both_directions': Nodes with both incoming and outgoing edges
    """
    categories = {
        'no_connections': [],
        'only_incoming': [],
        'only_outgoing': [],
    }

    print(f"There are {len(G.nodes())} nodes in the graph")

    for node in G.nodes():
        has_outgoing = False
        has_incoming = False
        
        # Check outgoing edges
        for successor in G.successors(node):
            has_outgoing = True
            break
            
        # Check incoming edges    
        for predecessor in G.predecessors(node):
            has_incoming = True
            break
            
        # Categorize node based on connectivity
        if not has_incoming and not has_outgoing:
            categories['no_connections'].append(node)
        elif has_incoming and not has_outgoing:
            categories['only_incoming'].append(node)
        elif not has_incoming and has_outgoing:
            categories['only_outgoing'].append(node)

    # print the number of nodes in each category
    print("Number of nodes in each category:")
    for category, nodes in categories.items():
        print(f"{category}: {len(nodes)} ({len(nodes)/len(G.nodes())*100:.2f}%)")
            
    return categories


def check_strong_connectivity(G):
    is_strongly_connected = nx.is_strongly_connected(G)
    if not is_strongly_connected:
        print("Warning: Graph is not strongly connected. Some shortest paths will be infinite.")
        # Optionally, find and report the components
        components = list(nx.strongly_connected_components(G))
        print(f"Number of strongly connected components: {len(components)}")
        print(f"Sizes of components: {[len(c) for c in components]}")
        return components
    else:
        print("The graph is strongly connected.")
        return [list(G.nodes())]


def find_the_closest_node_of_the_bad_nodes(G, bad_nodes, nodes_gdf):
    """
    Find the closest node of the bad nodes in the graph.

    Args:
        G (networkx.DiGraph): Input directed graph
        bad_nodes (dict): Dictionary with lists of nodes in each category:
            - 'only_incoming': Nodes with only incoming edges
            - 'only_outgoing': Nodes with only outgoing edges
        nodes_gdf (geopandas.GeoDataFrame): GeoDataFrame with nodes

    Returns:
        dict: Dictionary with the closest node of the bad nodes
    """
    dict_incoming_closest_node = {}
    dict_outgoing_closest_node = {}

    for list in ['only_incoming', 'only_outgoing']:
        for node in bad_nodes[list]:
            # drop the node from the graph find the next closest node
            G_prov = G.copy()
            G_prov.remove_node(node)
            # find the closest node
            closest_node = ox.distance.nearest_nodes(
                G_prov, nodes_gdf.loc[node]['x'], nodes_gdf.loc[node]['y'])
            if list == 'only_incoming':
                dict_incoming_closest_node[node] = closest_node
            else:
                dict_outgoing_closest_node[node] = closest_node

    return dict_incoming_closest_node, dict_outgoing_closest_node


def find_distance_between_closest_nodes(nodes_gdf, dict_incoming_closest_node, 
                                        dict_outgoing_closest_node):
    """
    Find the distance between the closest nodes of the bad nodes in the graph.
    """
    # create a dataframe with the closest nodes
    df_in = pd.DataFrame.from_dict(
        dict_incoming_closest_node, orient='index', columns=['closest_node'])
    df_out = pd.DataFrame.from_dict(
        dict_outgoing_closest_node, orient='index', columns=['closest_node'])

    df_in['only_edge'] = 'incoming'
    df_out['only_edge'] = 'outgoing'

    # merge the dataframes
    df = pd.concat([df_in, df_out])

    # calculate the distance between the closest nodes
    # Using OSMnx's current approach for distance calculation
    def calculate_distance(row):
        # Get points from geometries
        orig_point = nodes_gdf.loc[row.name]['geometry']
        dest_point = nodes_gdf.loc[row['closest_node']]['geometry']
        
        # Calculate distance between points
        return orig_point.distance(dest_point)
    
    df['distance'] = df.apply(calculate_distance, axis=1)
    
    return df


def create_new_edges_between_closest_nodes(G, df_closest_nodes_distances):
    """
    Create new edges between the closest nodes.
    """
    for _, row in df_closest_nodes_distances.iterrows():
        if row['only_edge'] == 'incoming':
            # Node has only incoming edges, so add an outgoing edge
            G.add_edge(row.name, row['closest_node'], 
                      length=row['distance'],
                      edge_type='new')
        else:  # 'outgoing'
            # Node has only outgoing edges, so add an incoming edge
            G.add_edge(row['closest_node'], row.name, 
                      length=row['distance'],
                      edge_type='new')
    return G


def unify_components(G):
    """
    To unify the graph, first the edges that unite the components are found. Then,
    a edge on the opposite direction is added.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph to unify
        
    Returns:
    --------
    networkx.DiGraph
        A unified graph where all components are strongly connected
    """  
    # Make a copy of the input graph to avoid modifying the original
    unified_G = G.copy()
    
    # Find strongly connected components
    components = list(nx.strongly_connected_components(G))

    print(f"The graph has {len(components)} components.")
    
    if len(components) <= 1:
        # Graph is already strongly connected
        return unified_G
    
    # Find edges that connect different components
    # For each edge, create a map of node -> component index
    node_to_component = {}
    for i, component in enumerate(components):
        for node in component:
            node_to_component[node] = i
    
    # For each edge in the original graph
    for u, v, key in G.edges(keys=True):
        comp_u = node_to_component[u]
        comp_v = node_to_component[v]
        
        # If this edge connects different components
        if comp_u != comp_v:
            # Add the reverse edge with the same attributes, if it doesn't exist
            if not unified_G.has_edge(v, u):
                # Get the edge attributes and ensure they're string keys
                data = G.get_edge_data(u, v, key=key)
                clean_attrs = {}
                
                # Clean up attributes - only keep basic attributes with string keys
                for k, val in data.items():
                    if isinstance(k, str):
                        # Only copy simple attribute types
                        if isinstance(val, (str, int, float, bool)) or val is None:
                            clean_attrs[k] = val
                
                # Add basic attributes for the reverse edge
                clean_attrs['length'] = data.get('length', 1000)  # Default length if not present
                clean_attrs['reversed'] = True  # Mark as a reversed edge
                
                # Add the edge
                unified_G.add_edge(v, u, **clean_attrs)
    
    # Verify the graph is now strongly connected
    if len(list(nx.strongly_connected_components(unified_G))) > 1:
        # If not, connect the components in a chain
        new_components = list(nx.strongly_connected_components(unified_G))
        print(f"The graph is not strongly connected, there are {len(new_components)} components.")

    # Convert all 'length' distance attributes to 'weight'
    for u, v, key in unified_G.edges(keys=True):
        data = unified_G.get_edge_data(u, v, key=key)
        if data.get('length') is not None:
            data['weight'] = data['length']
    
    return unified_G


def plot_bad_nodes(G, bad_nodes, root):
    """
    Plot the bad nodes in the graph.
    """
    plot_name = f"{root}{VISUALIZATIONS}/bike_graph_bad_nodes.png"
    if os.path.exists(plot_name):
        return None

    bcn_boundary = dl.load_bcn_boundary()
    fig, ax = plt.subplots(figsize=(20, 20))
    bcn_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

    # Plot nodes with only incoming edges
    nodes_to_highlight = bad_nodes['only_incoming'] + bad_nodes['only_outgoing']
    node_colors = ['red' if node in bad_nodes['only_incoming'] 
                else 'blue' if node in bad_nodes['only_outgoing']
                else 'gray' for node in G.nodes()]
    node_sizes = [10 if node in nodes_to_highlight else 1 for node in G.nodes()]

    ox.plot_graph(
        G,
        ax=ax,
        show=False,
        close=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='gray', 
        edge_linewidth=1,
        edge_alpha=0.7,
        bgcolor='white'
    )
    
    # North arrow (adjusted parameters, moved away from map edge and centered)
    fontsize = 15
    ax.annotate(
        '', 
        xy=(0.12, 0.25), 
        xytext=(0.12, 0.25 - 0.05 * 0.8), 
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=10, headlength=15),
        ha='center', va='center'
    )
    
    # Scale bar (positioned below arrow, centered with 'N')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate scale bar position - centered with arrow
    x_center = xlim[0] + 0.12 * (xlim[1] - xlim[0])  # Center point
    x_start = x_center - 1000  # 1 km to left of center
    x_end = x_center + 1000    # 1 km to right of center
    y_scale = ylim[0] + 0.18 * (ylim[1] - ylim[0])  # Below arrow
    
    # Draw the scale bar
    ax.plot([x_start, x_end], [y_scale, y_scale], color='black', linewidth=3, solid_capstyle='butt')
    
    # 'N' label positioned above scale bar, centered with it
    ax.text(0.12, 0.20, 'N', transform=ax.transAxes,  # Centered with scale bar
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')
    
    # Add "2 km" label below the scale bar
    ax.text(x_center, y_scale - 0.01 * (ylim[1] - ylim[0]), '2 km',
            ha='center', va='top', fontsize=fontsize, color='black', fontweight='bold')

    ax.axis('off')
    plt.savefig(plot_name, dpi=300)
    plt.close()


def plot_distance_distributions(df_closest_nodes_distances, root):
    """
    Plot the distance distributions of the closest nodes.
    """
    plot_name = f"{root}{VISUALIZATIONS}/bike_graph_missing_edge_distances_distributions.png"
    if os.path.exists(plot_name):
        return None

    df_in = df_closest_nodes_distances[df_closest_nodes_distances['only_edge'] == 'incoming']
    df_out = df_closest_nodes_distances[df_closest_nodes_distances['only_edge'] == 'outgoing']
    
    # plot the distance distributions
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].hist(df_in['distance'], bins=100)
    axs[0].set_title('Distance to the closest node (only incoming)')
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Frequency')
    
    axs[1].hist(df_out['distance'], bins=100)
    axs[1].set_title('Distance to the closest node (only outgoing)')
    axs[1].set_xlabel('Distance')
    axs[1].set_ylabel('Frequency')

    incoming_text = f"{len(df_in)} nodes\n Avg dist (Std): {df_in['distance'].mean():.2f}m ({df_in['distance'].std():.2f}m)"
    outgoing_text = f"{len(df_out)} nodes\n Avg dist (Std): {df_out['distance'].mean():.2f}m ({df_out['distance'].std():.2f}m)"

    axs[0].text(0.95, 0.95, incoming_text, ha='right', va='center', transform=axs[0].transAxes)
    axs[1].text(0.95, 0.95, outgoing_text, ha='right', va='center', transform=axs[1].transAxes)

    plt.savefig(plot_name, dpi=300)
    plt.close()


def plot_new_edges(G_new, root):
    """
    Plot the new edges in the same plot as the original edges.
    """
    plot_name = f"{root}{VISUALIZATIONS}/bike_graph_with_new_edges.png"
    if os.path.exists(plot_name):
        return None

    bcn_boundary = dl.load_bcn_boundary()
    fig, ax = plt.subplots(figsize=(40, 40))
    bcn_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

    # Plot edges with different colors based on edge_type
    edge_colors = []
    edge_widths = []
    for _, _, data in G_new.edges(data=True):
        if data.get('edge_type') == 'new':
            edge_colors.append('red')
            edge_widths.append(8)
        else:
            edge_colors.append('gray') 
            edge_widths.append(1)

    ox.plot_graph(G_new, ax=ax, show=False, close=False,
                node_color='gray', node_size=1,
                edge_color=edge_colors, edge_linewidth=edge_widths,
                edge_alpha=1, bgcolor='white')
    
    plt.savefig(plot_name, dpi=300)
    plt.close()


def fix_directed_graph(G: nx.MultiDiGraph, nodes_gdf: gpd.GeoDataFrame, root: str, plots: bool = True) -> nx.MultiDiGraph:
    """
    Fixes a given directed graph by creating edges in both directions for all nodes.
    Edges are created for those nodes that have only one (unidirectional) edge. 
    This avoids errors when computing the shortest path between nodes.

    Args:
        G (nx.MultiDiGraph): The directed graph to fix.
        nodes_gdf (gpd.GeoDataFrame): The nodes of the graph.
        root (str): The root directory of the project.

    Returns:
        nx.MultiDiGraph: The fixed graph.
    """
    # categorize the nodes by connectivity
    bad_nodes = categorize_nodes_by_connectivity(G)
    _ = check_strong_connectivity(G)

    # plot the bad nodes
    if plots:
        plot_bad_nodes(G, bad_nodes, root)

    # find the closest node of the bad nodes
    dict_in_closest_node, dict_out_closest_node = find_the_closest_node_of_the_bad_nodes(
        G, bad_nodes, nodes_gdf)

    # create a dataframe with the closest nodes
    df_closest_nodes_distances = find_distance_between_closest_nodes(
        nodes_gdf, dict_in_closest_node, dict_out_closest_node)
    
    # plot the distance distributions
    if plots:
        plot_distance_distributions(df_closest_nodes_distances, root)

    # create new edges between the closest nodes
    G_new = create_new_edges_between_closest_nodes(G, df_closest_nodes_distances)

    # plot the new edges
    if plots:
        plot_new_edges(G_new, root)

    # unify the components
    G_unified = unify_components(G_new)

    # Check if the graph is now fully connected
    bad_nodes = categorize_nodes_by_connectivity(G_unified)
    _ = check_strong_connectivity(G_unified)
   
    return G_unified
