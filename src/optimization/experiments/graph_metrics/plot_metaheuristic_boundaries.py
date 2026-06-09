# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
import networkx as nx  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import contextily as ctx  # type:ignore
import pandas as pd  # type:ignore
import geopandas as gpd  # type:ignore
from shapely import wkt  # type:ignore
import osmnx as ox
import pickle

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.experiments.helper_experiment as he
import src.optimization.GA.graph_metric.graph_normalization as gn
import src.optimization.helper_optimization as ho


def get_all_cities():
    """Get list of all cities to process"""
    cities = [
        'Barcelona',  # Default city
        # 'Bilbao',
        # # 'Málaga',  # 26000 nodes
        # # 'Murcia',  # 30000 nodes
        # 'Palma',
        # 'Palmas de Gran Canaria',
        # 'Sevilla', 
        # 'València',
        # 'Zaragoza'
    ]
    return cities


def load_city_data(city):
    """
    Load data for a specific city.
    
    Args:
        city (str): Name of the city
        
    Returns:
        tuple: Contains graph, distance matrix, mappings and boundary data
    """
    if city == 'Barcelona':
        graph_file = f"{RAW_GRAPH}/bike_graph.pickle"
    else:
        graph_file = f"{RAW_GRAPH}/different_cities/bike_graph_{city}.pickle"

    # Load graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Compute distance matrix
    distance_matrix, id_to_idx, idx_to_id = ho.compute_all_pairs_shortest_paths_dijkstra(city, G, weight='weight', use_elevation=False)
    
    # Download city boundary from OSM
    city_boundary = ox.geocode_to_gdf(city)
    city_boundary = city_boundary.to_crs(G.graph['crs'])
        
    return G, distance_matrix, id_to_idx, idx_to_id, city_boundary


def add_north_arrow(ax, size=0.05, location=(0.9, 0.15), linewidth=1, fontsize=12, color='k'):
    """
    Adds a north arrow to the given axes.
    """
    ax.annotate(
        '', 
        xy=(location[0], location[1]), 
        xytext=(location[0], location[1] - size * 1.2),  # Make arrow line longer
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(facecolor=color, edgecolor=color, width=linewidth, headwidth=8, headlength=12),
        ha='center', va='center'
    )
    ax.text(location[0], location[1] - size * 1.3 - 0.018, 'N', transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color)
    

def add_scalebar(ax, length, location=(0.1, 0.05), linewidth=3, units='m', fontsize=14, color='k'):
    """
    Adds a scale bar to the map.
    """
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate start and end points in data coordinates
    x_start = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_start = ylim[0] + location[1] * (ylim[1] - ylim[0])
    x_end = x_start + length
    
    # Draw the scale bar
    ax.plot([x_start, x_end], [y_start, y_start], color=color, linewidth=linewidth, solid_capstyle='butt')

    # Add the label - convert to display units if needed
    if units == 'km':
        display_length = length / 1000
        label_text = f'{display_length:.0f} {units}'
    else:
        label_text = f'{int(length)} {units}'
    
    ax.text((x_start + x_end) / 2, y_start - 0.01 * (ylim[1] - ylim[0]), label_text,
            ha='center', va='top', fontsize=fontsize, color=color, fontweight='bold')


def plot_graph_with_highlighted_nodes(G, city_boundary, title_string, ax,
                                    selected_nodes=False, 
                                    selected_node_color='red',
                                    selected_node_size=5,
                                    show_arrow_scale=True):
    """
    Plots the graph with highlighted nodes based on selection status.
    
    Args:
        G (networkx.Graph): Graph to plot
        city_boundary (GeoDataFrame): Boundary polygon of the city
        title_string (str): Title for the plot
        selected_nodes (list): List of nodes to highlight
        ax (matplotlib.axes.Axes): Axes to plot on
        selected_node_color (str): Color for selected nodes
        selected_node_size (float): Size for selected nodes
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    # Reproject boundary if needed
    if 'crs' in G.graph:
        city_boundary = city_boundary.to_crs(G.graph['crs'])
    
    # Get selected and non-selected nodes
    if not selected_nodes:
        selected_nodes = [node for node, data in G.nodes(data=True) if data.get('selected')]
    
    # Get node positions
    pos = {node: (data.get('x'), data.get('y')) for node, data in G.nodes(data=True)}

    # Plot boundary
    city_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    
    # Draw selected nodes
    nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes,
                          node_color=selected_node_color,
                          node_size=selected_node_size, ax=ax)
    
    # # Add basemap
    # if 'crs' in G.graph:
    #     ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=G.graph['crs'], zoom=12) # type: ignore
    
    # Set title with smaller font

    # Create simplified title with just the metrics   
    ax.set_title(title_string, fontsize=10, pad=0)
    
    # Turn off axis
    ax.set_axis_off()
    
    # Add north arrow and scale bar only if requested
    if show_arrow_scale:
        add_north_arrow(ax, size=0.05, location=(0.158, 0.19), linewidth=1, fontsize=8)
        add_scalebar(ax, length=2000, location=(0.1, 0.075), linewidth=3, units='km', fontsize=8, color='k')
    
    return ax


def process_city(city):
    """Process a single city and generate its visualization"""
    results_dir = f"{VISUALIZATIONS}/graph_metrics/normalization_boundaries"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{city}_graph_metric_bounds.png"

    # # Skip if results already exist
    # if os.path.exists(results_file):
    #     print(f"Skipping {city} - results already exist in {results_file}")
    #     return

    print(f"Processing {city}...")

    # Constants
    ALPHA = 0.5
    n_station_values = [50]
    STATION_MIN_DISTANCE = 300
    
    # Load city data
    G, distance_matrix, id_to_idx, idx_to_id, city_boundary = load_city_data(city)
    
    # Compute metric bounds
    metric_bounds, nodes_bounds = {}, {}
    for n_stations in n_station_values:
        disp_bounds, acc_bounds, disp_nodes, acc_nodes = he.compute_metric_bounds_and_get_nodes(
            G, distance_matrix, id_to_idx, idx_to_id, n_stations, STATION_MIN_DISTANCE)
        
        metric_bounds[n_stations] = (disp_bounds, acc_bounds)
        nodes_bounds[n_stations] = (disp_nodes, acc_nodes)
        
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.3})
    
    # Configuration names - using A, B, C for the 3 maps
    config_names = ['A', 'B', 'C']
    
    # Assuming we're using the first (and only) n_stations value
    n_stations = n_station_values[0]
    
    # Get the metrics and nodes for each configuration
    min_disp_nodes, max_disp_nodes = nodes_bounds[n_stations][0]
    min_acc_nodes, max_acc_nodes = nodes_bounds[n_stations][1]
    min_disp, max_disp = metric_bounds[n_stations][0]
    min_acc, max_acc = metric_bounds[n_stations][1]
    
    # The three configurations to plot (A, B, C)
    configs = [max_disp_nodes, min_acc_nodes, max_acc_nodes]
    metric_values = [(min_disp, max_disp), (min_acc, max_acc)]
    
    # Plot each configuration
    for col_idx, (config_name, nodes_list) in enumerate(zip(config_names, configs)):
        nodes_idx_list = [id_to_idx[node] for node in nodes_list]
        
        ax = axes[col_idx]
        
        # Skip plotting for empty node lists but still add a "No data" message
        if not nodes_list:
            ax.axis('off')
            ax.text(0.5, 0.5, "No data",
                   horizontalalignment='center', verticalalignment='center')
            continue
        
        ho.validate_solution(nodes_list, distance_matrix, id_to_idx, idx_to_id, n_stations, STATION_MIN_DISTANCE)

        # Mark nodes in graph
        G_marked = he.mark_graph(G.copy(), nodes_list)
        
        # Evaluate metrics          
        inv_pro, inv_acc, composite = gn.evaluate_normalize_and_invert_stations_set(
            nodes_idx_list, distance_matrix, ALPHA, metric_values[0], metric_values[1])
            
        title_string = f"Prox: {inv_pro:.2f}, Acc: {inv_acc:.2f}"

        # Plot
        plot_graph_with_highlighted_nodes(
            G_marked, 
            city_boundary, 
            title_string, 
            ax=ax,
            selected_node_size=1,
            selected_node_color='red',
            show_arrow_scale=(col_idx == 0)  # Only show arrow and scale for first subplot
        )
        
        # Add letter label in top left corner
        ax.text(-0.1, 1.1, config_name, transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top', horizontalalignment='left')
    
    plt.savefig(results_file, dpi=500, bbox_inches='tight', pad_inches=0)
    print(f"Saved {results_file}")
    plt.show()


def main():
    # Get list of all cities
    cities = get_all_cities()
    
    # Process each city
    for city in cities:
        process_city(city)
        print(f"Finished {city}!!")

if __name__ == "__main__":
    main()
