# -*- coding: utf-8 -*-
import sys
import os
import ast
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx  # type:ignore
import contextily as ctx  # type:ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type:ignore
from typing import Dict
import osmnx as ox
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.experiments.helper_experiment as he
import src.optimization.GA.graph_metric.graph_normalization as gn
import src.optimization.helper_optimization as ho
import src.optimization.experiments.graph_metrics.different_cities.GA_execution as gae
import src.optimization.experiments.graph_metrics.alpha_screening.alpha_screening_plot as asp

def parse_weights_from_string(weights_str):
    """
    Parse weights string from CSV into a dictionary.
    """
    items = ast.literal_eval(weights_str)
    return {item.split(': ')[0]: float(item.split(': ')[1].rstrip("'")) 
            for item in items}

def dict_to_str(d):
    """Convert a dictionary to a hashable string representation"""
    return json.dumps(d, sort_keys=True)

def get_file_names(score_combination: str, penalty_power: int = None) -> dict:
    """
    Centralize file naming patterns for the different cities experiment.
    """
    # Base names for files
    base_name = f"alpha_comparison_{score_combination}"
    config_name = f"experiment_config_{score_combination}"
    
    # Add penalty power suffix if using power_penalty
    if score_combination == 'power_penalty':
        base_name = f"{base_name}_{penalty_power}"
        config_name = f"{config_name}_{penalty_power}"
    
    # Construct output directory path
    output_dir = f"{VISUALIZATIONS}/graph_metrics/GA_alpha_screening/different_cities/{score_combination}"
    if score_combination == 'power_penalty':
        output_dir = f"{output_dir}/penalty_power_{penalty_power}"
    
    return {
        'config': f"{PR_EXP}/GA_alpha_screening/different_cities/{config_name}.txt",
        'experiment': f"{PR_EXP}/GA_alpha_screening/different_cities/{base_name}.csv",
        'output_dir': output_dir,
        'plot_base': base_name
    }

def load_experiment_data(score_combination):
    """
    Load the experiment data from CSV.
    """
    file_names = get_file_names(score_combination)
    results_df = pd.read_csv(file_names['experiment'])
    results_df['weights_dict'] = results_df['weights'].apply(asp.parse_weights_from_string)
    results_df['best_solution'] = results_df['best_solution'].fillna("[]")
    results_df['best_solution'] = results_df['best_solution'].apply(ast.literal_eval)
    results_df['weights_str'] = results_df['weights_dict'].apply(asp.dict_to_str)
    
    return results_df

def plot_all_nodes_map(G, city_boundary, df_weighted, ax):
    """
    Plot a map showing scores of all nodes.
    """
    # Get node positions
    pos = {node: (data.get('x'), data.get('y')) for node, data in G.nodes(data=True)}
    
    # Plot boundary
    city_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1, alpha=0.7)
    
    # Get all nodes and their scores
    all_nodes = list(G.nodes())
    node_scores = [df_weighted.loc[node, 'norm_score'] for node in all_nodes]
    
    # Draw all nodes with colors based on scores
    scatter = nx.draw_networkx_nodes(G, pos, nodelist=all_nodes,
                                   node_color=node_scores,
                                   cmap=cm.viridis_r,
                                   vmin=0, vmax=1,
                                   node_size=0.3,
                                   ax=ax)
    
    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    # Add basemap
    if 'crs' in G.graph:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=G.graph['crs'], zoom=12)
    
    ax.set_title("All Nodes Scores", fontsize=9)
    ax.set_axis_off()
    
    return ax

def plot_alpha_solution(df_solution, ax, city_boundary, metric_values, 
                        id_to_idx, distance_matrix, alpha, df_weighted,
                        score_combination, title_prefix=""):
    """
    Plot the solution for a specific alpha value.
    """    
    # Convert node IDs to indices for evaluation
    nodes_list = df_solution.index.tolist()
    nodes_idx_list = [id_to_idx.get(node, -1) for node in nodes_list if node in id_to_idx]
    valid_nodes_idx_list = [idx for idx in nodes_idx_list if idx != -1]
    
    if not valid_nodes_idx_list:
        ax.axis('off')
        ax.text(0.5, 0.5, "No valid nodes", 
               horizontalalignment='center', verticalalignment='center')
        return
    
    # Extract dispersion and accessibility bounds
    min_disp, max_disp, min_acc, max_acc = metric_values
    disp_bounds = (min_disp, max_disp)
    acc_bounds = (min_acc, max_acc)
    
    # Calculate metrics
    if alpha is None:
        inv_disp, acc_val, _ = gn.evaluate_normalize_and_invert_stations_set(
            valid_nodes_idx_list, distance_matrix, 0.0, disp_bounds, acc_bounds
        )
        composite = None
    else:
        inv_disp, acc_val, composite = gn.evaluate_normalize_and_invert_stations_set(
            valid_nodes_idx_list, distance_matrix, alpha, disp_bounds, acc_bounds
        )
    
    # Plot boundary
    city_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1, alpha=0.7)
    
    # Plot stations
    scatter = df_solution.plot(
        column='norm_score',
        cmap=plt.cm.viridis_r,
        ax=ax,
        markersize=0.5,
        legend=False,
        vmin=0, vmax=1)
    
    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    if len(df_weighted) > 0:
        original_min, original_max = df_weighted["score"].min(), df_weighted["score"].max()
        cbar.ax.set_ylabel(f'Score (min={original_min:.2f}, max={original_max:.2f})', fontsize=8)
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=df_weighted.crs, zoom=12)
    
    # Create title
    sum_scores = sum(df_solution['norm_score'])
    title_string = f"{title_prefix} Node Scores: {sum_scores:.2f}"
    if alpha is None:
        title_string += f"\nProx: {inv_disp:.2f}, Acc: {acc_val:.2f}"
    else:
        if score_combination == 'multiply':
            combined_score = sum_scores * composite
        elif score_combination == 'exponential':
            combined_score = sum_scores ** composite
        elif score_combination == 'power_penalty':
            combined_score = sum_scores * (composite ** 2)  # default penalty power
        else:
            combined_score = sum_scores * composite
            
        title_string += f"\nProx: {inv_disp:.2f}, Acc: {acc_val:.2f}, Comp: {composite:.2f}\nCombined ({score_combination}): {combined_score:.2f}"
    
    ax.set_title(title_string, fontsize=9)
    ax.set_axis_off()
    
    return ax

def create_city_comparison_plot(
    df_weighted, df_city_weight_stations, G, city_boundary, distance_matrix, 
    id_to_idx, bounds, n_stations, score_combination, city_name):
    """
    Create a plot comparing different alpha values for a city.
    """
    # Get alpha values, excluding None
    alpha_values = [a for a in df_city_weight_stations['alpha'].unique() if pd.notna(a)]
    
    # Create a grid of subplots (maximum 4 per row)
    n_alphas = len(alpha_values)
    n_cols = min(4, n_alphas + 2)  # +2 for all nodes map and no metrics solution
    n_rows = (n_alphas + 2 + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(15, 4 * n_rows))
    
    # Get a summary of the weight dictionary for the title
    sample_weight_dict = df_city_weight_stations.iloc[0]['weights_dict']
    weight_summary = ", ".join([f"{k}: {v:.2f}" for k, v in sample_weight_dict.items()])
    
    fig.suptitle(f"{city_name} - {n_stations} stations, {weight_summary}", fontsize=12, y=0.98)
    
    # Plot all nodes map first
    ax = fig.add_subplot(n_rows, n_cols, 1)
    plot_all_nodes_map(G, city_boundary, df_weighted, ax)
    
    # Plot solution without graph metrics second
    ax = fig.add_subplot(n_rows, n_cols, 2)
    
    # Get the solution without graph metrics (alpha=None)
    no_metrics_row = df_city_weight_stations[df_city_weight_stations['alpha'].isna()].iloc[0]
    solution_nodes = no_metrics_row['best_solution']
    df_solution = df_weighted.loc[solution_nodes]
    
    plot_alpha_solution(
        df_solution, ax, city_boundary, bounds, 
        id_to_idx, distance_matrix, None, df_weighted, score_combination,
        title_prefix="No metrics"
    )
    
    # Create subplots for alpha values
    for i, alpha in enumerate(sorted(alpha_values)):
        row = df_city_weight_stations[df_city_weight_stations['alpha'] == alpha].iloc[0]
        solution_nodes = row['best_solution']
        df_solution = df_weighted.loc[solution_nodes]
        ax = fig.add_subplot(n_rows, n_cols, i + 3)
        
        plot_alpha_solution(
            df_solution, ax, city_boundary, bounds, 
            id_to_idx, distance_matrix, alpha, df_weighted, score_combination,
            title_prefix=f"α={alpha}"
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main() -> None:
    """
    Main function to process and visualize different cities results.
    """
    score_combination = 'multiply'  # Options: 'multiply', 'exponential', 'power_penalty'
    penalty_power = None
    
    # Get all file names and paths
    file_names = get_file_names(score_combination, penalty_power)
    
    # Create output directory
    os.makedirs(file_names['output_dir'], exist_ok=True)
    
    # Load experiment data
    df_exp = load_experiment_data(score_combination)
    
    # Process each city separately
    for city in gae.CITIES:
        # if city == 'Sevilla': continue
        # if city == 'Palmas de Gran Canaria': continue
        print(f"\nProcessing city: {city}")
        
        # Filter data for this city
        df_city = df_exp[df_exp['city'] == city]
        if len(df_city) == 0:
            print(f"No data found for {city}")
            continue
        
        # Get node scores according to weights and their geometries
        weights = {'population': 1}
        df, G, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE = gae.compute_shared_data(city)
        df_weighted = ho.sum_and_normalize_all_node_scores(df.copy().reset_index(), weights)
        
        # Get city boundary from OSM
        city_boundary = ox.geocode_to_gdf(f"{city}, Spain")
        city_boundary = city_boundary.to_crs(G.graph['crs'])
        if city == 'Palma':
            # If city is Palma, remove the smallest polygon from the multi-polygon
            geometry = city_boundary.iloc[0].geometry
            if geometry.geom_type == 'MultiPolygon':
                # Find the largest polygon by area
                largest_polygon = max(geometry.geoms, key=lambda p: p.area)
                # Create a new GeoDataFrame with just the largest polygon
                city_boundary = city_boundary.iloc[[0]].copy()
                city_boundary.geometry = [largest_polygon]
            else:
                city_boundary = city_boundary.iloc[[0]]
        
        # Get normalization bounds for all station counts
        bounds = {}
        for n_stations in df_city['N_stations'].unique():
            disp_bounds, acc_bounds, _, _ = he.compute_metric_bounds_and_get_nodes(
                G, distance_matrix, id_to_idx, idx_to_id, n_stations, STATION_MIN_DISTANCE)
            bounds[n_stations] = (disp_bounds[0], disp_bounds[1], acc_bounds[0], acc_bounds[1])
        
        # Process each weight configuration
        for _, weight_df in df_city.groupby('weights_str'):

            # Create a weight identifier for the filename
            weights_dict = weight_df.iloc[0]['weights_dict']
            weight_id = '_'.join([f"{k}{int(v*100)}" for k, v in weights_dict.items() if v > 0])

            # Compute node scores according to weights and add their geometries
            if (type(df_weighted) != gpd.GeoDataFrame) & ('geometry' not in df_weighted.columns):
                # Get the nodes' geometry from the graph
                nodes_gdf, _ = ox.graph_to_gdfs(G)
                nodes_gdf = nodes_gdf[['geometry']]
                nodes_gdf.index.name = 'node_id'
                nodes_gdf.index = nodes_gdf.index.astype(int)

                # merge on index
                df_weighted = df_weighted.merge(nodes_gdf, left_index=True, right_index=True)
                df_weighted = gpd.GeoDataFrame(df_weighted, crs=G.graph['crs'])

            # Create a plot for each number of stations
            for n_stations in df_city['N_stations'].unique():

                # Create a weight + station count identifier for the filename
                weight_file = f"{file_names['output_dir']}/{city}_{file_names['plot_base']}_{n_stations}_stations_{weight_id}.png"

                # # Skip if the file already exists
                # if os.path.exists(weight_file):
                #     print(f"Skipping {weight_file} because it already exists")
                #     continue
                    
                # Filter data for this station count
                df_city_weight_stations = weight_df[weight_df['N_stations'] == n_stations]
                
                if len(df_city_weight_stations) == 0:
                    continue
                
                # Create the plot
                bounds_n_stations = bounds[n_stations]
                fig = create_city_comparison_plot(
                    df_weighted, df_city_weight_stations, 
                    G, city_boundary, distance_matrix, id_to_idx, 
                    bounds_n_stations, n_stations, score_combination, city
                )
                
                # Save the plot
                fig.savefig(weight_file, bbox_inches='tight')
                print(f"Saved plot to {weight_file}")
                plt.close(fig)

if __name__ == "__main__":
    main() 