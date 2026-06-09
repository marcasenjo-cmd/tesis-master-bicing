# -*- coding: utf-8 -*-
import sys
import os
import ast
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx  # type:ignore
import contextily as ctx  # type:ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type:ignore
from typing import Dict

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.experiments.helper_experiment as he
import src.optimization.GA.graph_metric.graph_normalization as gn
import src.optimization.helper_optimization as ho

def parse_weights_from_string(weights_str):
    """
    Parse weights string from CSV into a dictionary.
    """
    # The string format is like "['var1: val1', 'var2: val2']"
    # Remove the outer brackets and split by comma
    items = ast.literal_eval(weights_str)
    # Create an ordered dictionary from the items
    return {item.split(': ')[0]: float(item.split(': ')[1].rstrip("'")) 
            for item in items}

def dict_to_str(d):
    """Convert a dictionary to a hashable string representation"""
    return json.dumps(d, sort_keys=True)

def get_file_names(experiment_name: str, score_combination: str, penalty_power: int = None) -> dict:
    """
    Centralize file naming patterns for the alpha screening experiment.
    
    Args:
        score_combination (str): The score combination strategy (e.g., 'multiply', 'power_penalty')
        penalty_power (int, optional): The penalty power for power_penalty strategy
        
    Returns:
        dict: Dictionary containing all relevant file paths and names
    """
    # Base names for files
    base_name = f"alpha_comparison_{score_combination}"
    config_name = f"experiment_config_{score_combination}"
    
    # Add penalty power suffix if using power_penalty
    if score_combination == 'power_penalty':
        base_name = f"{base_name}_{penalty_power}"
        config_name = f"{config_name}_{penalty_power}"
    
    # Construct output directory path
    output_dir = f"{VISUALIZATIONS}/graph_metrics/{experiment_name}/{score_combination}"
    if score_combination == 'power_penalty':
        output_dir = f"{output_dir}/penalty_power_{penalty_power}"
    
    return {
        'config': f"{PR_EXP}/{experiment_name}/{config_name}.txt",
        'experiment': f"{PR_EXP}/{experiment_name}/{base_name}.csv",
        'output_dir': output_dir,
        'plot_base': base_name
    }

def load_experiment_data(experiment_name, score_combination):
    """
    Load the experiment data from CSV.
    
    Args:
        score_combination (str): The score combination strategy
        
    Returns:
        pd.DataFrame: Processed dataframe with weights and solutions
    """
    file_names = get_file_names(experiment_name, score_combination)
    results_df = pd.read_csv(file_names['experiment'])
    results_df['weights_dict'] = results_df['weights'].apply(parse_weights_from_string)
    results_df['best_solution'] = results_df['best_solution'].apply(ast.literal_eval)
    
    # Create string representation of weights for grouping
    results_df['weights_str'] = results_df['weights_dict'].apply(dict_to_str)
    
    return results_df

def compute_bounds_for_stations(G, distance_matrix, id_to_idx, idx_to_id, n_stations_values, min_distance=300):
    """
    Compute metric bounds for all station counts.
    
    Args:
        G (networkx.Graph): The graph
        distance_matrix (numpy.ndarray): Distance matrix
        id_to_idx (dict): Node ID to matrix index mapping
        idx_to_id (dict): Matrix index to node ID mapping
        n_stations_values (list): List of station counts
        min_distance (float): Minimum distance between stations
        
    Returns:
        dict: Dictionary mapping station counts to metric bounds
    """
    bounds = {}
    for n_stations in n_stations_values:
        disp_bounds, acc_bounds, _, _ = he.compute_metric_bounds_and_get_nodes(
            G, distance_matrix, id_to_idx, idx_to_id, n_stations, min_distance)
        
        min_disp, max_disp = disp_bounds
        min_acc, max_acc = acc_bounds
        bounds[n_stations] = (min_disp, max_disp, min_acc, max_acc)
    
    return bounds

def process_weight_category(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process weight categories from experiment data.
    
    Args:
        data: DataFrame containing experiment results
        
    Returns:
        Dictionary mapping weight configurations to their corresponding data
    """        
    # Group by weight configurations
    weight_dfs = {}
    for weight_str, group in data.groupby('weights_str'):
        # Sort by alpha, putting None first
        group = group.sort_values('alpha', na_position='first')
        weight_dfs[weight_str] = group.copy()
        
    if not weight_dfs:
        raise ValueError("No weight configurations found in data")
        
    return weight_dfs

def plot_all_nodes_map(G, bcn_boundary, df_weighted, ax):
    """
    Plot a map showing scores of all nodes.
    
    Args:
        G (networkx.Graph): The graph
        bcn_boundary (GeoDataFrame): Boundary of Barcelona
        df_weighted (pd.DataFrame): DataFrame containing node scores
        ax (matplotlib.axes.Axes): Axes to plot on
    """
    # Get node positions
    pos = {node: (data.get('x'), data.get('y')) for node, data in G.nodes(data=True)}
    
    # Plot boundary
    bcn_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    
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

def plot_alpha_solution(G, bcn_boundary, nodes_list, metric_values, ax, 
                        id_to_idx, distance_matrix, alpha, df_weighted,
                        experiment_name, score_combination, title_prefix="", node_size=2.5):
    """
    Plot the solution for a specific alpha value using the same visualization style
    as the third subplot in plot_all_weight_comparison.
    
    Args:
        G (networkx.Graph): The graph
        bcn_boundary (GeoDataFrame): Boundary of Barcelona
        nodes_list (list): List of selected node IDs
        metric_values (tuple): Tuple of (min_disp, max_disp, min_acc, max_acc)
        ax (matplotlib.axes.Axes): Axes to plot on
        id_to_idx (dict): Node ID to matrix index mapping
        distance_matrix (numpy.ndarray): Distance matrix
        alpha (float): Alpha value used for weighting
        df_weighted (pd.DataFrame): DataFrame containing node scores
        score_combination (str): The score combination strategy
        title_prefix (str): Prefix for the plot title
        node_size (float): Size of the selected nodes
    """
    # Mark nodes in graph
    G_marked = he.mark_graph(G.copy(), nodes_list)
    
    # Convert node IDs to indices for evaluation
    nodes_idx_list = [id_to_idx.get(node, -1) for node in nodes_list if node in id_to_idx]
    
    # Skip nodes that aren't in the mapping
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
        # For no metrics case, calculate individual metrics without composite
        inv_disp, acc_val, _ = gn.evaluate_normalize_and_invert_stations_set(
            valid_nodes_idx_list, distance_matrix, 0.0, disp_bounds, acc_bounds
        )
        composite = None
    else:
        # For alpha cases, calculate all metrics including composite
        inv_disp, acc_val, composite = gn.evaluate_normalize_and_invert_stations_set(
            valid_nodes_idx_list, distance_matrix, alpha, disp_bounds, acc_bounds
        )
    
    # Get node positions
    pos = {node: (data.get('x'), data.get('y')) for node, data in G.nodes(data=True)}
    
    # Plot boundary
    bcn_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    
    # Draw selected nodes with colormap based on scores
    selected_nodes = [node for node in nodes_list if node in G_marked.nodes()]
    
    # Get scores for selected nodes and calculate sum
    node_scores = []
    sum_scores = 0
    for node in selected_nodes:
        score = df_weighted.loc[node, 'norm_score']
        node_scores.append(score)
        sum_scores += score
    
    # Draw nodes with colors based on scores
    scatter = nx.draw_networkx_nodes(G_marked, pos, nodelist=selected_nodes,
                                   node_color=node_scores,
                                   cmap=cm.viridis_r,
                                   vmin=0, vmax=1,
                                   node_size=node_size,
                                   ax=ax)
    
    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    if len(df_weighted) > 0:
        original_min, original_max = df_weighted["score"].min(), df_weighted["score"].max()
        cbar.ax.set_ylabel(f'Score (min={original_min:.2f}, max={original_max:.2f})', fontsize=8)
    
    # Add basemap
    if 'crs' in G.graph:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=G.graph['crs'], zoom=12)
    
    # Create title
    title_string = f"{title_prefix} Node Scores: {sum_scores:.2f}"
    if alpha is None:
        title_string += f"\nProx: {inv_disp:.2f}, Acc: {acc_val:.2f}"
    else:
        # Get score combination strategy from config
        file_names = get_file_names(experiment_name, score_combination)
        with open(file_names['config'], 'r') as f:
            config_text = f.read()
            # Parse config text to get parameters
            penalty_power = 2  # default
            for line in config_text.split('\n'):
                if 'penalty_power' in line and 'None' not in line:
                    penalty_power = float(line.split(':')[1].strip())
        
        # Calculate combined score based on strategy
        if score_combination == 'multiply':
            combined_score = sum_scores * composite
        elif score_combination == 'exponential':
            combined_score = sum_scores ** composite
        elif score_combination == 'power_penalty':
            combined_score = sum_scores * (composite ** penalty_power)
        else:
            combined_score = sum_scores * composite  # default to multiply
            
        title_string += f"\nProx: {inv_disp:.2f}, Acc: {acc_val:.2f}, Comp: {composite:.2f}\nCombined ({score_combination}): {combined_score:.2f}"
    
    ax.set_title(title_string, fontsize=9)
    ax.set_axis_off()
    
    return ax

def create_alpha_comparison_plot(
    df_alpha_one_weight, G, bcn_boundary, distance_matrix, 
    id_to_idx, bounds, n_stations, df_nodes, weights, 
    experiment_name, score_combination):
    """
    Create a plot comparing different alpha values for a weight configuration.
    
    Args:
        df_alpha_one_weight (pd.DataFrame): Dataframe containing experiments for one weight configuration
        G (networkx.Graph): The graph
        bcn_boundary (GeoDataFrame): Boundary of Barcelona
        distance_matrix (numpy.ndarray): Distance matrix
        id_to_idx (dict): Node ID to matrix index mapping
        bounds (dict): Cache of metric bounds for each station count
        n_stations (int): Number of stations
        df_nodes (pd.DataFrame): DataFrame containing all nodes
        weights (dict): Weights for the nodes
        experiment_name (str): The experiment name
        score_combination (str): The score combination strategy
    """
    # Get alpha values, excluding None
    alpha_values = [a for a in df_alpha_one_weight['alpha'].unique() if pd.notna(a)]
    df_weighted = ho.sum_and_normalize_all_node_scores(df_nodes.copy().reset_index(), weights)
    
    # Create a grid of subplots (maximum 4 per row)
    n_alphas = len(alpha_values)
    n_cols = min(4, n_alphas + 2)  # +2 for all nodes map and no metrics solution
    n_rows = (n_alphas + 2 + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(15, 4 * n_rows))
    
    # Get a summary of the weight dictionary for the title
    sample_weight_dict = df_alpha_one_weight.iloc[0]['weights_dict']
    weight_summary = ", ".join([f"{k}: {v:.2f}" for k, v in sample_weight_dict.items()])
    
    fig.suptitle(f"{n_stations} stations, {weight_summary}", fontsize=12, y=0.98)
    
    # Plot all nodes map first
    ax = fig.add_subplot(n_rows, n_cols, 1)
    plot_all_nodes_map(G, bcn_boundary, df_weighted, ax)
    
    # Plot solution without graph metrics second
    ax = fig.add_subplot(n_rows, n_cols, 2)
    
    # Get the solution without graph metrics (alpha=None)
    no_metrics_row = df_alpha_one_weight[df_alpha_one_weight['alpha'].isna()].iloc[0]
    solution_nodes = no_metrics_row['best_solution']

    # Get the metric bounds for this station count
    bounds_values = bounds[n_stations]
    
    # Use plot_alpha_solution with None alpha and dummy metric values
    plot_alpha_solution(
        G, bcn_boundary, solution_nodes, bounds_values, ax, 
        id_to_idx, distance_matrix, None, df_weighted, experiment_name, score_combination,
        title_prefix="No metrics"
    )
    
    # Create subplots for alpha values
    for i, alpha in enumerate(sorted(alpha_values)):
        # Get the row for this alpha
        row = df_alpha_one_weight[df_alpha_one_weight['alpha'] == alpha].iloc[0]
        
        # Extract solution and station count
        solution_nodes = row['best_solution']
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 3)  # +3 because first two subplots are at 1 and 2
        
        # Plot the solution
        plot_alpha_solution(
            G, bcn_boundary, solution_nodes, bounds_values, ax, 
            id_to_idx, distance_matrix, alpha, df_weighted, experiment_name, score_combination,
            title_prefix=f"α={alpha}"
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to leave room for the suptitle
    
    return fig

def main() -> None:
    """
    Main function to process and visualize alpha screening results.
    """
    score_combination = 'exponential' # 'multiply', 'exponential', 'power_penalty'
    penalty_power = None
    experiment_name = 'GA_alpha_screening/var_exploration'
    
    # Get all file names and paths
    file_names = get_file_names(experiment_name, score_combination, penalty_power)
    
    # Create output directory
    os.makedirs(file_names['output_dir'], exist_ok=True)
    
    # Load experiment data
    df_alpha_exp = load_experiment_data(experiment_name, score_combination)
    weight_dfs_dict = process_weight_category(df_alpha_exp)

    # Load graph and scores data
    df_nodes, G, distance_matrix, id_to_idx, idx_to_id, distance_matrix_binary, STATION_MIN_DISTANCE = he.load_data(root='./')
    bcn_boundary = dl.load_bcn_boundary()

    # Precompute bounds for all station counts
    bounds = {}
    for n_stations in df_alpha_exp['N_stations'].unique():
        bounds[n_stations] = compute_bounds_for_stations(G, distance_matrix, id_to_idx, idx_to_id, [n_stations])
    
    # Process each weight configuration separately
    for weight_str, weight_df in weight_dfs_dict.items():
        # Get the weights dictionary for this configuration
        weights_dict = weight_df.iloc[0]['weights_dict']
        
        # Create a plot for each number of stations
        for n_stations in df_alpha_exp['N_stations'].unique():
            # Create a weight identifier for the filename
            weight_id = '_'.join([f"{k}{int(v*100)}" for k, v in weights_dict.items() if v > 0])
            weight_file = f"{file_names['output_dir']}/{file_names['plot_base']}_{n_stations}_stations_{weight_id}.png"
            if os.path.exists(weight_file):
                print(f"Skipping {weight_file} because it already exists")
                continue
            
            # Filter data for this station count
            df_alpha_one_weight = weight_df[weight_df['N_stations'] == n_stations]

            if len(df_alpha_one_weight) == 0:
                continue
            
            # Create the plot
            fig = create_alpha_comparison_plot(
                df_alpha_one_weight, 
                G, bcn_boundary, distance_matrix, id_to_idx, 
                bounds[n_stations], n_stations, df_nodes, weights_dict,
                experiment_name, score_combination
            )
            
            # Save the plot
            fig.savefig(weight_file, bbox_inches='tight')
            print(f"Saved plot to {weight_file}")
            plt.close(fig)

if __name__ == "__main__":
    main()
