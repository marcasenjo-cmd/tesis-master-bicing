import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt
import ast

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.GA.graph_metric.graph_normalization as gn
import src.optimization.helper_optimization as ho
import src.optimization.experiments.helper_experiment as he
import src.results_exploration.helper_visualizations as hv



def compute_graph_scores( N_STATIONS, STATION_MIN_DISTANCE, distance_matrix, id_to_idx, idx_to_id, G):

    # Compute graph scores
    min_disp_bound, _ = gn.min_dispersion_bound(N_STATIONS, STATION_MIN_DISTANCE)
    max_disp_metric, _ = gn.max_dispersion_metric(N_STATIONS, distance_matrix, STATION_MIN_DISTANCE, idx_to_id)
    min_acc_bound, _ = gn.min_accessibility_bound_kmeans(G, N_STATIONS, distance_matrix, id_to_idx)
    max_acc_bound, _ = gn.max_accessibility_bound(N_STATIONS, distance_matrix, idx_to_id, STATION_MIN_DISTANCE)

    return min_disp_bound, max_disp_metric, min_acc_bound, max_acc_bound


def load_data():

    # Load nodes data
    all_data = he.load_data(root='./')
    df_nodes, G, distance_matrix, id_to_idx, idx_to_id, _, STATION_MIN_DISTANCE = all_data
    bcn_boundary = dl.load_bcn_boundary()

    # Load experiment results
    df_results = pd.read_csv(f'{PR_DATA}/experiments/MILP_for_each_var/MILP_scores_for_each_var.csv')
    
    # Transform string to list of integers
    df_results['best_solution'] = df_results['best_solution'].apply(
        lambda x: [int(i) for i in x.strip('[]').split(',')])
    
    N_STATIONS = len(df_results['best_solution'].values[0])
    
    # Transform string to dict of floats using ast.literal_eval
    df_results['weights'] = df_results['weights'].apply(
        lambda x: {k.strip("' "): float(v.strip("' ")) for k, v in [pair.split(':') for pair in ast.literal_eval(x)]})


    min_disp_bound, max_disp_metric, min_acc_bound, max_acc_bound = compute_graph_scores(N_STATIONS, STATION_MIN_DISTANCE, distance_matrix, id_to_idx, idx_to_id, G)
    bounds = (min_disp_bound, max_disp_metric, min_acc_bound, max_acc_bound)
    
    return df_results, df_nodes, bcn_boundary, bounds, distance_matrix, id_to_idx



def plot_solutions(results_list, df_nodes, bcn_boundary, bounds, distance_matrix, id_to_idx):

    # Get solutions and weights
    stations_list = [result['best_solution'].values[0] for result in results_list]
    weights_list = [result['weights'].values[0] for result in results_list]

    # Convert solution to indices
    stations_idx_list = []
    for solution in stations_list:
        stations_idx_list.append([id_to_idx[node_id] for node_id in solution])
        
    # Compute normalized graph scores
    alpha = 0
    min_disp_bound, max_disp_metric, min_acc_bound, max_acc_bound = bounds
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.005, 1, 1], width_ratios=[0.005, 1, 1], 
                         wspace=0.1, hspace=0.25)
    
    # Adjust figure margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Create axes for plots
    axs = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            axs[i][j] = fig.add_subplot(gs[i+1, j+1])
    
    # Add row and column labels
    row_labels = ['Max', 'Min']
    col_labels = ['Dispersion', 'Accessibility']
    
    # Add row labels
    for i, label in enumerate(row_labels):
        ax = fig.add_subplot(gs[i+1, 0])
        ax.axis('off')
        ax.text(0.5, 0.5, label, rotation=90, fontsize=12, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
    
    # Add column labels
    for j, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, j+1])
        ax.axis('off')
        ax.text(0.5, 0.5, label, fontsize=12, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
    
    # Plot each solution
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax = axs[i][j]
            
            # Get node normalized scores
            df_w = ho.sum_and_normalize_all_node_scores(df_nodes, weights_list[idx])
            df_w = df_w[df_w.index.isin(stations_list[idx])]

            # Compute graph scores
            dispersion, accessibility, _ = gn.evaluate_normalize_and_invert_stations_set(
                stations_idx_list[idx],
                distance_matrix,
                0,
                (min_disp_bound, max_disp_metric),
                (min_acc_bound, max_acc_bound)
                )

            # Plot nodes on map
            title = f"{weights_list[idx]}\nDisp: {dispersion:.2f}, Acc: {accessibility:.2f}"
            ax.set_title(title, fontsize=9)
            hv.plot_nodes_on_map(df_w, ax, bcn_boundary)
            
    return fig



def main():
    df_results, df_nodes, bcn_boundary, bounds, distance_matrix, id_to_idx = load_data()

    df_results.sort_values(by='dispersion_score', ascending=False, inplace=True)
    top_dispersion_df, low_dispersion_df = df_results.head(1), df_results.tail(1)

    df_results.sort_values(by='accessibility_score', ascending=False, inplace=True)
    top_accessibility_df, low_accessibility_df = df_results.head(1), df_results.tail(1)

    results_list = [top_dispersion_df, low_dispersion_df, top_accessibility_df, low_accessibility_df]
    fig = plot_solutions(results_list, df_nodes, bcn_boundary, bounds, distance_matrix, id_to_idx)
    plt.savefig(f'{VISUALIZATIONS}/graph_metrics/var_exploration_min_max_graph_score/vars_with_top_and_low_scores.png')


if __name__ == '__main__':
    main()