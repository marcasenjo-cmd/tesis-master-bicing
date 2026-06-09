import pulp
import sys
import os 
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely import wkt  # type: ignore
import multiprocessing as mp
from functools import partial
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.helper_optimization as ho

# Set Gurobi environment variables
gurobi_path = list(Path('/opt').glob('gurobi*/linux64'))[0]
os.environ['GUROBI_HOME'] = str(gurobi_path)
os.environ['PATH'] = os.environ.get('PATH', '') + f":{gurobi_path}/bin"
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + f":{gurobi_path}/lib"
os.environ['GRB_LICENSE_FILE'] = '/home/ubuntu/gurobi.lic'  

# ===== Solver Functions =====
def compute_one_weight_exact_solution(solver, scores, distance_matrix, station_min_distance, id_to_idx, idx_to_id, n_stations, test_size=None):
    """
    Computes the optimal solution for the station selection problem using binary decision variables.
    This formulation is equivalent to the original integer programming problem. It selects nodes 
    (stations) to maximize the total score subject to incompatibility constraints from the distance_matrix.
    
    Parameters:
      solver(str): 'gurobi' or 'cbc'
      scores(pandas Series or dict-like): Mapping from node IDs to scores.
      distance_matrix(2D array (numpy array or similar)): Indicates the distance between nodes in meters.
      station_min_distance(int): Minimum distance between stations in meters.
      id_to_idx(dict): Mapping from node IDs to indices in the distance_matrix.
      idx_to_id(dict): Mapping from indices in the distance_matrix to node IDs.
      n_stations(int): Number of stations (nodes) to select.
      test_size(int or None): If provided, restrict the computation to the first test_size nodes.
    
    Returns:
      A tuple (optimal_value, selected_nodes, prob) where:
         - optimal_value(float): The total score of the selected nodes.
         - selected_nodes(list): A list of tuples (node_id, value) for nodes selected (value should be 1).
         - prob(pulp.LpProblem): The PuLP problem object.
    """
    # If test_size is provided, limit the computation to the first test_size nodes
    if test_size is not None and test_size < len(idx_to_id):
        n = test_size
        # Create a subset of idx_to_id for the test
        test_idx_to_id = {i: idx_to_id[i] for i in range(n)}
        # Also create a subset of scores
        considered_nodes = [idx_to_id[i] for i in range(n)]
        scores = scores[scores.index.isin(considered_nodes)]
    else:
        n = len(idx_to_id)
        test_idx_to_id = idx_to_id
        considered_nodes = list(scores.index)
    
    # Create a mixed-integer programming problem to maximize the total score.
    prob = pulp.LpProblem("ExactMaximization", pulp.LpMaximize)
    
    # Create binary decision variables for each node
    x = {node_id: pulp.LpVariable(f"x_{node_id}", cat='Binary') for node_id in considered_nodes}
    
    # Objective: maximize the weighted sum of selected nodes.
    prob += pulp.lpSum([scores[node_id] * x[node_id] for node_id in considered_nodes]), "TotalScore"
    
    # Constraint: select exactly n_stations nodes (or fewer if n_stations > n)
    prob += pulp.lpSum([x[node_id] for node_id in considered_nodes]) == min(n_stations, n), "NumberOfStations"
    
    # Add directed incompatibility constraints: for every pair of nodes (i,j) where i->j is incompatible
    for i, node_i in enumerate(considered_nodes):
        for j in range(i+1, len(considered_nodes)):  # avoid double and self loops
            node_j = considered_nodes[j]
            if distance_matrix[id_to_idx[node_i]][id_to_idx[node_j]] < station_min_distance or \
            distance_matrix[id_to_idx[node_j]][id_to_idx[node_i]] < station_min_distance:
                prob += x[node_i] + x[node_j] <= 1
    
    # Solve the LP problem.
    if solver == 'gurobi':
        solver = pulp.GUROBI_CMD(
            mip=True, 
            msg=False, 
            timeLimit=None, 
            gapRel=None, 
            gapAbs=None,
            threads=None, 
            logPath=None, 
            # path=None, 
            # keepFiles=False, 
            # options=None,
            # mip_start=False,
            )
    elif solver == 'cbc':
        solver = pulp.PULP_CBC_CMD()
    else:
        raise ValueError(f"Solver {solver} not supported. Use 'gurobi' or 'cbc'.")
    prob.solve(solver)
    
    # Check solver status
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        print(f"Warning: Solver did not find an optimal solution (status: {status}).")
        return None, None
    
    # After solving, get the nodes that were selected (should be 1 in the optimal solution)
    selected_nodes = []
    for node_id in considered_nodes:
        value = x[node_id].value()
        if value is not None and value != 0:
            selected_nodes.append(node_id)
    
    return pulp.value(prob.objective), selected_nodes


# ===== Processing Functions =====
def process_one_weight_combination(
    weight, df, N_STATIONS, distance_matrix, id_to_idx, idx_to_id, station_min_distance, solver='gurobi', test_size=None
):
    """
    Process a single weight combination using exact solver.
    
    Args:
        weight (dict): Weight dictionary for this combination.
        df (GeoDataFrame): DataFrame with node attributes.
        N_STATIONS (int): Number of stations.
        distance_matrix (ndarray): Distance matrix.
        station_min_distance (int): Minimum distance between stations in meters.
        id_to_idx (dict): Maps node IDs to matrix indices.
        idx_to_id (dict): Maps matrix indices to node IDs.
        solver (str): 'gurobi' or 'cbc'.
        test_size (int, optional): If provided, restrict computation to first test_size nodes.
    
    Returns:
        tuple: (optimal_value, selected_nodes, weight)
    """
    start_time = time.time()  # Start timing
    # Compute and normalize scores according to weights
    df_weighted = ho.sum_and_normalize_all_node_scores(df.copy(), weight)
    
    # Run exact solver
    optimal_value, selected_nodes = compute_one_weight_exact_solution(
        solver=solver,
        scores=df_weighted['norm_score'],
        distance_matrix=distance_matrix,
        station_min_distance=station_min_distance,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        n_stations=N_STATIONS,
        test_size=test_size
    )
    
    execution_time = (time.time() - start_time)/60 # minutes
    
    print(f"\tWeight set: {list(weight.items())[:3]}... - Score: {round(optimal_value, 3)} - Time: {round(execution_time, 2)}m")
    
    return optimal_value, selected_nodes, weight, execution_time


