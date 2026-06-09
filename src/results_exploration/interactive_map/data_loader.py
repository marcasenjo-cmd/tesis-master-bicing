"""
Data loader module for loading solution data, weights, and datasets.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import osmnx as ox

# Add project root to the path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.assign_to_nodes.utils.load_input_datasets as lid 
import src.optimization.helper_optimization as ho
from src.optimization.experiments.helper_experiment import _load_node_scores, _load_graph
import src.optimization.experiments.graph_metrics.different_cities.GA_execution as ga


def load_best_solution(experiment_path=None):
    """
    Load the best solution from the results file.
    
    Args:
        root (str): Optional custom root path
        experiment_path (str): Optional custom experiment path
        
    Returns:
        list: List of node IDs representing the best solution
    """
    
    file = os.path.join(experiment_path, 'GA_scores_and_nodes.csv')
    df_solutions = pd.read_csv(file)
    best_solution_row = df_solutions.sort_values(by='best_score', ascending=False).iloc[0]
    
    # Print hyperparameters and best score
    print(f"Hyperparameters: Pop={best_solution_row['population_size']}, Mut={best_solution_row['mutation_rate']}, " +
          f"Elit={best_solution_row['elite_fraction']}, Sel={best_solution_row['selection_strategy']}, Cross={best_solution_row['crossover_strategy']}")
    print(f"Best score: {best_solution_row['best_score']}, Generations: {best_solution_row['generations']}, " +
          f"Time: {round(best_solution_row['minutes_to_complete'], 2)} minutes")
    
    # Convert the string representation of the list to an actual list
    return eval(best_solution_row['best_solution'])


def load_weights(experiment_path=None):
    """
    Load optimization weights from experiment config file.
    
    Args:
        root (str): Optional custom root path
        experiment_path (str): Optional custom experiment path
        
    Returns:
        dict: Dictionary of weights for different features
    """
    # try:
    #     config_file = os.path.join(experiment_path, 'experiment_config.txt')
    #     with open(config_file, 'r') as f:
    #         for line in f:
    #             if 'weights' in line:
    #                 # Extract the dictionary part and evaluate it properly
    #                 weights_str = '{' + line.split('{')[1].strip()
    #                 return eval(weights_str)
    # except Exception as e:
    #     print(f"Error loading weights: {e}")
    #     return {}

    weights = {
        '10-19': 0.026383856253122746,
        '20-29': 0.018128479443205663,
        '30-39': 0.020894182641601894,
        '40-49': 0.04284082787505742,
        '50-59': 0.00693530730518993,
        '60-69': 0.0265122063045541,
        '70+': 0.022878782841162175, 
        'altitude': 0.026062603881983865, 
        'bike_lane_kms': 0.019181282226755886, 
        'bus_lines': 0.04171997495496813, 
        'cars': 0.0008190318940605601, 
        'education_college': 0.015634078987251363, 
        'education_primary': 0.024544677910801838, 
        'education_secondary': 0.0113875961823151, 
        'f': 0.008212745017959844, 
        'has_bike_lane': 0.02743864880168448, 
        'household_avg_m2': 0.027652978320578343, 
        'income_2022_pers': 0.03138377359804678, 
        'm': 0.041084617984563816, 
        'metro_lines': 0.0068985683675105965, 
        'motos': 0.03944174747558563, 
        'n_civic': 0.041645014689607684, 
        'n_culture': 0.04225668791795071, 
        'n_economic_retail': 0.04453937963935907, 
        'n_education': 0.02346441763609586, 
        'n_green': 0.0056378617088957165, 
        'n_health_care': 0.03330564300951996, 
        'n_industrial': 0.03793911179824263, 
        'n_recreation': 0.020389181464479544, 
        'n_sport': 0.005121183312228021, 
        'n_superpois': 0.02674039340740258, 
        'n_tourism': 0.04185877623928338, 
        'n_worship': 0.03934100681166191, 
        'non_spanish_population': 0.04215498914552694, 
        'pois_entropy': 0.0024765915801834766, 
        'pois_total': 0.018044871521954033, 
        'population': 0.03150598853238253, 
        'tram_lines': 0.03198482157579907, 
        'unemployment_percentage': 0.0255581117414667
    }

    for key, value in weights.items():
        weights[key] = round(value, 3)

    return weights


def load_datasets(city_name, weights):
    """
    Load all required datasets for map visualization.
    
    Args:
        root (str): Optional custom root path
        
    Returns:
        dict: Dictionary containing all loaded datasets
    """
    try:
        # Load configuration and graph data
        df, G, distance_matrix, id_to_idx, idx_to_id, STATION_MIN_DISTANCE = ga.compute_shared_data(city_name)

        
        # Load node attributes
        node_attrs_path = f'data/processed/nodes/node_attributes.csv'
        df_raw = pd.read_csv(node_attrs_path)
        
        # Load normalized node attributes and convert to WGS84
        df_weighted = ho.sum_and_normalize_all_node_scores(df, weights)

        if 'geometry' not in df_weighted.columns:
            # Get the nodes' geometry from the graph
            nodes_gdf, _ = ox.graph_to_gdfs(G)
            nodes_gdf = nodes_gdf.to_crs(4326)
            nodes_gdf = nodes_gdf[['geometry']]
            nodes_gdf.index.name = 'node_id'
            nodes_gdf.index = nodes_gdf.index.astype(int)
            df_weighted = df_weighted.merge(nodes_gdf, left_index=True, right_index=True)

        if 'altitude' in df_weighted.columns:
            df_weighted = df_weighted.drop(columns=['altitude'])

        df_weighted = gpd.GeoDataFrame(df_weighted).to_crs(4326)
        df_weighted = df_weighted[list(weights.keys()) + ['norm_score', 'geometry']]
        
        # Additional datasets that might be used
        bcn_buildings = dl.load_bcn_building_data().to_crs(epsg=4326)
        df_socio_census = lid.load_socioeconomic_census_section().to_crs(epsg=4326)
        df_socio_nei = lid.load_socioeconomic_neighborhood().to_crs(epsg=4326)
        df_bike_lanes = lid.load_bike_lanes().to_crs(epsg=4326)
        df_pois = lid.load_pois().to_crs(epsg=4326)
        bus, tram, metro = lid.load_pt_infrastructure()
        bus, tram, metro = bus.to_crs(epsg=4326), tram.to_crs(epsg=4326), metro.to_crs(epsg=4326)
        
        return {
            'df': df,
            'G': G, 
            'distance_matrix': distance_matrix,
            'id_to_idx': id_to_idx,
            'idx_to_id': idx_to_id,
            'df_raw': df_raw,
            'df_weighted': df_weighted,
            'STATION_MIN_DISTANCE': STATION_MIN_DISTANCE,
            'bcn_buildings': bcn_buildings,
            'df_socio_census': df_socio_census,
            'df_socio_nei': df_socio_nei,
            'pt': {'bus': bus, 'tram': tram, 'metro': metro},
            'df_bike_lanes': df_bike_lanes,
            'df_pois': df_pois
        }
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise 