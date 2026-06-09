import sys
import os
import pickle
from pathlib import Path
from shapely import wkt  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import osmnx as ox  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable  # type: ignore
import contextily as ctx  # type: ignore

# Add project root to path
project_root = Path().resolve().parents[4]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.experiments.helper_experiment as he
import src.optimization.helper_optimization as ho


########################################################
# Prepare data for different cities
########################################################

# Read census sections' shapefile and select and rename columns
shp_file = f'{RAW_INE}/seccionado_2022/SECC_CE_20220101.shp'
census_sections = gpd.read_file(shp_file)
census_sections = census_sections[['NPRO', 'CPRO', 'NMUN', 'CMUN', 'CUSEC', 'geometry']]
census_sections.rename(columns={
    'NPRO': 'prov', 
    'CPRO': 'cpro',
    'NMUN': 'mun',
    'CMUN': 'cmun',
    'CUSEC': 'census_section'
    }, inplace=True)

# Read census section's population data and sum by census section
population_dir = f'{RAW_DATA}/ine'
population = f'{population_dir}/ine_census_section_population.parquet'
df_pop = pd.read_parquet(population)
df_pop.rename(columns={'muni_dist_section': 'census_section'}, inplace=True)
df_pop['population'] = df_pop['population'].astype(int)
df_pop = df_pop.groupby('census_section')['population'].sum().reset_index()

# Merge census sections with census section's population
census_sections = census_sections.merge(df_pop, on='census_section', how='left')

########################################################
# Chose cities to run the experiment
########################################################

top_10_cities = census_sections.groupby('mun')['population'].sum(
    ).sort_values(ascending=False).head(10)

top_10_cities = top_10_cities.reset_index()['mun'].tolist()[2:] # Remove 'Barcelona' and 'Madrid'
# [València, Sevilla, Zaragoza, Málaga, Murcia, Palma, Palmas de Gran Canaria, Bilbao]

# Create visualization directory
DIR_VISUALIZATION = f'{VISUALIZATIONS}/var_exploration_min_max_graph_score'
os.makedirs(DIR_VISUALIZATION, exist_ok=True)
city_rename = {'Palmas de Gran Canaria, Las': 'Palmas de Gran Canaria'}

# Plot census sections' population for the cities
for city in top_10_cities[2:]:
    print(city)

    # Filter census sections for the city
    city_cs = census_sections[census_sections['mun'] == city]
    
    # Plot census sections
    fig, ax = plt.subplots(figsize=(10, 10))
    city_cs.plot(ax=ax, column='population', alpha=0.5, legend=True)
    ax.axis('off')

    if city in city_rename:
        city = city_rename[city]
    plt.savefig(f'{DIR_VISUALIZATION}/census_sections_{city}.png', dpi=300)
    plt.close()


########################################################
# Download OSM bike network for the cities
########################################################

from src.assign_to_nodes.utils.fix_directed_graph import fix_directed_graph

def download_fix_and_save_graph(location, graph_file, crs, root, graph_path):
    """Helper function to download a graph, add edges on nodes with only one 
    directed edge, project the graph and save in both pickle (protocol 5) and 
    GraphML formats.
    
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
    G = fix_directed_graph(G, temp_nodes_gdf, root, plots=False)
    
    # Save the graph in both formats
    if graph_path:
        Path(f"{root}{graph_path}").mkdir(parents=True, exist_ok=True)
        
        # Save as pickle with protocol 5
        print(f"Saving graph to {graph_file} with protocol 5")
        with open(graph_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
            
    return G

crs = 25830
graph_path = f"{RAW_GRAPH}"
city_rename = {'Palmas de Gran Canaria, Las': 'Palmas de Gran Canaria'}

for city in top_10_cities[2:]: # Skip Madrid and Barcelona
    print(city)
    # Apply renaming if city is in the mapping
    if city in city_rename:
        city = city_rename[city]

    # Define location and graph file
    location = f"{city}, Spain"
    graph_file = f"{graph_path}/bike_graph_{city}.pkl"

    # Download and save graph if it doesn't exist
    if not os.path.exists(graph_file):
        download_fix_and_save_graph(location, graph_file, crs, graph_path=graph_path, root='./')


########################################################
# Assign population to bike OSM graph nodes
########################################################






