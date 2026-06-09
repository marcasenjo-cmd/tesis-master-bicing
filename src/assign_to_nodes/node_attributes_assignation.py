"""
Node Attributes Assignment Pipeline

This module orchestrates the complete process of assigning spatial and demographic attributes
to graph nodes that will be used in the BSS station placement optimization. It uses various 
spatial assignment methods.

The module handles:
1. **Infrastructure Data**: Public transport lines (bus, metro, tram) and bike lanes
2. **Urbanism Data**: Points of Interest (POIs) and cycling infrastructure
3. **Socioeconomic Data**: Population demographics, education, income, and vehicle ownership
4. **Building Data**: Residential buildings for weighted spatial assignments

Key Features:
- Spatial assignment using buffer-based methods for each graph node and weighted means
- Building-area weighted assignments for accurate population distribution
- Parallel processing for large datasets

Data Sources:
- Graph nodes from network analysis
- Datasets from create_input_dataset pipeline

Processing Methods:
- **Weighted Mean**: For continuous variables using spatial buffers
- **Building-Area Weighted**: For count data using residential building footprints
- **Spatial Join**: For categorical and boundary-based data

Author: Jordi Grau Escolano
"""
import sys
import os
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from shapely.geometry import Point  # type: ignore
import numpy as np

# Define project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.assign_to_nodes.utils.load_input_datasets as lid 
import src.assign_to_nodes.utils.class_node_assigner as cna

# Constants
LOCATION = 'Barcelona, Barcelona, Catalunya, España'
BUFFER_SIZE = 300
CRS = 25831

# Initialize Node Assigner
node_assigner = cna.NodeAttributesAssigner(
    LOCATION, graph_path=RAW_GRAPH, crs=CRS, buffer_size=BUFFER_SIZE
)
node_assigner.node_attributes.drop(columns=[], inplace=True) # for testing purposes

# Load datasets
bcn_buildings = dl.load_bcn_building_data()
df_socio_census = lid.load_socioeconomic_census_section()
df_socio_nei = lid.load_socioeconomic_neighborhood()
bus, metro, tram = lid.load_pt_infrastructure(crs=CRS)
df_bike_lanes = lid.load_bike_lanes()
df_pois = lid.load_pois(crs=CRS)


# INFRASTRUCTURE
if not all(col in node_assigner.node_attributes.columns for col in ['bus_lines', 'tram_lines', 'metro_lines']):
    for transport_gdf, transport_mode in zip([bus, tram, metro], ['bus', 'tram', 'metro']):
        node_assigner.assign_public_transport_unique_lines(
            transport_gdf=transport_gdf, transport_mode=transport_mode
        )

# URBANISM
if not all(col in node_assigner.node_attributes.columns for col in ['bike_lane_kms', 'pois_total', 'pois_entropy']):
    node_assigner.assign_bike_lanes(bike_lanes_gdf=df_bike_lanes)
    node_assigner.assign_bike_lane_kms(bike_lanes_gdf=df_bike_lanes)
    node_assigner.assign_pois(pois_gdf=df_pois)

del(bus, metro, tram, df_bike_lanes, df_pois)

# SOCIOECONOMIC DATA
print("Assigning census section data to nodes with weighted mean")
vars_to_assign_with_residential_buildings = [
    'education_primary', 'education_secondary', 'education_college', 
    'f', 'm', '10-19', 
    '20-29', '30-39', '40-49', 
    '50-59', '60-69', '70+', 
    'population', 'cars_abs', 'motos_abs', 
    'others_abs', 'non_spanish_population'
]
socio_census_columns = [col for col in df_socio_census.columns if col not in [
        'census_section', 'geometry']]
socio_census_columns = [col for col in socio_census_columns if col not in vars_to_assign_with_residential_buildings]

# Assign census section data to nodes by weighted mean. Without considering buildings.
if not all(col in node_assigner.node_attributes.columns for col in socio_census_columns):
    # Print missing columns
    print(f"Missing columns: {set(socio_census_columns) - set(node_assigner.node_attributes.columns)}")

    node_assigner.assign_polygon_layer(
        gdf_layer=df_socio_census,
        value_cols=socio_census_columns,
        method='weighted_mean',
        buffer_size=BUFFER_SIZE, 
        agg_method='weighted_mean'
    )

# Assign neighborhood data to nodes by buildings area
print("Assigning neighborhood data to nodes by buildings area")
if 'unemployment' not in node_assigner.node_attributes.columns:
    # Prepare population data from census sections
    df_unemployment = gpd.GeoDataFrame(df_socio_nei[['neighborhood', 'geometry', 'unemployment', 'unemployment_percentage']].copy())
    
    # Assign population from census sections to nodes via buildings
    node_assigner.assign_count_data_using_residential_buildings(
        buildings_gdf=bcn_buildings,
        source_gdf=df_unemployment,
        count_column='unemployment',
        area_id_column='neighborhood',
        weight_by='area',  # Use building area for more accurate population distribution
        aggregation='sum',  # Sum populations within buffer
        fillna_value=0,  # Areas with no buildings get 0 population
        point_buffer_size=10 # Buffer size for buildings with point geometries
    )
    del(df_unemployment)

if 'unemployment_percentage' not in node_assigner.node_attributes.columns:
    node_assigner.assign_polygon_layer(
        gdf_layer=df_socio_nei,
        value_cols=['unemployment_percentage'],
        method='weighted_mean',
        buffer_size=BUFFER_SIZE, 
        agg_method='weighted_mean'
    )
del(df_socio_nei)



# Assign remaining census section data to nodes by buildings area
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Get variables that need to be computed
print("Assigning census section data to nodes by buildings area")
vars_to_compute = [var for var in vars_to_assign_with_residential_buildings 
                  if var not in node_assigner.node_attributes.columns]

if vars_to_compute:
    print("Number of variables to compute: ", len(vars_to_compute))
    
    # Create partial function with fixed arguments
    assign_var = partial(
        node_assigner.assign_count_data_using_residential_buildings,
        buildings_gdf=bcn_buildings,
        source_gdf=df_socio_census,
        area_id_column='census_section',
        weight_by='area', 
        aggregation='sum',
        fillna_value=0,
        point_buffer_size=10
    )

    # Process variables in batches
    batch_size = 3  # Adjust this based on available RAM
    var_batches = np.array_split(vars_to_compute, len(vars_to_compute) // batch_size + 1)
    
    for batch_num, batch_vars in enumerate(var_batches, 1):
        print(f"\nProcessing batch {batch_num}/{len(var_batches)}")
        print(f"Variables in this batch: {', '.join(batch_vars)}")
        
        # Process batch in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            try:
                # Map each variable in the batch to the assign_var function
                list(executor.map(lambda var: assign_var(count_column=var), batch_vars))
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                raise
        
        print(f"Completed batch {batch_num}")

        # Save dataset
        node_assigner.save_node_attributes()

# Save dataset
node_assigner.save_node_attributes()
