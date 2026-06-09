"""
This module applies various statistical normalization methods to prepare data for 
optimization algorithms and generates visualizations.

The module provides:
1. **Data Normalization**: Multiple normalization strategies for different variable types
2. **Distribution Visualization**: Before/after plots for each normalization method
3. **Correlation Analysis**: Correlation matrices for feature selection

Normalization Methods:
- **minmax**: Scales data to range [0, 1] using min/max values
- **log**: Natural logarithm transformation for skewed distributions
- **boxcox**: Box-Cox transformation for non-normal distributions
- **robust**: Median and IQR-based normalization
- **zscore**: Standard score normalization
- **0-max**: Scales from 0 to maximum value
- **inverted_***: Inverted versions of all methods (1 - normalized_value)

Variable Groups:
- **Demographics**: Age groups, population, gender distribution
- **Economic**: Income, household size, unemployment
- **Education**: Primary, secondary, and college education levels
- **Infrastructure**: Public transport lines, bike lanes
- **POIs**: Points of interest counts and entropy
- **Transport**: Vehicle ownership and motorization

Output:
- Normalized dataset ready for optimization algorithms
- Comprehensive visualization plots
- Correlation analysis for feature selection

Author: Jordi Grau Escolano
"""

import sys
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from shapely.geometry import Point  # type: ignore

# Define project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.assign_to_nodes.utils.load_input_datasets as lid 
import src.assign_to_nodes.utils.class_node_assigner as cna
import src.assign_to_nodes.utils.class_dataset_normalizer as cdn

# Constants
LOCATION = 'Barcelona, Barcelona, Catalunya, España'
BUFFER_SIZE = 300
CRS = 25831
normalized_df_filepath =  f'{PR_NODES}/normalized_node_attributes.csv'

# Initialize Node Assigner and Normalizer
node_assigner = cna.NodeAttributesAssigner(LOCATION, graph_path=RAW_GRAPH, crs=CRS, buffer_size=BUFFER_SIZE)
normalizer = cdn.DatasetNormalizer(node_assigner.get_node_dataframe(), normalized_df_filepath)


# Normalization strategy for each variable
normalization_columns_map = {

    'minmax': [
        'tram_lines', 'bike_lane_kms',
        'cars_abs', 'motos_abs', 'unemployment_percentage',
    ],

    'log': [
        'bus_lines', 'metro_lines', 
        'n_health_care', 'n_culture', 'n_tourism', 'n_recreation', 'n_sport',
        'n_economic_retail', 'n_industrial', 'n_green', 'n_civic', 'n_worship',
        'n_education', 'n_superpois', 
    ],

    'boxcox': [
        '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+',
        'household_avg_m2', 'unemployment',
        'education_primary', 'education_secondary', 'education_college',
        'pois_total', 'pois_entropy', 
        'population','f','m','non_spanish_population',
    ],
    
    'robust': [],
    'zscore': [],
    '0-max':[],
    'inverted_boxcox': [
        'income_2022_pers'
    ]
}


# Normalize and plot raw and normalized distributions
plot_vars_groups = {
    'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'],
    'economic': ['household_avg_m2', 'income_2022_pers', 'unemployment'],
    'education': ['education_primary', 'education_secondary', 'education_college'],
    'pois1': ['n_health_care', 'n_culture', 'n_tourism', 'n_recreation', 'n_sport'],
    'pois2': ['n_economic_retail', 'n_industrial', 'n_green', 'n_civic', 'n_worship'],
    'pois3': ['n_education', 'n_superpois', 'pois_total', 'pois_entropy'],
    'public_transport': ['bus_lines', 'metro_lines', 'tram_lines', 'bike_lane_kms'],
    'socio': ['population', 'f', 'm', 'non_spanish_population'],
    'transport': ['cars_abs', 'motos_abs'],
}

normalizer.normalize_columns(
    columns_mapping=normalization_columns_map, 
    plot_multi=plot_vars_groups
)

# Plot raw and normalized correlation matrix
normalizer.plot_correlation_matrix()

# Drop highly correlated variables
normalizer.normalized_df.drop(['others', 'population_density'], axis=1, inplace=True)

# Save normalized node attributes
normalizer.save_normalized_data()










