"""
This module provides utility functions for processing socioeconomic and OSM data.
It handles data download, cleaning, and transformation operations.

The module includes functions for:
1. **Population Data Processing**: INE data download and demographic processing
2. **Building Data Processing**: OSM building download and classification
3. **Statistical Processing**: Demographic calculations and aggregations

Key Features:
- Data download and cleaning from INE and OSM sources
- Demographic data aggregation and standardization
- Building footprint processing and classification

Data Sources:
- **INE (Spanish Statistics)**: Population demographics and census data
- **OpenStreetMap**: Building footprints and urban infrastructure
- **Barcelona Open Data**: Municipal statistics and indicators

Author: Jordi Grau Escolano
"""

import os
import sys
from pathlib import Path
from pyaxis import pyaxis  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import osmnx as ox  # type: ignore
import matplotlib.pyplot as plt    # type: ignore
import yaml

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
from src.input.ages import AGE_RANGES
from src.input.OSM_tags import (
    NON_RESIDENTIAL_BUILDING_TAGS,
    NON_RESIDENTIAL_AMENITY_TAGS,
    JOB_AMENITY_TAGS,
    JOB_TOURISM_TAGS)
from src.data_loader import load_bcn_boundary

EPSG = 25831


################################################################################
##################### POPULATION DOWNLOAD AND CLEANING #########################
################################################################################

def download_population_data():
    """
    This function fetches population statistics from the Spanish National Statistics
    Institute (INE) and parses the proprietary .px format into a pandas DataFrame.
    The data includes population counts by census section, age groups, and sex.
    
    Returns:
        pd.DataFrame: DataFrame containing population data with columns:
        - sección: Census section identifier
        - sexo: Sex classification (Hombres, Mujeres, Ambos Sexos)
        - edad (grupos quinquenales): Age group classifications
        - DATA: Population count values
        
    Data Source:
    - INE Population Census: https://ine.es/pcaxisdl/t20/e245/p07/a2022/l0/0001.px
        
    Technical Notes:
    - Uses pyaxis library for .px file parsing
    - Handles ISO-8859-2 encoding for Spanish characters
    - Returns raw data requiring further processing and cleaning
        
    Raises:
        Exception: If download or parsing fails
    """
    url = 'https://ine.es/pcaxisdl/t20/e245/p07/a2022/l0/0001.px'
    try:
        # Download and parse the .px file
        px = pyaxis.parse(url, encoding='ISO-8859-2')
        print(f"\tSuccessfully downloaded INE population data")
    except Exception as e:
        print(f"\tError downloading the INE population data: {e}\n")
        return None
    data = px['DATA']
        
    # Save the data to a CSV file
    df = pd.DataFrame(data)
    return df 

def download_and_clean_population_data(ine_municipality_code):
    """  
    This function orchestrates the complete population data processing pipeline,
    from raw data download to clean, analysis-ready datasets. It handles age
    group mapping, sex categorization, and geographic filtering.
    
    Args:
        ine_municipality_code (str): INE municipality code (e.g., '08019' for Barcelona)
        
    Returns:
        None: The function saves the processed dataset to a CSV file.
        
    Processing Pipeline:
    1. **Data Download**: Fetch raw population data from INE
    2. **Age Mapping**: Convert quinquennial age groups to standardized ranges 
        using the `src/input/ages.py` file
    3. **Sex Standardization**: Map Spanish terms to English abbreviations to 'm' and 
        'f' for males and females respectively
    4. **Geographic Filtering**: Filter data to specific municipality
    5. **Data Cleaning**: Remove totals and invalid entries
    6. **File Export**: Save processed data to CSV
        
    Output File:
    - {RAW_INE}/ine_census_section_population.csv: Clean population dataset
    """
    output_filepath = f'{RAW_INE}/ine_census_section_population.csv'
    print(f"Downloading and processing INE population data")
    if os.path.exists(output_filepath):
        print(f"\tFile already exists: {output_filepath}.")
        return None

    df = download_population_data()

    # Change age ranges categories and group population according to the new age label
    def map_age_to_group(age):
        for new_age_label, ages_list in AGE_RANGES.items():
            if age in ages_list:
                    return new_age_label
    df['edad (grupos quinquenales)'] = df['edad (grupos quinquenales)'].apply(map_age_to_group)

    df.rename(columns={
        'sección': 'census_section',
        'sexo': 'sex',
        'edad (grupos quinquenales)': 'age',
        'DATA': 'population'
    }, inplace=True)

    # Column with the province of each census section
    df['province'] = df['census_section'].str[:2]
    df.loc[df['province'] == 'TO', 'province'] = 'TOTAL'
    df = df[df['census_section'] != 'TOTAL']

    # Filter by municipality
    df['municipality'] = df['census_section'].str[:5]
    df = df[df['municipality'] == ine_municipality_code]

    # Change sex strings
    sex_mapping = {'Hombres': 'm', 'Mujeres': 'f', 'Ambos Sexos': 'total'}
    df['sex'] = df['sex'].map(sex_mapping)    

    # Remove rows with total sex or total age
    df = df[df['sex'] != 'total']
    df = df[df['age'] != 'total']
    df = df[['census_section', 'sex', 'age', 'population']]

    # Sum the groups' population
    df['population'] = df['population'].apply(int)
    df = df.groupby(['census_section', 'sex', 'age']).sum().reset_index() 
    
    # Create the output directory if it doesn't exist and save the data
    Path(RAW_INE).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"\tINE population cleaned data saved in {output_filepath}")   

################################################################################
################### POPULATION, AGE & SEX TO CENSUS SECTIONS ###################
################################################################################

def compute_demographic_data(df_pop, demographic_variable, output_type='percentage'):
    """
    Compute either the absolute population or the percentage of population by a specified demographic variable for each census section.

    Parameters:
    - df_pop: DataFrame containing population data with columns for census_section, population, and the demographic variable.
    - demographic_variable: str, the column name of the demographic variable to compute data for (e.g., 'sex', 'age').
    - output_type: str, either 'percentage' or 'absolute'. Determines whether to return percentages or absolute population counts.

    Returns:
    - DataFrame with either the percentage or absolute population for each value of the demographic variable within each census section.
    """
    # Calculate total population for each census section
    df_pop['total_section_population'] = df_pop.groupby('census_section')['population'].transform('sum')

    # Group by census section and the specified demographic variable
    df_demographic_total = df_pop.groupby(['census_section', demographic_variable])['population'].sum().reset_index()

    if output_type == 'percentage':
        # Calculate the percentage for each demographic group within each census section
        df_demographic_total['percentage'] = df_demographic_total.groupby('census_section')['population'].transform(
            lambda x: round((x / x.sum()) * 100, 1)
        )
        # Return the pivoted DataFrame with percentages
        return df_demographic_total.pivot(index='census_section', columns=demographic_variable, values='percentage')
    
    elif output_type == 'absolute':
        # Return the pivoted DataFrame with absolute population counts
        return df_demographic_total.pivot(index='census_section', columns=demographic_variable, values='population')
    
    else:
        raise ValueError("Invalid value for 'output_type'. Choose either 'percentage' or 'absolute'.")



    

################################################################################
############################# OSM HOME BUILDINGS ###############################
################################################################################

def download_residential_buildings(location, save=True, save_fig=True, epsg=25831, root='./') -> gpd.GeoDataFrame:
    """
    This function downloads and filters residential buildings from OpenStreetMap.
    Uses OSM tags (building, amenity, man_made, office, power, shop) to filter
    out commercial, industrial, and amenity buildings. Applies two-tier filtering:
    first excludes non-residential types, then confirms residential use.
    
    Args:
        location: Geographic region (e.g., "Barcelona, Spain")
        save: Save to CSV file (default: True)
        save_fig: Generate visualization (default: True)
        epsg: Coordinate system (default: 25831)
        root: Root directory for outputs (default: './')
        
    Returns:
        GeoDataFrame with residential buildings or None if file exists
        
    Processing Steps:
    1. Check if file already exists (caching)
    2. Download building data from OSM with multiple tags 
    3. Apply two-tier filtering to identify residential buildings using 
        the tags and the `src/input/OSM_tags.py` file
    4. Clean and standardize building attributes
    5. Save to CSV and optionally generate visualization
        
    Outputs:
    - CSV file: {root}/{RAW_OSM}/{location}_residential_buildings.csv
    - Visualization: {root}/{VISUALIZATIONS}/residential_buildings/{location}_residential_buildings.png
    """
    location_str = location.replace(', ', '_')
    if 'Barcelona' in location:
        location_str = 'Barcelona_Spain'
    residential_buildings_filepath = f"{root}/{RAW_OSM}/{location_str}_residential_buildings.csv"
    residential_buildings_visualization_filepath = f"{root}/{VISUALIZATIONS}/residential_buildings/{location_str}_residential_buildings.png"

    if os.path.exists(residential_buildings_filepath):
        print(f"\tResidential buildings already exist in: {residential_buildings_filepath}")
        return None

    # Download all building footprints and convert to GeoDataFrame
    tags = {
        'building': True, 
        'building:condition': True,
        'amenity': True, 
        'man_made': True, 
        'office': True, 
        'power':True,
        'shop':True,
        }
    gdf_buildings = ox.features_from_place(location, tags=tags)
    gdf_buildings = gdf_buildings.to_crs(epsg=epsg)

    gdf_buildings = gpd.GeoDataFrame(gdf_buildings, geometry='geometry')

    # # Convert Polygons and MultiPolygons to their centroids, retain Points as is
    # gdf_buildings['geometry'] = gdf_buildings['geometry'].apply(
    #     lambda geom: geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom)

    # Filter out buildings via building tag
    gdf_residential_buildings = gdf_buildings[
        ~gdf_buildings['building'].isin(NON_RESIDENTIAL_BUILDING_TAGS)]
    
    # Filter out buildings via amenity and tourism tags
    gdf_residential_buildings = gdf_residential_buildings[
        ~gdf_residential_buildings['amenity'].isin(JOB_AMENITY_TAGS)]
    gdf_residential_buildings = gdf_residential_buildings[
        ~gdf_residential_buildings['amenity'].isin(NON_RESIDENTIAL_AMENITY_TAGS)]
    if 'tourism' in gdf_residential_buildings.columns:
        gdf_residential_buildings = gdf_residential_buildings[
            ~gdf_residential_buildings['tourism'].isin(JOB_TOURISM_TAGS)]
    
    # Filter residential buildings - keep buildings tagged as 'yes' with no other 
    # tags or names, or non-'yes' buildings with no other tags
    gdf_residential_buildings = gdf_residential_buildings[
        (
            (gdf_residential_buildings['building'] != 'yes') &
            (gdf_residential_buildings['man_made'].isna()) &
            (gdf_residential_buildings['office'].isna()) &
            (gdf_residential_buildings['power'].isna()) &
            (gdf_residential_buildings['shop'].isna()) &
            (gdf_residential_buildings['building:condition'].isna())
        ) |
        (
            (gdf_residential_buildings['building'] == 'yes') &
            (gdf_residential_buildings['man_made'].isna()) &
            (gdf_residential_buildings['office'].isna()) &
            (gdf_residential_buildings['power'].isna()) &
            (gdf_residential_buildings['shop'].isna()) &
            (gdf_residential_buildings['building:condition'].isna()) &
            (gdf_residential_buildings['name'].isna())
        )
    ]

    # Select relevant columns and rename them
    try:
        gdf_residential_buildings = gdf_residential_buildings[['building', 'amenity', 'addr:street', 
                                                               'addr:housenumber', 'addr:postcode', 
                                                               'name', 'geometry']]
    except:
        gdf_residential_buildings = gdf_residential_buildings[['building', 'amenity', 'addr:postcode', 
                                                               'name', 'geometry']]
    
    gdf_residential_buildings.rename(columns={
        'building': 'building', 
        'amenity': 'amenity',
        'man_made': 'man_made',
        'addr:street': 'street',
        'addr:housenumber': 'housenumber',
        'addr:postcode': 'postcode',
        'name': 'name',
        'geometry': 'geometry'}, inplace=True)


    if save:
        Path(RAW_OSM).mkdir(parents=True, exist_ok=True)
        gdf_residential_buildings.sort_values(by=['building', 'amenity'], inplace=True)
        gdf_residential_buildings.to_csv(residential_buildings_filepath)
        print(f"\tNumber of residential buildings: {len(gdf_residential_buildings)}")
        print(f"\tResidential buildings saved to {residential_buildings_filepath}")

    if save_fig:
        linewidth = 0.7
        markersize = 1
        bcn_boundary = load_bcn_boundary()

        _, ax = plt.subplots(figsize=(15, 15))
        gdf_residential_buildings.plot(ax=ax, markersize=markersize)
        bcn_boundary.boundary.plot(ax=ax, color='black', alpha=1, linewidth=linewidth)
        ax.set_axis_off()

        # North arrow
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
        ax.text(0.12, 0.20, 'N', transform=ax.transAxes, ha='center', 
            va='center', fontsize=fontsize, fontweight='bold', color='black')
        
        # Scale bar
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_center = xlim[0] + 0.12 * (xlim[1] - xlim[0])  # Center point
        x_start = x_center - 1000  # 1 km to left of center
        x_end = x_center + 1000    # 1 km to right of center
        y_scale = ylim[0] + 0.18 * (ylim[1] - ylim[0])  # Below arrow
        
        ax.plot([x_start, x_end], [y_scale, y_scale], color='black', linewidth=3, solid_capstyle='butt')
        ax.text(x_center, y_scale - 0.01 * (ylim[1] - ylim[0]), '2 km',
            ha='center', va='top', fontsize=fontsize, color='black', fontweight='bold')
        
        Path(VISUALIZATIONS).mkdir(parents=True, exist_ok=True)
        plt.savefig(residential_buildings_visualization_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\tResidential buildings visualization saved to {residential_buildings_visualization_filepath}")