"""
This module handles the extraction and processing of bike lanes and POIs from OpenStreetMap.

The module provides comprehensive urban data processing including:
1. **Bike Lane Networks**: Download, processing, and visualization of cycling infrastructure
2. **POI Classification**: Categorization of urban amenities into different categories defined by the user in the `src/input/POIs_OSM_tags.py` file.
3. **Geographic Filtering**: City boundary-based data filtering for spatial accuracy
4. **Data Visualization**: Automated generation of maps and plots for analysis

Key Features:
- File caching to avoid redundant downloads
- OSM data extraction via osmnx library and classification using predefined tag dictionaries
- Automated visualization generation

Output Files:
- urbanism_bike_lanes_epsg_{epsg}.csv: Bike lane network data
- urbanism_pois.csv: Classified POI data with category indicators
- Multiple visualization files: Maps showing bike lanes and POI distributions

Author: Jordi Grau Escolano
"""

import os
import sys
from pathlib import Path
import numpy as np  # type: ignore
import osmnx as ox  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
import contextily as ctx  # type: ignore

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
from src.input.POIs_OSM_tags import *


################################################################################
############################### Bike lanes #####################################
################################################################################

def download_bike_lanes(city_name, epsg):
    """
    Download bike lanes of a specified city from OpenStreetMap and convert them to a specified EPSG.

    Args:
        city_name (str): The name of the city (e.g., "Barcelona, Spain").
        epsg (int): EPSG code to convert the bike lanes to (e.g., 25831 for UTM31N).
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the bike lanes in the specified EPSG,
        or None if file already exists or no bike lanes found.
        
    Processing Steps:
    1. Check for existing cached file to avoid re-downloading
    2. Extract bike lanes using OSM tags (highway=cycleway) and transform coordinates to target EPSG
    3. Save processed data to CSV file
    """
    # Check if file already exists
    if city_name == "Barcelona, Spain":
        output_file = f"{PR_INPUT}/urbanism_bike_lanes_epsg_{epsg}.csv"
    else:
        output_file = f"{PR_INPUT}/urbanism_bike_lanes_epsg_{epsg}_{city_name}.csv"
    
    if os.path.exists(output_file):
        print(f"\nBike lanes file already exists in {output_file}")
        return None
    
    print(f"Downloading bike lanes for {city_name}...")

    # Download bike lanes using osmnx
    bike_lanes = ox.features_from_place(
        city_name, 
        tags={"highway": "cycleway"}
    )

    if bike_lanes.empty:
        print(f"No bike lanes found for {city_name}.")
        return None

    # Convert to GeoDataFrame with the specified EPSG
    bike_lanes = bike_lanes.to_crs(epsg=epsg)

    # Save the bike lanes GeoDataFrame to a CSV file
    bike_lanes[['geometry']].to_csv(output_file, index=True)
    print(f"Bike lanes saved to {output_file}.")

    return bike_lanes

def plot_bike_lanes_and_boundary(boundary_gdf, bike_lanes_gdf, city_name):
    """  
    This function generates maps showing the spatial distribution
    of cycling infrastructure within a city boundary.
    
    Args:
        boundary_gdf (gpd.GeoDataFrame): GeoDataFrame containing the city boundary.
        bike_lanes_gdf (gpd.GeoDataFrame): GeoDataFrame containing the bike lanes.
        city_name (str): Name of the city for file naming and plot titles.
        
    Output:
        Saves a high-resolution PNG file showing the city boundary and bike lanes.
    """
    if city_name == "Barcelona, Spain":
        output_file = f"{VISUALIZATIONS}/raw_data/bike_lanes_plot.png"
    else:
        output_file = f"{VISUALIZATIONS}/raw_data/{city_name}_bike_lanes_plot.png"

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    boundary_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label='City Boundary')
    bike_lanes_gdf.plot(ax=ax, color='blue', linewidth=0.5, label='Bike Lanes')
    
    # Create a custom legend and remove axis
    custom_legend = [
        Line2D([0], [0], color='black', linewidth=1, label='City boundary'),
        Line2D([0], [0], color='blue', linewidth=0.5, label='Bike lanes')
    ]
    ax.legend(handles=custom_legend)
    ax.axis('off')

    ctx.add_basemap(ax, crs=boundary_gdf.crs, source=ctx.providers.CartoDB.Positron)
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}.")


################################################################################
################################# OSM POIs #####################################
################################################################################

def fetch_and_classify_osm_pois(location, tags, category_dicts):
    """
    This function performs comprehensive POI extraction and classification. 
    It processes OSM data to identify and categorize urban amenities.
    
    Args:
        location (str): Name of the city to extract POIs for (e.g., "Barcelona, Spain").
        tags (dict): Dictionary of OSM tags to filter POIs (e.g., {"amenity": True}).
        category_dicts (dict): Dictionary mapping category names to classification dictionaries.
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing POIs with their geometries and 
        binary category classifications (1 for category membership, 0 otherwise).
        
    Processing Pipeline:
    1. **Data Extraction**: Fetch POIs from OSM using specified tags
    2. **Classification**: Apply category dictionaries to classify each POI using the `src/input/POIs_OSM_tags.py` file.
    3. **Data Cleaning**: Remove POIs not belonging to any category
    4. **Visualization**: Generate maps for each category
    5. **Output**: Save classified data and visualizations
        
    Output Files:
    - urbanism_pois.csv: Main dataset with POI classifications
    - pois_all.png: Overview map showing all POI categories
    - pois_{category}.png: Individual category maps for detailed analysis
    """
    output_file = os.path.join(f"{PR_INPUT}/urbanism_pois.csv")
    if os.path.exists(output_file):
        print(f"\nClassified OSM POIs file already exists in {output_file}")
        return None

    # Fetch POIs
    print(f"Fetching POIs for {location}...")
    pois = ox.features_from_place(location, tags)
    pois = gpd.GeoDataFrame(pois, crs=4326)
    
    # Filter: Keep only geometries fully inside Barcelona's boundary
    bcn_boundary = dl.load_bcn_boundary()
    pois.to_crs(bcn_boundary.crs, inplace=True)
    pois = pois[pois.geometry.within(bcn_boundary.union_all())]

    # Select relevant columns (OSM tags + geometry)
    relevant_columns = list(tags.keys()) + ['geometry']
    df = pois[relevant_columns].copy()

    # Separate geometry column to avoid dtype conflicts
    geometry_col = df[['geometry']].copy()
    df = df.drop(columns=['geometry'])  # Drop geometry for classification

    # Create classification DataFrame
    classification_df = pd.DataFrame(index=df.index)
    classification_df["osmid"] = df.index  # Keep OSM IDs

    # Classify POIs using vectorized operations
    for category, category_dict in category_dicts.items():
        mask = np.zeros(len(df), dtype=bool)  # Boolean mask initialization
        for key, values in category_dict.items():
            if key in df.columns:
                mask |= df[key].isin(values)  # Vectorized membership check

        classification_df[category] = mask.astype(int)  # Convert boolean to 1/0

    # Merge back geometry
    classification_df = classification_df.merge(geometry_col, left_index=True, right_index=True)

    # Remove points that are not part of any category
    category_cols = [col for col in classification_df.columns if col not in ['element', 'osmid', 'geometry']]
    classification_df = classification_df[classification_df[category_cols].sum(axis=1) > 0]

    # Save results
    classification_df.to_csv(output_file, index=True)
    print(f"\tClassified POIs saved to {output_file}")

    # Convert to GeoDataFrame before plotting
    classification_df = gpd.GeoDataFrame(classification_df, geometry='geometry', crs=bcn_boundary.crs)

    # Plot results
    # Plot all POIs together
    plot_file = f"{VISUALIZATIONS}/pois_all.png"
    bcn_boundary = dl.load_bcn_boundary()
    
    fig, ax = plt.subplots(figsize=(20,20))
    
    # Plot boundary
    bcn_boundary.boundary.plot(ax=ax, color='black', linewidth=1, label='City boundary')
    
    # Plot POIs by category with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(category_dicts)))
    for (category, _), color in zip(category_dicts.items(), colors):
        mask = classification_df[category] == 1
        if mask.any():
            classification_df.loc[mask, 'geometry'].plot(
                ax=ax,
                markersize=20,
                label=category,
                color=color,
                alpha=0.6
            )
    
    # Adjust legend and layout
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot individual categories
    for category, _ in category_dicts.items():
        plot_file = f"{VISUALIZATIONS}/pois_{category}.png"
        
        fig, ax = plt.subplots(figsize=(20,20))
        
        # Plot boundary
        bcn_boundary.boundary.plot(ax=ax, color='black', linewidth=1, label='City boundary')
        
        # Plot POIs for this category
        mask = classification_df[category] == 1
        if mask.any():
            classification_df.loc[mask, 'geometry'].plot(
                ax=ax,
                markersize=20,
                label=category,
                color='red',
                alpha=0.6
            )
        
        # Adjust legend and layout
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    EPSG = 25831
    location = "Barcelona, Spain"

    if location == "Barcelona, Spain":
        boundary = dl.load_bcn_boundary().to_crs(EPSG)
    else:
        boundary = ox.geocode_to_gdf(location)
        boundary.to_crs(epsg=EPSG, inplace=True)

    # Download bike lanes and convert to the target EPSG
    bike_lanes_gdf = download_bike_lanes(location, EPSG)

    # Compute number of km of bike lanes
    bike_lanes_gdf['length'] = bike_lanes_gdf.length
    total_length = bike_lanes_gdf['length'].sum()
    print(f"Total length of bike lanes: {total_length/1000} km")

    # Plot the boundary and bike lanes
    if bike_lanes_gdf is not None:
        plot_bike_lanes_and_boundary(boundary, bike_lanes_gdf, location)

    # Define the city and relevant tags
    relevant_tags = {
        "amenity": True, 
        "building": True, 
        "craft": True,
        "healthcare": True,
        "historic": True,
        "landuse": True,
        'leisure': True,
        'natural': True, 
        'office': True,
        'shop': True,
        'sport': True,
        'tourism': True,
        'water': True,
        'waterway': True
    }

    # Define classification dictionaries
    DICT_CATEGORIES = {
        "health_care": DICT_HEALTH_CARE,
        "culture": DICT_CULTURE,
        "tourism": DICT_TOURISM,
        "recreation": DICT_RECREATION,
        "sport": DICT_SPORT,
        "economic_retail": DICT_ECONOMIC_RETAIL,
        "industrial": DICT_INDUSTRIAL,
        "green": DICT_GREEN_NATURE,
        "civic": DICT_CIVIC,
        "worship": DICT_WORSHIP,
        "education": DICT_EDUCATION,
        "superpois": DICT_SUPERPOIS
    }

    # Fetch and classify POIs
    fetch_and_classify_osm_pois(location, relevant_tags, DICT_CATEGORIES)

    
