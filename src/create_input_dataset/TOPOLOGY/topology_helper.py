"""
This module provides utility functions for processing topological and elevation data.
It handles terrain analysis, slope calculations, and spatial grid operations for urban analysis.

The module includes functions for:
1. **Grid Generation**: Creating regular spatial grids for analysis
2. **Elevation Data Processing**: Fetching and processing altitude data from OpenTopoData API
3. **Slope Calculations**: Computing terrain slopes between points and grid cells
4. **Spatial Analysis**: Analyzing elevation patterns within geographic boundaries
5. **Visualization**: Creating maps of altitude and slope data

Data Sources:
- **OpenTopoData API**: SRTM90m global elevation data
- **Geographic Boundaries**: Custom boundary files for analysis regions

Technical Notes:
- Coordinates must be in EPSG:4326 (WGS84) for proper calculations
- Grid resolution is specified in meters and converted to degrees
- Slope calculations use great circle distance for horizontal measurements
- API rate limiting is implemented to respect service constraints

Author: Jordi Grau Escolano
"""

import sys
import requests
import time
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl


def create_grid(bounds, resolution):
    """   
    This function generates a uniform grid of latitude-longitude coordinates
    within the given geographic bounds. The grid resolution is specified in
    meters and automatically converted to appropriate degree increments for
    both latitude and longitude directions.
    
    Args:
        bounds (tuple): Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
        resolution (float): Grid resolution in meters.
        
    Returns:
        list of tuple: A list of (latitude, longitude) pairs representing the grid points.
        
    Technical Notes:
        - Uses approximate conversion factors: 111,320 m/degree for longitude, 110,540 m/degree for latitude
        - Grid points are generated in row-major order (latitude first, then longitude)
        - The actual grid spacing may vary slightly due to the spherical nature of Earth's surface
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    # Approx. meters per degree longitude
    lon_points = np.arange(min_lon, max_lon, resolution / 111320)
    # Approx. meters per degree latitude
    lat_points = np.arange(min_lat, max_lat, resolution / 110540)  
    grid_points = [(lat, lon) for lat in lat_points for lon in lon_points]
    return grid_points


def get_altitude_batch(coordinates, batch_size=100, api_url=\
                       "https://api.opentopodata.org/v1/srtm90m"):
    """   
    This function fetches altitude information from the OpenTopoData service, which provides
    access to SRTM90m global elevation data. It processes coordinates in batches to optimize
    API usage and implements rate limiting to respect service constraints.
    
    Args:
        coordinates (list of tuple): A list of (latitude, longitude) pairs.
        batch_size (int): The number of coordinates to process per API request (default: 100).
        api_url (str): The API endpoint for retrieving altitude data (default: SRTM90m).
        
    Returns:
        list: A list of altitude values corresponding to the input coordinates.
        
    Data Source:
        - OpenTopoData API: https://api.opentopodata.org/v1/srtm90m
        
    Technical Notes:
        - Implements 0.5 second delay between API requests to respect rate limits
        - Handles API errors gracefully by returning None for failed requests
        - Batch processing reduces the total number of HTTP requests
        - SRTM90m provides 90-meter resolution elevation data globally
    """
    results = []
    for i in range(0, len(coordinates), batch_size):
        batch = coordinates[i:i + batch_size]
        query = "|".join(f"{lat},{lon}" for lat, lon in batch)
        response = requests.get(f"{api_url}?locations={query}")
        if response.status_code == 200:
            data = response.json()
            results.extend([res.get("elevation", None) for res in data.get(\
                "results", [])])
        else:
            results.extend([None] * len(batch))
        time.sleep(0.5)
    return results


def calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2):
    """   
    This function computes the terrain slope between two points using their
    geographic coordinates and elevations. The slope is calculated as the ratio
    of vertical distance to horizontal distance, expressed as a percentage.
    
    Args:
        lat1, lon1, alt1 (float): Latitude, longitude, and altitude of the first point.
        lat2, lon2, alt2 (float): Latitude, longitude, and altitude of the second point.
        
    Returns:
        float: Slope percentage (positive for uphill, negative for downhill).
        
    Technical Notes:
        - Uses great circle distance calculation for horizontal distance
        - Earth radius approximation: 6,371,000 meters
        - Returns NaN if either altitude is missing
        - Returns 0% slope for identical points (avoids division by zero)
        - Slope calculation: (vertical_distance / horizontal_distance) * 100
    """
    if pd.isna(alt1) or pd.isna(alt2):
        return np.nan

    R = 6371000  # Radius of Earth in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Horizontal distance in meters
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    horizontal_distance = R * c

    if horizontal_distance == 0:  # Avoid division by zero
        return 0  # Assign slope 0% for identical points

    # Vertical distance
    vertical_distance = alt2 - alt1
    return (vertical_distance / horizontal_distance) * 100



def compute_avg_max_slope(df):
    """
    Compute the average and maximum slope for each grid point using its 4 neighbors.
    
    This function analyzes a regular grid of elevation data to calculate terrain
    slopes at each grid point. For each point, it computes slopes to its four
    cardinal neighbors (north, south, east, west) and calculates both the average
    and maximum slope values.
    
    Args:
        df (DataFrame): DataFrame containing latitude, longitude, and altitude data.
                       Must have columns: 'latitude', 'longitude', 'altitude'.
                       Coordinates must be in EPSG:4326 (WGS84).
        
    Returns:
        DataFrame: DataFrame with latitude, longitude, altitude, avg_slope, and max_slope.
        
    Processing Steps:
        1. Pivots the input data into a regular grid format
        2. For each grid point, calculates slopes to its 4 neighbors
        3. Computes average and maximum slopes for each point
        4. Handles edge cases (grid boundaries, missing data)
        5. Returns results in the original coordinate format
        
    Technical Notes:
        - Grid must be regular (uniform spacing in both directions)
        - Missing elevation data results in NaN slope values
        - Edge grid points have fewer neighbors (3 instead of 4)
        - Slope calculations use the calculate_slope_between_2_points function
        
    Raises:
        ValueError: If the grid is empty after pivoting or if coordinates are invalid.
    """
    # Pivot the DataFrame into a grid
    grid = df.pivot(
        index='latitude', columns='longitude', values="altitude").sort_index(
            ascending=False)

    if grid.empty:
        raise ValueError("The grid is empty after pivoting. Ensure latitude and"\
                         " longitude values are valid.")

    avg_slopes = []
    max_slopes = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            slopes = []

            lat1, lon1, alt1 = grid.index[i], grid.columns[j], grid.iloc[i, j]
            if pd.isna(alt1):
                avg_slopes.append(np.nan)
                max_slopes.append(np.nan)
                continue

            if i > 0:  # North
                lat2, lon2, alt2 = grid.index[i - 1], lon1, grid.iloc[i - 1, j]
                slopes.append(calculate_slope_between_2_points(
                    lat1, lon1, alt1, lat2, lon2, alt2))

            if i < grid.shape[0] - 1:  # South
                lat2, lon2, alt2 = grid.index[i + 1], lon1, grid.iloc[i + 1, j]
                slopes.append(calculate_slope_between_2_points(
                    lat1, lon1, alt1, lat2, lon2, alt2))

            if j > 0:  # West
                lat2, lon2, alt2 = lat1, grid.columns[j - 1], grid.iloc[i, j - 1]
                slopes.append(calculate_slope_between_2_points(
                    lat1, lon1, alt1, lat2, lon2, alt2))

            if j < grid.shape[1] - 1:  # East
                lat2, lon2, alt2 = lat1, grid.columns[j + 1], grid.iloc[i, j + 1]
                slopes.append(calculate_slope_between_2_points(
                    lat1, lon1, alt1, lat2, lon2, alt2))

            # Handle slopes and assign 0% if all neighbors are invalid
            slopes = [s for s in slopes if not pd.isna(s)]
            avg_slopes.append(np.mean(slopes) if slopes else 0)
            max_slopes.append(np.max(slopes) if slopes else 0)

    # Flatten the grid back into a DataFrame
    results_df = pd.DataFrame({
        "latitude": np.repeat(grid.index, grid.shape[1]),
        "longitude": np.tile(grid.columns, grid.shape[0]),
        "altitude": grid.values.flatten(),
        "avg_slope": avg_slopes,
        "max_slope": max_slopes
    }).sort_values(['latitude', 'longitude'])

    return results_df



def visualize_altitude_and_slope(boundary, df):
    """   
    This function creates a side-by-side visualization showing both elevation
    and maximum slope patterns within the specified geographic boundary. The
    visualization uses scatter plots with color-coded values and includes
    colorbars for interpretation.
    
    Args:
        boundary (GeoDataFrame): The geographic boundary of the analysis area.
        df (DataFrame): DataFrame containing latitude, longitude, altitude, and slope data.
        
    Returns:
        None: Saves a visualization file to the VISUALIZATIONS directory.
        
    Visualization Features:
        - Left panel: Elevation data with viridis colormap
        - Right panel: Maximum slope data with viridis colormap
        - Geographic boundary overlaid as black outline
        - Color-coded scatter plots with 5-pixel markers
        - Colorbars with appropriate labels and units
        - High-resolution output (300 DPI) for publication quality
        
    Output:
        - File: {VISUALIZATIONS}/altitude_slope_map.png
        - Format: PNG image with 15x10 inch dimensions
        - Resolution: 300 DPI for high-quality output
    """
    fig, axs = plt.subplots(1,2,figsize=(15, 10))
    
    boundary.plot(ax=axs[0], color='none', edgecolor='black', linewidth=1)
    boundary.plot(ax=axs[1], color='none', edgecolor='black', linewidth=1)
    
    scatter1 = axs[0].scatter(
        df["longitude"], df["latitude"], c=df["altitude"], cmap="viridis", s=5, alpha=0.7)
    scatter2 = axs[1].scatter(
        df["longitude"], df["latitude"], c=df["max_slope"], cmap="viridis", s=5, alpha=0.7)
    
    axs[0].axis('off')
    axs[1].axis('off')
    
    plt.colorbar(scatter1, ax=axs[0], label="Altitude (m)", shrink=0.6)
    plt.colorbar(scatter2, ax=axs[1], label="Slope (%)", shrink=0.6)
    plt.savefig(f"{VISUALIZATIONS}/raw_data/altitude_slope_map.png")