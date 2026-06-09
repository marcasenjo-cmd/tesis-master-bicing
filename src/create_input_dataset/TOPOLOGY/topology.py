"""
This module handles the extraction and processing of topographic data that provides elevation data
and slope calculations that enable terrain-aware accessibility analysis.

The module processes high-resolution elevation grids to compute:
1. **Altitude Data**: Digital elevation model (DEM) extraction from external APIs
2. **Slope Calculations**: Gradient analysis for realistic cycling effort estimation
3. **Grid Generation**: Configurable resolution spatial grids for comprehensive coverage
4. **Terrain Analysis**: Slope metrics for routing optimization algorithms

Key Features:
- Configurable grid resolution for different analysis scales
- Batch altitude data fetching for efficient processing
- Advanced slope computation using elevation gradients
- Automatic coordinate system transformations
- File caching to avoid redundant API calls

Output Files:
- topology_altitude_slope_{resolution}m.csv: Combined altitude and slope dataset
- Configurable resolution based on project requirements

Author: Jordi Grau Escolano
"""

import os
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point  # type: ignore
import topology_helper as th

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import PR_INPUT, VISUALIZATIONS
import src.data_loader as dl


def download_altitude(boundary, resolution):
    """   
    This function creates a regular grid of points within the specified boundary
    and fetches elevation data for each point. It handles coordinate system
    transformations and ensures comprehensive coverage of the study area.
    
    Args:
        boundary (gpd.GeoDataFrame): The boundary of the location in any EPSG.
        resolution (float): Grid resolution in meters (e.g., 100 for 100m grid).
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing latitude, longitude, and altitude
        in EPSG:4326 coordinate system.
        
    Processing Steps:
    1. Transform boundary to WGS84 (EPSG:4326) for global elevation APIs
    2. Create regular grid of points within boundary
    3. Filter points to those within the boundary geometry
    4. Batch fetch elevation data for all grid points
    5. Validate data consistency and return results
        
    Technical Notes:
    - Grid resolution determines the density of elevation sampling
    - Higher resolution provides more detailed terrain analysis
    - Output is always in WGS84 for compatibility with elevation APIs
    - Grid points are filtered to ensure complete boundary coverage
    """
    # Transform the boundary to EPSG:4326 for grid creation and altitude queries
    boundary_4326 = boundary.to_crs(epsg=4326)
    bounds = boundary_4326.total_bounds  # (min_lon, min_lat, max_lon, max_lat)

    # Create a grid of points within the boundary
    grid_points = th.create_grid(bounds, resolution)
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in grid_points],
        crs="EPSG:4326"
    )

    # Filter grid points to those within the boundary
    points_within_boundary = points_gdf[points_gdf.geometry.within(boundary_4326.union_all())]
    coordinates = [(point.y, point.x) for point in points_within_boundary.geometry]

    # Fetch altitude data for the filtered points based on the OpenTopoData API
    altitudes = th.get_altitude_batch(coordinates)
    if len(altitudes) != len(coordinates):
        raise ValueError("Mismatch between coordinates and fetched altitudes!")

    # Prepare results in EPSG:4326
    results_4326 = pd.DataFrame({
        "latitude": [coord[0] for coord in coordinates],
        "longitude": [coord[1] for coord in coordinates],
        "altitude": altitudes
    })

    # Convert results back to the original EPSG
    return gpd.GeoDataFrame(
        results_4326,
        geometry=gpd.points_from_xy(results_4326.longitude, results_4326.latitude),
        crs="EPSG:4326"
    )


def compute_slope(results_df):
    """   
    This function calculates both average and maximum slope values for each grid point
    using elevation gradients.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing latitude, longitude, and altitude columns
        
    Returns:
        pd.DataFrame: original DataFrame with additional columns:
        - avg_slope: Average slope in the surrounding area
        - max_slope: Maximum slope encountered in the vicinity
        
    Processing Details:
    - Uses helper functions for slope computation algorithms
    - Rounds results for improved readability and memory efficiency
    """
    slope_results = th.compute_avg_max_slope(results_df)

    # Round altitude and slope for better readability
    slope_results['altitude'] = round(slope_results['altitude'], 1)
    slope_results['max_slope'] = round(slope_results['max_slope'], 1)
    return slope_results


def create_altitude_and_slope_grid_dataset(grid_resolution):
    """   
    This function orchestrates the complete topography data processing pipeline,
    from elevation data extraction to slope computation and data export.

    Args:
        grid_resolution (float): Grid resolution in meters (e.g., 100 for 100m grid).
    
    Returns:
        None: A CSV file is saved with the combined altitude and slope data.
        
    Processing Pipeline:
    1. **Configuration Loading**: Extract resolution settings from config file
    2. **File Check**: Verify if processed data already exists
    3. **Data Download**: Fetch elevation data using configurable grid resolution
    4. **Slope Computation**: Calculate slope metrics for all grid points
    5. **Data Export**: Save comprehensive dataset to CSV file

    Output Format:
    The resulting CSV contains columns for latitude, longitude, altitude,
    average slope, and maximum slope, providing comprehensive terrain
    information for optimization algorithms.
    """
    # Get location and data resolution
    EPSG = 25831
    ALTITUDE_RESOLUTION = grid_resolution

    # Check if file already exists
    output_file = f"{PR_INPUT}/topology_altitude_slope_{ALTITUDE_RESOLUTION}.csv"
    if os.path.exists(output_file):
        print(f"\nAltitude-slope in {ALTITUDE_RESOLUTION}m resolution dataset already exists in {output_file}")
        return None

    # Process altitude and slope
    boundary = dl.load_bcn_boundary()
    altitude_df = download_altitude(boundary, ALTITUDE_RESOLUTION)

    altitude_df = gpd.GeoDataFrame(
        altitude_df,
        geometry=gpd.points_from_xy(altitude_df.longitude, altitude_df.latitude),
        crs="EPSG:4326"
    )

    altitude_slope_df = compute_slope(altitude_df)

    # Convert results back to original EPSG
    altitude_slope_gdf = gpd.GeoDataFrame(
        altitude_slope_df,
        geometry=gpd.points_from_xy(altitude_slope_df.longitude, altitude_slope_df.latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG)

    # Save and visualize
    altitude_slope_df.to_csv(output_file, index=False)
    print(f"\nAltitude and slope data saved in: {output_file}")
    altitude_slope_gdf['longitude'] = altitude_slope_gdf['geometry'].x
    altitude_slope_gdf['latitude'] = altitude_slope_gdf['geometry'].y
    th.visualize_altitude_and_slope(boundary, altitude_slope_gdf)



if __name__ == "__main__":

    grid_resolution = 50
    create_altitude_and_slope_grid_dataset(grid_resolution)

    

    

    

    