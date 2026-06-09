import sys
from pathlib import Path

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
from shapely.geometry import Point  # type: ignore

import topology_helper as th

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import PR_INPUT  # noqa: E402
import src.data_loader as dl  # noqa: E402


def download_altitude(boundary, resolution):
    """
    Create a regular grid within the boundary and fetch altitude per point.

    Args:
        boundary (gpd.GeoDataFrame): analysis boundary
        resolution (float): grid resolution in meters

    Returns:
        gpd.GeoDataFrame in EPSG:4326 with latitude, longitude, altitude, geometry
    """
    boundary_4326 = boundary.to_crs(epsg=4326)
    bounds = boundary_4326.total_bounds  # (min_lon, min_lat, max_lon, max_lat)

    grid_points = th.create_grid(bounds, resolution)
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in grid_points],
        crs="EPSG:4326",
    )

    points_within_boundary = points_gdf[
        points_gdf.geometry.within(boundary_4326.union_all())
    ].copy()

    coordinates = [(point.y, point.x) for point in points_within_boundary.geometry]
    print(f"Grid points total: {len(grid_points)}")
    print(f"Grid points within boundary: {len(coordinates)}")

    altitudes = th.get_altitude_batch(coordinates)

    if len(altitudes) != len(coordinates):
        raise ValueError("Mismatch between coordinates and fetched altitudes")

    results_4326 = pd.DataFrame(
        {
            "latitude": [coord[0] for coord in coordinates],
            "longitude": [coord[1] for coord in coordinates],
            "altitude": altitudes,
        }
    )

    non_null = results_4326["altitude"].notna().sum()
    print(f"Altitude non-null after download: {non_null}/{len(results_4326)}")

    return gpd.GeoDataFrame(
        results_4326,
        geometry=gpd.points_from_xy(results_4326.longitude, results_4326.latitude),
        crs="EPSG:4326",
    )


def compute_slope(results_df):
    """
    Compute avg_slope and max_slope from altitude grid.
    """
    slope_results = th.compute_avg_max_slope(results_df)

    slope_results["altitude"] = slope_results["altitude"].round(1)
    slope_results["avg_slope"] = slope_results["avg_slope"].round(3)
    slope_results["max_slope"] = slope_results["max_slope"].round(3)

    return slope_results


def create_altitude_and_slope_grid_dataset(grid_resolution, force_rebuild=False):
    """
    Build the altitude+slope CSV at the requested grid resolution.

    Args:
        grid_resolution (float): grid size in meters
        force_rebuild (bool): overwrite existing output if True
    """
    epsg_out = 25831
    resolution = grid_resolution

    output_file = Path(PR_INPUT) / f"topology_altitude_slope_{resolution}.csv"

    if output_file.exists() and not force_rebuild:
        print(f"\nAltitude-slope dataset already exists: {output_file}")
        return None

    boundary = dl.load_bcn_boundary()

    altitude_gdf = download_altitude(boundary, resolution)

    altitude_df = pd.DataFrame(
        {
            "latitude": altitude_gdf["latitude"],
            "longitude": altitude_gdf["longitude"],
            "altitude": altitude_gdf["altitude"],
        }
    )

    altitude_slope_df = compute_slope(altitude_df)

    print("Diagnostics before save:")
    print(f"rows total: {len(altitude_slope_df)}")
    print(f"altitude non-null: {altitude_slope_df['altitude'].notna().sum()}")
    print(f"avg_slope non-null: {altitude_slope_df['avg_slope'].notna().sum()}")
    print(f"max_slope non-null: {altitude_slope_df['max_slope'].notna().sum()}")

    altitude_slope_gdf = gpd.GeoDataFrame(
        altitude_slope_df,
        geometry=gpd.points_from_xy(
            altitude_slope_df["longitude"],
            altitude_slope_df["latitude"],
        ),
        crs="EPSG:4326",
    ).to_crs(epsg=epsg_out)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    altitude_slope_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nAltitude and slope data saved in: {output_file}")

    viz_gdf = altitude_slope_gdf.copy()
    viz_gdf["longitude"] = viz_gdf.geometry.x
    viz_gdf["latitude"] = viz_gdf.geometry.y
    th.visualize_altitude_and_slope(boundary, viz_gdf)

    return altitude_slope_df


if __name__ == "__main__":
    grid_resolution = 50
    create_altitude_and_slope_grid_dataset(grid_resolution=grid_resolution, force_rebuild=True)