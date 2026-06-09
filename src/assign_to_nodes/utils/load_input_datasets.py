"""
This module loads the input datasets for the BSS optimization.

Author: Jordi Grau Escolano
"""

import sys
import os
import glob
from functools import reduce
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely import wkt  # type: ignore
from shapely.geometry import Point  # type: ignore

notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '../..'))
sys.path.append(project_root)

from paths import *
import src.data_loader as dl


def load_socioeconomic_census_section(root: str = ".") -> gpd.GeoDataFrame:
    """
    Loads and merges multiple socioeconomic datasets at the census section level.

    Args:
        root (str, optional): Root directory for data files. Defaults to "".

    Returns:
        gpd.GeoDataFrame: Merged socioeconomic dataset with census section geometries.
    """

    # Find all files 
    files = glob.glob(f"{root}/{PR_INPUT}/socioeconomic_census_section*")

    # Load dfs
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file, dtype={'census_section':str})
            if 'cars_abs' in df.columns:
                df.drop(columns=['population'], inplace=True)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Merge all DataFrames on 'census_section' using an outer join.
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="census_section", how="outer"), dataframes)

    # Clear dataset
    merged_df.drop(['income_2022_house'], axis=1, inplace=True)

    # Rename column
    merged_df.rename(columns={
    'total_population': 'population'
    }, inplace=True)

    # Add census section polygons
    cs = dl.load_bcn_census_sections(root=root).reset_index()
    merged_df = merged_df.merge(cs, on='census_section', how='outer')
    merged_df = gpd.GeoDataFrame(merged_df, crs=25830)
    merged_df.to_crs(25831, inplace=True)

    return merged_df


def load_socioeconomic_neighborhood(crs: str = "EPSG:25831", root: str = ".") -> gpd.GeoDataFrame:
    """
    Loads socioeconomic data at the neighborhood level and assigns geometries.

    Args:
        crs (str, optional): Output CRS. Defaults to "EPSG:25831".
        root (str, optional): Root directory for data files. Defaults to "".

    Returns:
        gpd.GeoDataFrame: Merged socioeconomic dataset with neighborhood geometries.
    """
    # Load file
    file = f'{root}/{PR_INPUT}/socioeconomic_neighborhood_unemployment.csv'
    df_neigh = pd.read_csv(file)

    # Add neighborhood geometry
    neigh_geoms = f'{root}/{RAW_BCN}/BarcelonaCiutat_Barris.csv'
    df_geoms = pd.read_csv(neigh_geoms)
    df_geoms.drop(
        ['codi_districte', 'nom_districte', 'codi_barri', 'geometria_wgs84'], 
        axis=1, 
        inplace=True
    )

    # Translate
    df_geoms = df_geoms.rename(columns={
        'geometria_etrs89': 'geometry',
        'nom_barri': 'neighborhood'
    })[["neighborhood", "geometry"]]

    # Merged and convert to geodataframe
    df_merged = df_neigh.merge(df_geoms, on='neighborhood', how='outer')
    df_merged['geometry'] = df_merged['geometry'].apply(wkt.loads)
    gdf_merged = gpd.GeoDataFrame(df_merged, crs=crs)
    gdf_merged.to_crs(crs, inplace=True)

    return gdf_merged


def load_pt_infrastructure(crs: int = 25831, root: str = ".") -> list:
    """
    Loads public transport infrastructure data from CSV files and converts them into GeoDataFrames.

    Args:
        crs (int, optional): Coordinate Reference System for output. Defaults to 25831.
        root (str, optional): Root directory for data files. Defaults to "".

    Returns:
        list: List of GeoDataFrames for bus stops, metro entrances, and tram stops.
    """

    files = [f'{root}/{PR_INPUT}/infrastructure_bus_stops_lines.csv',
             f'{root}/{PR_INPUT}/infrastructure_metro_entrances_stops_lines.csv',
             f'{root}/{PR_INPUT}/infrastructure_tram_entrances_lines.csv']
    
    dfs = []
    for file in files:
        df = pd.read_csv(file, dtype={'census_section':str})

        # Convert lat/lon to geometry, convert to GeoDataFrame and set crs
        df["geometry"] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        df.drop(['lat', 'lon'], axis=1, inplace=True)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        gdf.to_crs(crs, inplace=True)
        dfs.append(gdf)

    return dfs


def load_bike_lanes(crs: int = 25831, root='.'):
    """
    Loads bike lane infrastructure data from CSV file and converts it to a GeoDataFrame.

    Args:
        crs (str, optional): Coordinate Reference System for output. Defaults to "EPSG:25831".
        root (str, optional): Root directory for data files. Defaults to ''.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing bike lane geometries with the specified CRS.
            The geometries are loaded from WKT format and converted to GeoDataFrame objects.
    """
    file = f'{root}/{PR_INPUT}/urbanism_bike_lanes_epsg_25831.csv'
    df = pd.read_csv(file)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:25831")
    gdf.set_crs(crs)
    return gdf[['geometry']]


def load_pois(crs: int = 25831, root='.'):
    """
    Loads OSM POIs data from CSV file and converts it to a GeoDataFrame.

    Args:
        crs (int, optional): Coordinate Reference System for output. Defaults to 25831.
        root (str, optional): Root directory for data files. Defaults to ''.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing POI data with the specified CRS.
            Contains category columns indicating POI types (e.g. restaurants, shops)
            as binary flags, and a geometry column with point locations.
    """
    file = f'{root}/{PR_INPUT}/urbanism_pois.csv'
    df = pd.read_csv(file)
    df.drop(['osmid', 'id'], axis=1, inplace=True)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs=25831)
    gdf.to_crs(crs, inplace=True)

    return gdf







