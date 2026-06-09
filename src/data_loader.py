import sys
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely import wkt  # type: ignore
import osmnx as ox  # type: ignore
import yaml
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from paths import *

EPSG = 25831

def load_clean_bcn_population_data(root='./'):
    """
    Loads Barcelona's population data.
    
    Returns: 
        DataFrame: People counts categorized by census section, age and sex. 
    """
    ine_pop_file = f'{root}{RAW_INE}/ine_census_section_population.csv'
    df_pop = pd.read_csv(ine_pop_file, dtype={'census_section': str})
    df_pop.set_index('census_section', inplace=True)
    return df_pop


def load_bcn_census_sections(root='./'):
    """
    Loads Barcelona's census sections.
    
    Returns:
        GeoDataFrame: Polygons and identifier for the census sections of Barcelona
    """
    shp_file = f'{root}/{RAW_INE}/seccionado_2022/SECC_CE_20220101.shp'
    census_sections = gpd.read_file(shp_file)
    census_sections = census_sections[(census_sections['NPRO'] == 'Barcelona') & (census_sections['CMUN'] == '019')]
    census_sections = census_sections[['CUSEC', 'geometry']]
    census_sections.rename(columns={'CUSEC': 'census_section'}, inplace=True)
    census_sections.set_index('census_section', inplace=True)
    return census_sections


def load_bcn_neighborhoods(root='.'):
    """
    Load and process the Barcelona neighborhoods dataset.

    Returns:
        GeoDataFrame: A GeoDataFrame with neighborhoods in EPSG:25831 (ETRS89).
    """
    # Load CSV
    file_path=f'{root}/{RAW_BCN}/BarcelonaCiutat_Barris.csv'
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    drop_cols = ['codi_districte', 'nom_districte', 'codi_barri', 'geometria_wgs84']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Convert WKT to geometry
    df.rename(columns={
        'geometria_etrs89': 'geometry',
        'nom_barri': 'neighborhood'
        }, inplace=True) # ETRS89 == EPSG:25831
    df['geometry'] = df['geometry'].apply(wkt.loads)


    # Convert to GeoDataFrame
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:25831")


def load_bcn_districts(root='.'):
    """
    Loads Barcelona's districts.

    Returns:
        GeoDataFrame: A GeoDataFrame with districts in EPSG:25831 (ETRS89).
    """
    file_path=f'{root}/{RAW_BCN}/BarcelonaCiutat_Districtes.csv'
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    drop_cols = ['geometria_wgs84']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Convert WKT to geometry
    df.rename(columns={
        'geometria_etrs89': 'geometry',
        'nom_districte': 'district'
        }, inplace=True) # ETRS89 == EPSG:25831
    df['geometry'] = df['geometry'].apply(wkt.loads)

    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:25831")


def load_bcn_boundary(remove_top_left_part=True):
    """
    Downloads the boundary of Barcelona using OSMnx and transforms its coordinates
    to the one specified in the config file of the project.

    Returns:
        GeoDataFrame: The boundary of the location in the specified EPSG coordinates.
    """
    location = 'Barcelona, Barcelona, Catalunya, España'
    boundary = ox.geocode_to_gdf(location)
    boundary.to_crs(epsg=EPSG, inplace=True)

    if remove_top_left_part:
        # Remove a part of the city, since there is no bike network in Collserola
        gdf_exploded = boundary.explode(index_parts=False)
        to_reassemble = gdf_exploded.geometry.iloc[1:]  # Remove the empty part
        merged_geom = MultiPolygon([poly for poly in to_reassemble])
        new_gdf = gpd.GeoDataFrame(
            {'geometry': [merged_geom]},
            crs=boundary.crs
        )
    else:
        new_gdf = boundary
    
    return new_gdf


def load_bcn_building_data(root='./'):
    """
    Loads the residential buildings of Barcelona.
    Returns:
        GeoDataFrame: The boundary of the location in the config-specified EPSG coordinates.
    """
    file = f'{root}/{RAW_OSM}/Barcelona_Spain_residential_buildings.csv'
    df_buildings = pd.read_csv(file)
    df_buildings['geometry'] = df_buildings['geometry'].apply(wkt.loads)
    df_buildings = df_buildings[['element', 'id', 'geometry']]
    df_buildings = gpd.GeoDataFrame(data=df_buildings, geometry='geometry', crs=EPSG)
    return df_buildings

