"""
This module distributes area-level count data (e.g., population, jobs, etc.) to buildings.

Author: Jordi Grau Escolano
"""

from typing import Optional, Union

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore


def distribute_counts_to_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    count_column: str,
    area_id_column: str,
    weight_by: str = 'equally',
    fillna_value: Optional[Union[int, float]] = None,
    crs: Optional[int] = None,
    buffer_point_buildings: bool = True,
    point_buffer_size: float = 10.0
) -> gpd.GeoDataFrame:
    """
    Distribute area-level count data (e.g., population, jobs, etc.) to buildings.

    Args:
        buildings_gdf (GeoDataFrame): GeoDataFrame of buildings; must include a 
            geometry column.
        source_gdf (GeoDataFrame): GeoDataFrame of source areas (e.g., census sections,
            neighborhoods) with count data.
        count_column (str): Name of the column containing the counts to distribute.
        area_id_column (str): Name of the column containing the area identifier 
            (e.g., 'census_section', 'neighborhood_id').
        weight_by (str): Distribution method; choose 'equally' or 'area'. Defaults to 'equally'.
        fillna_value (Union[int, float], optional): Value to fill missing count data. Defaults to None.
        crs (int, optional): CRS to which both GeoDataFrames should be reprojected.
        buffer_point_buildings (bool, optional): Whether to buffer point geometries. Defaults to True.
        point_buffer_size (float, optional): Buffer size in meters for buildings with point geometries. Defaults to 10.0.

    Returns:
        GeoDataFrame: The buildings GeoDataFrame with an additional column '{count_column}_assigned' 
            containing distributed count data.
    """
    
    # Required columns validation
    required_columns = {area_id_column, count_column, 'geometry'}
    missing_columns = required_columns - set(source_gdf.columns)
    if missing_columns:
        raise ValueError(f"The source_gdf is missing the following required column(s): " +
                         f"{', '.join(missing_columns)}")
    
    # Buildings columns validation
    if 'geometry' not in buildings_gdf.columns:
        raise ValueError("The buildings_gdf must include a 'geometry' column.")

    if weight_by not in ['equally', 'area']:
        raise ValueError("Unsupported weighting method. Choose 'equally' or 'area'.")
    
    # Make copies to avoid modifying the original data
    buildings = buildings_gdf.copy()
    source = source_gdf.copy()

    # Handle CRS
    if crs is not None:
        buildings = buildings.to_crs(crs)  # type: ignore
        source = source.to_crs(crs)  # type: ignore

    # Buffer point geometries if needed
    if buffer_point_buildings:
        buildings['geometry'] = buildings.apply(
            lambda row: row.geometry.buffer(point_buffer_size) 
            if row.geometry.geom_type == 'Point' 
            else row.geometry,
            axis=1
        )

    # Perform spatial join: assign areas to buildings
    joined = gpd.sjoin(buildings, source, how='left', predicate='within')
    if joined.empty:
        raise ValueError(f"No buildings intersect with any area in source_gdf.")

    # Distribute counts
    if weight_by == 'equally':
        # Count buildings per area
        bld_counts = joined.groupby(area_id_column).size().reset_index(name='bld_count')
        joined = joined.merge(bld_counts, on=area_id_column, how='left')
        output_col = f'{count_column}_assigned'
        joined[output_col] = joined[count_column] / joined['bld_count']
        joined.drop(['bld_count', count_column, 'index_right'], axis=1, inplace=True)
        
    elif weight_by == 'area':
        # Compute building area and sum area per area
        joined['building_area'] = joined.geometry.area
        area_sums = joined.groupby(area_id_column)['building_area'].sum().rename('sum_area')
        joined = joined.join(area_sums, on=area_id_column)
        output_col = f'{count_column}_assigned'
        joined[output_col] = (joined[count_column] * joined['building_area'] / joined['sum_area'])
        joined.drop(['building_area', count_column, 'index_right'], axis=1, inplace=True)

    # Fill NaN values if specified
    if fillna_value is not None:
        joined[output_col] = joined[output_col].fillna(fillna_value)

    return joined


def assign_building_counts_to_nodes(
    buildings_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    count_column: str,
    aggregation: str = 'sum',
    buffer_size: float = 300.0,
    fillna_value: Optional[Union[int, float]] = None,
    crs: Optional[int] = None
) -> pd.DataFrame:
    """
    Assigns aggregated building count data to nodes based on spatial intersection.
    For polygon buildings, counts are distributed proportionally to intersection area.
    For point buildings, counts are assigned to nodes whose buffers contain the point.
    Handles mixed geometry types by processing points and polygons separately.

    Args:
        buildings_gdf (GeoDataFrame): GeoDataFrame of buildings with a count column.
        nodes_gdf (GeoDataFrame): GeoDataFrame of node points (must have an index that serves as 'node_id').
        count_column (str): Name of the column containing the count data to aggregate.
        aggregation (str): Specifies how the count data is aggregated at the node level (e.g., 'sum').
        buffer_size (float): The buffer (in meters) to apply around each node.
        fillna_value (Union[int, float], optional): Value to fill missing aggregated data.
        crs (int, optional): CRS to which both GeoDataFrames should be reprojected.

    Returns:
        DataFrame: A DataFrame indexed by node identifier containing the aggregated count values.
    """
    # Validate inputs
    if count_column not in buildings_gdf.columns:
        raise ValueError(f"Count column '{count_column}' not found in buildings_gdf")
    
    # Make copies to avoid modifying the original data
    buildings = buildings_gdf.copy()
    nodes = nodes_gdf.copy()

    # Handle CRS
    if crs is not None:
        buildings = buildings.to_crs(crs)  # type: ignore
        nodes = nodes.to_crs(crs)  # type: ignore
    
    # Check geometry types
    building_types = buildings.geometry.type.unique()
    if not all(t in ['Point', 'Polygon', 'MultiPolygon'] for t in building_types):
        raise ValueError("Building geometries must be Points, Polygons, or MultiPolygons")
    
    # Buffer nodes
    buffered_nodes = nodes.reset_index()
    buffered_nodes['geometry'] = buffered_nodes['geometry'].buffer(buffer_size)

    # Split buildings into points and polygons
    point_buildings = buildings[buildings.geometry.type == 'Point'].copy()
    poly_buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    
    results = []
    
    # Process points if they exist
    if not point_buildings.empty:
        point_intersection = gpd.sjoin(point_buildings, buffered_nodes, how='left', predicate='within')
        point_intersection['count_weighted'] = point_intersection[count_column]
        results.append(point_intersection)
    
    # Process polygons if they exist
    if not poly_buildings.empty:
        poly_buildings['total_area'] = poly_buildings.geometry.area
        poly_intersection = gpd.overlay(poly_buildings, buffered_nodes, how='intersection')
        poly_intersection['intersection_area'] = poly_intersection.geometry.area
        poly_intersection['proportion'] = poly_intersection['intersection_area'] / poly_intersection['total_area']
        poly_intersection['count_weighted'] = poly_intersection[count_column] * poly_intersection['proportion']
        results.append(poly_intersection)

    # Combine results
    if not results:
        raise ValueError("No buildings to process")
    
    intersection = pd.concat(results, ignore_index=True)

    # Aggregate the counts for each node
    agg_result = intersection.groupby('node_id')['count_weighted'].agg(aggregation)
    agg_result = agg_result.rename(count_column)
    result_df = pd.DataFrame(agg_result).reindex(nodes_gdf.index, fill_value=None)

    if fillna_value is not None:
        result_df.fillna(fillna_value, inplace=True)

    return result_df

