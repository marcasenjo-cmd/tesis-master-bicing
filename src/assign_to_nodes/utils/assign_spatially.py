"""
This module assigns attributes to nodes based on a spatial join.

Author: Jordi Grau Escolano
"""

import os
import sys
from joblib import Parallel, delayed
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

file_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(file_dir, '../../..'))
sys.path.append(project_root)

from paths import *
import src.data_loader as dl

def compute_area_weighted_mean(
        gdf_left: gpd.GeoDataFrame,
        gdf_right: gpd.GeoDataFrame,
        join_index: str,
        value_col: str
    ) -> pd.DataFrame:
        """
        Compute area-weighted mean of `value_col` from gdf_right polygons
        for each geometry in gdf_left.
        
        Args:
            gdf_left: A GeoDataFrame (e.g., node buffers) with a geometry column.
                        Must have a column or index named `join_index`.
            gdf_right: A GeoDataFrame with polygon geometries and a numeric attribute `value_col`.
            join_index: The column name in gdf_left that you want to group by for aggregation.
            value_col: The polygon attribute to compute a weighted mean for.
        
        Returns:
            A DataFrame indexed by `join_index` with a single column `weighted_mean`.
        """
        # Ensure unique column names to avoid collisions in overlay
        gdf_left_ren = gdf_left.reset_index().rename(columns={'index': 'temp_left_index'})
        if join_index not in gdf_left_ren.columns:
            raise ValueError(f"join_index='{join_index}' not found in gdf_left")

        # Produces a geometry for the intersection + all columns from both dataframes.
        intersected = gpd.overlay(gdf_left_ren, gdf_right, how='intersection')

        # Compute intersection area and weighted value
        intersected['intersection_area'] = intersected.geometry.area
        intersected['weighted_val'] = intersected[value_col] * intersected['intersection_area']

        # Aggregate of (weighted_val) and total intersection_area for each gdf_left feature
        aggregated = intersected.groupby(join_index).agg(
            total_weighted_val=('weighted_val', 'sum'),
            total_area=('intersection_area', 'sum')
        )

        # Compute weighted mean
        aggregated['weighted_mean'] = aggregated['total_weighted_val'] / aggregated['total_area']

        return aggregated[['weighted_mean']]


def assign_polygon_layer_to_nodes(
    nodes_gdf: gpd.GeoDataFrame,
    node_attributes: pd.DataFrame,
    gdf_layer: gpd.GeoDataFrame,
    value_cols: list,
    method: str = 'within',
    buffer_size: float = 0,
    agg_method: str = 'first',
    fillna_value=None,
    n_jobs: int = -1,
    crs: int = None
) -> pd.DataFrame:
    """
    Assigns attributes from a polygon layer to nodes using a specified spatial predicate.
    Supports parallel processing for multiple attribute columns.

    Args:
        nodes_gdf (GeoDataFrame): GeoDataFrame of node points.
        node_attributes (DataFrame): DataFrame of node attributes (indexed by node id).
        gdf_layer (GeoDataFrame): GeoDataFrame of polygons; must include a geometry column and the columns in value_cols.
        value_cols (list): List of column names in gdf_layer to assign to nodes.
        method (str): Spatial predicate to use (e.g., 'within', 'intersects'). Defaults to 'within'.
        buffer_size (float): Buffer size in meters to apply to nodes prior to the spatial join. Defaults to 0.
        agg_method (str): Aggregation method if multiple polygons match a node.
                          Supported methods include 'first', 'min', 'max', 'mean', or 'weighted_mean'.
        fillna_value (Any, optional): Value to fill missing data in the assigned columns. Defaults to None.
        n_jobs (int): Number of parallel jobs to use. Defaults to -1 (all cores).
        crs (int, optional): The target CRS to which gdf_layer should be reprojected. If None, no reprojection occurs.

    Returns:
        DataFrame: The updated node_attributes DataFrame with additional columns corresponding to each value in value_cols.
    """
    # Validate gdf_layer
    if gdf_layer.empty:
        raise ValueError("The provided GeoDataFrame (gdf_layer) is empty.")
    for col in value_cols:
        if col not in gdf_layer.columns:
            raise ValueError(f"Column '{col}' is not present in gdf_layer.")
    if gdf_layer.crs is None:
        raise ValueError("The provided GeoDataFrame (gdf_layer) must have a defined CRS.")

    if crs is not None:
        gdf_layer = gdf_layer.to_crs(crs).copy()

    raw_data_dir = f'{VISUALIZATIONS}/raw_data'
    os.makedirs(raw_data_dir, exist_ok=True)
    for col in value_cols:
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_layer.plot(ax=ax, column=col, legend=True)
        fig.savefig(f'{raw_data_dir}/raw_{col}.png', dpi=300)
        ax.axis('off')
        plt.close(fig)

    # Create a node layer: buffer nodes if buffer_size is greater than 0
    if buffer_size > 0:
        node_layer = nodes_gdf.copy()
        node_layer['geometry'] = node_layer['geometry'].buffer(buffer_size)
    else:
        node_layer = nodes_gdf

    def process_column(col):
        """
        Processes a single attribute column and returns a Series indexed by node id.
        """
        if agg_method != 'weighted_mean':
            joined = gpd.sjoin(node_layer, gdf_layer, how='left', predicate=method)
            if joined.empty or joined[col].isna().all():
                raise ValueError(f"No nodes intersect with any polygon for column '{col}'.")
            summarized = joined.groupby(joined.index).agg({col: agg_method})
            return summarized[col].rename(col)
        else:
            # For weighted_mean, the node_layer must be buffered to have an area.
            if buffer_size <= 0:
                raise ValueError("Weighted mean aggregation requires a node buffer. Set buffer_size > 0.")
            # Reset index to retain node id in a column (here renamed to 'node_index')
            node_layer_ren = node_layer.reset_index().rename(columns={'node_id': 'node_index'})
            aggregated = compute_area_weighted_mean(
                gdf_left=node_layer_ren,
                gdf_right=gdf_layer,
                join_index='node_index',
                value_col=col
            )
            return aggregated['weighted_mean'].rename(col)

    # Process each value column in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_column)(col) for col in value_cols)

    # Merge the results into node_attributes
    for series in results:
        if series.name in node_attributes.columns:
            continue
        else:
            node_attributes = node_attributes.join(series, how='left')
        
    # Post-process: ensure each value column exists, fill NaNs, and round values if needed
    for col in value_cols:
        if col not in node_attributes.columns:
            node_attributes[col] = None
        if fillna_value is not None:
            node_attributes[col].fillna(fillna_value, inplace=True)
        # Optionally round numeric values (adjust precision as needed)
        node_attributes[col] = node_attributes[col].round(1)

    return node_attributes


def assign_pt_unique_lines_per_node(
    nodes_gdf: gpd.GeoDataFrame, 
    transport_gdf: gpd.GeoDataFrame,
    transport_mode: str,
    buffer_size: float,
    crs: int=25831,
    root:str=''
    ) -> pd.DataFrame:
    """
    Counts the number of unique lines of a public transport within a buffer around each node.

    Args:
        nodes_gdf (GeoDataFrame): GeoDataFrame of nodes with polygon geometries.
        transport_gdf (GeoDataFrame): GeoDataFrame of stops with a 'line' column and POINT geometries.
        transport_mode (str): name of the transport mode to name the plot file.
        buffer_size (float): Buffer radius (in meters if crs=25831) around each node.
        crs (int, optional): Coordinate reference system (CRS) to ensure consistency. Defaults to EPSG:25831.

    Returns:
        DataFrame: A DataFrame indexed by node_id with the count of unique lines.
    """
    # Create buffers around nodes
    buffers = nodes_gdf[['geometry']].buffer(buffer_size).reset_index()
    buffers.rename(columns={0: 'geometry'}, inplace=True)
    buffers = buffers.set_geometry("geometry").set_crs(crs)

    # Perform spatial join: find which bus stops fall within each node buffer
    stops_inside_nodes = gpd.sjoin(transport_gdf, buffers, how="inner", predicate="within")

    # Check if 'line' column contains only strings or lists of strings
    if stops_inside_nodes['line'].apply(lambda x: isinstance(x, str)).all():
        # Count unique bus lines per node
        unique_lines_per_node = stops_inside_nodes.groupby('node_id')['line'].nunique()
    else:
        # Extract unique lines from lists of strings
        stops_inside_nodes['line'] = stops_inside_nodes['line'].apply(lambda x: x if isinstance(x, list) else [x])
        stops_inside_nodes = stops_inside_nodes.explode('line')
        unique_lines_per_node = stops_inside_nodes.groupby('node_id')['line'].nunique()

    # Ensure all nodes are included, filling NaNs with 0
    unique_lines_per_node = unique_lines_per_node.reindex(nodes_gdf.index, fill_value=0).reset_index()

    # Plot unique lines per node 
    plot_file = f'{root}{VISUALIZATIONS}/unique_{transport_mode}_lines_per_node.png'
    if not os.path.exists(plot_file):
        fig, ax = plt.subplots(figsize=(20, 20))
        nodes_gdf['line'] = unique_lines_per_node['line'].values
        bcn_boundary = dl.load_bcn_boundary()
        bcn_boundary.boundary.plot(ax=ax)
        nodes_gdf.plot(ax=ax, column='line', cmap='Purples', legend=True, markersize=10,
            alpha=0.8, edgecolor='white')
        ax.axis('off')
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)

    return unique_lines_per_node