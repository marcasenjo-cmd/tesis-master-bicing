"""
This module assigns POI counts to nodes based on buffer intersections using a spatial join.

Author: Jordi Grau Escolano
"""

import pandas as pd  # type:ignore
import geopandas as gpd  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from scipy.stats import entropy  # type:ignore


def assign_pois_to_nodes(
    nodes_gdf: gpd.GeoDataFrame,
    pois_gdf: gpd.GeoDataFrame, 
    buffer_size: float,
    crs: int,
) -> pd.DataFrame:
    """
    Assigns POI counts to nodes based on buffer intersections using a spatial join.
    
    Args:
        nodes_gdf (GeoDataFrame): Nodes with point geometries
        pois_gdf (GeoDataFrame): POIs with point geometries and category columns
        buffer_size (float): Buffer size in meters
        crs (int): Target coordinate reference system

    Returns:
        DataFrame: Node POI counts for each category
    """
    # Ensure both geodataframes are in the same CRS
    nodes_gdf = nodes_gdf.to_crs(crs)
    pois_gdf = pois_gdf.to_crs(crs)
    
    # Get category columns (excluding non-category columns)
    category_cols = [col for col in pois_gdf.columns if col not in ['element', 'geometry']]
    
    # Create buffered nodes (but don’t store them in the original DataFrame)
    nodes_buffer = nodes_gdf.copy()
    nodes_buffer['geometry'] = nodes_buffer.geometry.buffer(buffer_size)
    
    # Perform spatial join to get POIs within each node buffer
    joined = gpd.sjoin(pois_gdf, nodes_buffer, predicate='intersects', how='inner')

    # Count occurrences of each category per node
    results = joined.groupby('node_id')[category_cols].sum()

    # Ensure all nodes are present (fill missing nodes with 0)
    results = results.reindex(nodes_gdf.index, fill_value=0)

    # Rename columns to match original format (n_category)
    results = results.add_prefix('n_')

    # Remove 'superpois' from category columns if it exists
    category_cols = [col for col in results.columns if col != 'superpois']

    # Compute the total count of POIs per node
    results['pois_total'] = results[[col for col in category_cols]].sum(axis=1)

    return results


def compute_poi_entropy(poi_counts_df: pd.DataFrame) -> pd.Series:
    """
    Computes the Shannon entropy for each node based on POI category distributions.

    Args:
        poi_counts_df (pd.DataFrame): DataFrame with POI counts per category for each node.

    Returns:
        pd.Series: Entropy values for each node.
    """
    # Ensure non-negative counts
    poi_counts_df = poi_counts_df.clip(lower=0)

    # Convert to probabilities
    poi_probs = poi_counts_df.div(poi_counts_df.sum(axis=1), axis=0)

    # Replace NaN values (caused by division by zero) with 0
    poi_probs = poi_probs.fillna(0)

    # Compute Shannon entropy, setting entropy = 0 where all POIs are zero
    poi_probs = poi_probs.apply(lambda row: entropy(row, base=2) if row.sum() > 0 else 0, axis=1)

    return poi_probs.round(2)









