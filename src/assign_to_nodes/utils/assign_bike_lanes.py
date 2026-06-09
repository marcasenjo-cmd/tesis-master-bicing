"""
This module assigns bike lanes to nodes based on a buffer distance and plots results.

Author: Jordi Grau Escolano
"""

import os
import sys
import pandas as pd  # type:ignore
import geopandas as gpd  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '../../..'))
sys.path.append(project_root)

import src.data_loader as dl
from paths import *

def assign_bike_lanes(
    nodes_gdf: pd.DataFrame,
    bike_lanes_gdf: gpd.GeoDataFrame,
    buffer_distance: float,
    root: str = ''
) -> gpd.GeoDataFrame:
    """
    Assigns bike lanes to nodes based on a buffer distance and plots results.

    Args:
        nodes_gdf (GeoDataFrame): GeoDataFrame with nodes (Point geometries).
        bike_lanes_gdf (GeoDataFrame): GeoDataFrame with bike lanes (LineString geometries).
        buffer_distance (float): Buffer distance in meters.
        root (str, optional): Root directory path prefix for saving visualization.
            Defaults to empty string.

    Returns:
    - GeoDataFrame: Updated nodes_gdf with a new column 'has_bike_lane' (True/False).
    """
    # Create buffer around nodes
    nodes_gdf["buffer"] = nodes_gdf.geometry.buffer(buffer_distance)

    # Check for intersection with bike lanes
    nodes_gdf["has_bike_lane"] = nodes_gdf["buffer"].apply(
        lambda buf: bike_lanes_gdf.intersects(buf).any()
    )

    # Plot results
    plot_file = f"{root}{VISUALIZATIONS}/has_bike_lanes.png"
    if not os.path.exists(plot_file):

        boundary = dl.load_bcn_boundary().to_crs(bike_lanes_gdf.crs)
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot bike lanes
        boundary.boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label='City Boundary')
        bike_lanes_gdf.plot(ax=ax, color='blue', linewidth=0.5, label='Bike Lanes')

        # Plot nodes that don't intersect with bike lanes
        nodes_no_bike = nodes_gdf[~nodes_gdf['has_bike_lane']]
        nodes_no_bike.plot(ax=ax, color='gray', alpha=0.3, markersize=2, label='Nodes without bike lanes')

        # Plot nodes that intersect with bike lanes
        nodes_with_bike = nodes_gdf[nodes_gdf['has_bike_lane']]
        nodes_with_bike.plot(ax=ax, color='red', markersize=5, label='Nodes with bike lanes')
        
        # Create legend and remove axis
        ax.legend()
        ax.axis('off')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    # Drop the buffer column
    nodes_gdf = nodes_gdf.drop(columns=["buffer"])

    return nodes_gdf


def assign_km_of_bike_lanes(
    nodes_gdf: gpd.GeoDataFrame,
    bike_lanes_gdf: gpd.GeoDataFrame,
    buffer_distance: float,
    root: str = ''
) -> gpd.GeoDataFrame:
    """
    Computes the kilometers of bike lanes that intersect with the buffer of the nodes.
    
    Args:
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing the nodes
        bike_lanes_gdf (gpd.GeoDataFrame): GeoDataFrame containing the bike lanes
        buffer_distance (float): Distance in meters to create a buffer around each node
        root (str, optional): Root directory path for saving visualizations. Defaults to ''.
        
    Returns:
        gpd.GeoDataFrame: Original nodes GeoDataFrame with an additional column 'km_bike_lanes'
                          representing the length of bike lanes intersecting with each node's buffer
    """    
    # Create a buffer around each node
    nodes_gdf['buffer'] = nodes_gdf.geometry.buffer(buffer_distance)
    
    # Initialize the column for bike lane kilometers
    nodes_gdf['km_bike_lanes'] = 0.0
    
    # For each node, find intersecting bike lanes and sum their lengths
    print("Calculating intersections and lengths...")
    for idx, node in nodes_gdf.iterrows():
        # Get bike lanes that intersect with the buffer
        intersecting_lanes = bike_lanes_gdf[bike_lanes_gdf.intersects(node['buffer'])]
        
        if len(intersecting_lanes) > 0:
            # Calculate the total length of intersecting bike lanes in kilometers
            # For each lane, get only the portion inside the buffer
            total_length = 0
            for _, lane in intersecting_lanes.iterrows():
                # Clip the bike lane to the buffer
                intersection = lane.geometry.intersection(node['buffer'])
                # Add the length of the intersection in kilometers
                total_length += intersection.length / 1000  # Convert from meters to kilometers
            
            # Assign the total length to the node
            nodes_gdf.loc[idx, 'km_bike_lanes'] = total_length
    
    # Plot results
    plot_file = f"{root}{VISUALIZATIONS}/has_bike_lanes_km.png"
    if not os.path.exists(plot_file):

        boundary = dl.load_bcn_boundary().to_crs(bike_lanes_gdf.crs)        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot bike lanes
        boundary.boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label='City Boundary')
        bike_lanes_gdf.plot(ax=ax, color='blue', linewidth=0.5, label='Bike Lanes')
        
        # Plot the nodes with color based on km_bike_lanes
        nodes_gdf.plot(ax=ax, column='km_bike_lanes', cmap='viridis', markersize=2, alpha=1, label='Nodes')
        
        plt.title(f'Bike Lanes Intersection with Node Buffers (Buffer: {buffer_distance}m)')
        plt.legend()
        
        # Save the visualization
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print some statistics
    total_bike_lanes = nodes_gdf['km_bike_lanes'].sum()
    max_bike_lanes = nodes_gdf['km_bike_lanes'].max()
    avg_bike_lanes = nodes_gdf['km_bike_lanes'].mean()
    
    print(f"Total bike lanes assigned: {total_bike_lanes:.2f} km")
    print(f"Maximum bike lanes for a node: {max_bike_lanes:.2f} km")
    print(f"Average bike lanes per node: {avg_bike_lanes:.2f} km")
    print(f"Nodes with bike lanes: {len(nodes_gdf[nodes_gdf['km_bike_lanes'] > 0])} ({len(nodes_gdf[nodes_gdf['km_bike_lanes'] > 0])/len(nodes_gdf)*100:.2f}%)")
    
    return nodes_gdf
    
