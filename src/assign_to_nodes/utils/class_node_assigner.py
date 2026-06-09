"""
This module provides a comprehensive interface for assigning various types of spatial
data to network nodes, including socioeconomic indicators, infrastructure metrics,
and demographic variables. It supports multiple assignment methods and handles
complex spatial relationships between different data layers.

Author: Jordi Grau Escolano
"""

import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from typing import Optional, Union
import networkx as nx  # type: ignore

from .load_graph import (
    load_or_download_graph,
    save_node_attributes,
    load_node_attributes
)
from .assign_altitude import get_nodes_altitude
from .assign_bike_lanes import assign_bike_lanes, assign_km_of_bike_lanes
from .assign_spatially import assign_polygon_layer_to_nodes, assign_pt_unique_lines_per_node
from .assign_pois import assign_pois_to_nodes, compute_poi_entropy
from .assign_counts_using_buildings import (
    distribute_counts_to_buildings,
    assign_building_counts_to_nodes
)

class NodeAttributesAssigner:
    """   
    This class provides a comprehensive interface for assigning various types of spatial
    data to network nodes, including socioeconomic indicators, infrastructure metrics,
    and demographic variables. It supports multiple assignment methods and handles
    complex spatial relationships between different data layers.
    
    Key Features:
    - **Spatial Assignment**: Buffer-based and weighted mean assignments
    - **Building-Weighted Assignment**: Uses residential building footprints for accurate population distribution
    - **Public Transport**: Assigns unique transport lines to nodes
    - **POI Processing**: count POIs and computes entropy to obtain a measure of diversity
    - **File Management**: Automatic saving and loading of node attributes
    
    Assignment Methods:
    - **Weighted Mean**: For continuous variables using spatial buffers
    - **Building-Area Weighted**: For count data using residential building footprints
    - **Spatial Join**: For categorical and boundary-based data
    - **Unique Count**: For transport lines and infrastructure features
    
    Data Types Supported:
    - Socioeconomic indicators (population, education, income)
    - Infrastructure metrics (transport lines, bike lanes)
    - Points of interest (POIs) with entropy calculations
    - Building footprints and area-based weights
    - Elevation and terrain data
    
    Coordinate Systems:
    - Input: Configurable CRS (default: EPSG:25831)
    - Processing: Automatic coordinate transformations as needed
    - Output: Consistent CRS across all node attributes
    """
    G: nx.Graph
    nodes_gdf: gpd.GeoDataFrame
    node_attributes: pd.DataFrame
    location: str
    graph_path: str
    crs: int
    root: str
    buffer_size: float
    attr_file: str
    use_elevation: bool

    def __init__(self, location: str, graph_path: str, crs: int = 25831, root: str = './', buffer_size: float = 300, use_elevation: bool = False):
        """
        Initializes the NodeAttributesAssigner.

        Args:
            location (str): The study area location.
            graph_path (str): Path to the graph file.
            crs (int, optional): Coordinate Reference System. Defaults to EPSG:25831.
            root (str, optional): Root directory for file storage.
            buffer_size (float, optional): Buffer size (in meters) for spatial assignments.
            use_elevation (bool, optional): Whether to load a graph with elevation data. Defaults to False.
        """
        location = location.split(',')[0]
        self.location = location
        self.graph_path = graph_path
        self.crs = crs
        self.root = root
        self.buffer_size = buffer_size
        self.use_elevation = use_elevation
        
        # Define the attribute file name
        attr_file = f"node_attributes" if 'Barcelona' in location else f"node_attributes_{location}"
        self.attr_file = f"{self.root}/data/processed/nodes/{attr_file}.csv"

        # Load or download the graph and prepare node GeoDataFrame
        self.G, nodes_gdf = load_or_download_graph(location, graph_path, crs, root=self.root, use_elevation=use_elevation)
        self.nodes_gdf = nodes_gdf
        self.node_attributes = self.load_node_attributes()


    def save_node_attributes(self):
        """
        Saves the node attributes to a Parquet file.
        """
        save_node_attributes(self.node_attributes, self.attr_file)


    def load_node_attributes(self) -> pd.DataFrame:
        """
        Loads the node attributes from a Parquet file if available.

        Returns:
            pd.DataFrame: The loaded node attributes DataFrame.
        """
        return load_node_attributes(self.attr_file, self.nodes_gdf.index)


    def assign_count_data_using_residential_buildings(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        source_gdf: gpd.GeoDataFrame,
        count_column: str,
        area_id_column: str,
        weight_by: str = 'equally',
        aggregation: str = 'sum',
        fillna_value: Optional[Union[int, float]] = 0,
        buffer_point_buildings: bool = True,
        point_buffer_size: float = 10.0
    ) -> None:
        """
        Assigns count data (e.g., population, jobs) from source areas to nodes via buildings.

        Args:
            buildings_gdf (GeoDataFrame): Buildings dataset.
            source_gdf (GeoDataFrame): Source dataset containing count data (e.g., census sections, neighborhoods).
            count_column (str): Name of the column containing the count data to distribute.
            area_id_column (str): Name of the column containing the area identifier.
            weight_by (str, optional): Count distribution method ('equally' or 'area').
            aggregation (str, optional): Aggregation method ('sum', 'mean', etc.).
            fillna_value (Union[int, float], optional): Default value for missing data.
            buffer_point_buildings (bool, optional): Whether to buffer point geometries. Defaults to True.
            point_buffer_size (float, optional): Buffer size in meters for point geometries. Defaults to 10.0.
        """
        
        if count_column not in self.node_attributes.columns:
            
            # Distribute area-level count data to buildings
            buildings_with_counts = distribute_counts_to_buildings(
                buildings_gdf=buildings_gdf,
                source_gdf=source_gdf,
                count_column=count_column,
                area_id_column=area_id_column,
                weight_by=weight_by,
                fillna_value=fillna_value,
                crs=self.crs,
                buffer_point_buildings=buffer_point_buildings,
                point_buffer_size=point_buffer_size
            )
            
            # Aggregate building count data to nodes
            node_counts = assign_building_counts_to_nodes(
                buildings_gdf=buildings_with_counts,
                nodes_gdf=self.nodes_gdf,
                count_column=f"{count_column}_assigned",
                aggregation=aggregation,
                buffer_size=self.buffer_size,
                fillna_value=fillna_value,
                crs=self.crs
            )
            
            self.node_attributes[count_column] = round(node_counts[f"{count_column}_assigned"], 2).values

            del(buildings_with_counts, node_counts)


    def assign_polygon_layer(
        self, 
        gdf_layer: gpd.GeoDataFrame, 
        value_cols: list, 
        method: str = 'within', 
        buffer_size: float = 0, 
        agg_method: str = 'first', 
        fillna_value=None,
        n_jobs: int = -1
    ):
        """
        Assigns attributes from a polygon layer to nodes.

        Args:
            gdf_layer (GeoDataFrame): Polygon layer.
            value_cols (list): List of attribute columns to assign.
            method (str, optional): Spatial join method ('within' or 'intersects').
            buffer_size (float, optional): Buffer size for matching nodes.
            agg_method (str, optional): Aggregation method for multiple matches.
            fillna_value (Any, optional): Default value for missing assignments.
            n_jobs (int, optional): Number of parallel jobs (-1 for all available cores).
        """
        self.node_attributes = assign_polygon_layer_to_nodes(
            gdf_layer=gdf_layer,
            value_cols=value_cols,
            method=method,
            buffer_size=buffer_size,
            agg_method=agg_method,
            fillna_value=fillna_value,
            n_jobs=n_jobs,
            nodes_gdf=self.nodes_gdf,
            node_attributes=self.node_attributes,
            crs=self.crs
        )
    
    
    def assign_public_transport_unique_lines(
        self,
        transport_gdf: gpd.GeoDataFrame,
        transport_mode: str,
    ):
        """
        Assigns the number of unique public transport lines per node.

        Args:
            transport_gdf (GeoDataFrame): Transport stops dataset with a 'line' column and POINT geometries.
            transport_mode (str): Name of the transport mode (e.g., 'bus', 'metro') for file and column naming.
        """
        unique_lines_per_node = assign_pt_unique_lines_per_node(
            transport_gdf=transport_gdf,
            transport_mode=transport_mode,
            nodes_gdf=self.nodes_gdf,
            buffer_size=self.buffer_size,
            crs=self.crs,
            root=self.root
        )
        self.node_attributes[f'{transport_mode}_lines'] = unique_lines_per_node['line'].values


    def assign_altitude(self):
        """
        Assigns altitude data to each node.
        """
        nodes_alt = get_nodes_altitude(self.nodes_gdf, self.root)
        print(nodes_alt)
        self.node_attributes['altitude'] = nodes_alt['altitude'].values


    def assign_bike_lanes(self, bike_lanes_gdf):
        """
        Checks for bike lanes within the buffer distance of each node and 
        assigns a boolean value to indicate presence/absence in the node_attributes

        Args:
            bike_lanes_gdf (GeoDataFrame): GeoDataFrame containing bike lane geometries.
        """
        nodes_gdf = assign_bike_lanes(
            nodes_gdf=self.nodes_gdf,
            bike_lanes_gdf=bike_lanes_gdf,
            buffer_distance=self.buffer_size,
            root=self.root
        )
        self.node_attributes['has_bike_lane'] = nodes_gdf['has_bike_lane'].astype(int).values
    

    def assign_bike_lane_kms(self, bike_lanes_gdf):
        """
        Assigns the kilometers of bike lanes to each node.
        """
        nodes_gdf = assign_km_of_bike_lanes(
            nodes_gdf=self.nodes_gdf,
            bike_lanes_gdf=bike_lanes_gdf,
            buffer_distance=self.buffer_size,
            root=self.root
        )
        self.node_attributes['bike_lane_kms'] = nodes_gdf['km_bike_lanes'].values

    def assign_pois(self, pois_gdf):
        """
        Assigns counts of POIs by category to the nodes.

        Args:
            pois_gdf (GeoDataFrame): GeoDataFrame containing POI locations and categories.

        """
        # Counts of pois by amenity categories
        nodes_with_pois = assign_pois_to_nodes(
            nodes_gdf=self.nodes_gdf,
            pois_gdf=pois_gdf,
            buffer_size=self.buffer_size,
            crs=self.crs
        )
        for col in nodes_with_pois.columns:
            self.node_attributes[col] = nodes_with_pois[col].values

        # Entropy
        self.node_attributes['pois_entropy'] = compute_poi_entropy(nodes_with_pois)

        
    def get_node_dataframe(self) -> gpd.GeoDataFrame:
        """
        Returns the node attributes as a GeoDataFrame with geometries.

        Returns:
            GeoDataFrame: Node attributes merged with geometries.
        """
        df = self.node_attributes.join(self.nodes_gdf[['geometry']])
        return gpd.GeoDataFrame(df, geometry='geometry', crs=self.crs)
