"""
Public Transport Infrastructure Data Processing Pipeline

This module handles the extraction, processing, and cleaning of public transport data
from OpenStreetMap via the Overpass API. It processes three main transport modes:

1. **Bus**: Extracts bus stops and their associated route references
2. **Tram**: Processes tram stations and line connections
3. **Metro**: Handles metro entrances, stops, and line mappings with complex merging logic

The module includes data cleaning, boundary filtering for Barcelona, and comprehensive merging 
to create a clean, analysis-ready dataset. The initial queries are defined in the `src/input/PT_stations_queries.py` file.

Key Features:
- Overpass API integration for real-time OSM data
- Automatic file caching to avoid redundant downloads
- City boundary filtering for geographic accuracy
- Merging of metro entrances and station data
- Comprehensive data validation and cleaning

Output Files:
- infrastructure_bus_stops_lines.csv: Bus stops with lines
- infrastructure_tram_entrances_lines.csv: Tram entrances, stops, and lines
- infrastructure_metro_entrances_stops_lines.csv: Metro entrances, stops, and lines

Author: Jordi Grau Escolano
"""

import os
import time
import re
import sys
import requests
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point  # type: ignore

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from paths import *
import src.data_loader as dl
from src.input.PT_stations_queries import OVERPASS_URL, QUERIES

# Mapping for long-distance train services to standardized names
TRAIN_LINE_MAPPING = {
    'AVE': 'Long distance',
    'Alvia 00621': 'Long distance',
    'Alvia 00626': 'Long distance',
    'Alvia 00661': 'Long distance',
    'Alvia 00664': 'Long distance',
    'Iryo': 'Long distance',
    'Ouigo': 'Long distance',
    '9704': 'Long distance',  # Renfe-SNCF 9704
    '9713': 'Long distance',  # Renfe-SNCF
    '9736': 'Long distance',  # Renfe-SNCF
    '9743': 'Long distance',  # Renfe-SNCF
    '34': 'Long distance', # 34: Barcelona-Estació de França → Saragossa-Delicias
}


def fetch_overpass_data(query, max_retries=3, sleep_seconds=10):
    """
    Fetch data from OpenStreetMap via the Overpass API.
    
    Args:
        query (str): Overpass QL query string
        
    Returns:
        list: List of OSM elements (nodes, ways, relations)
        
    Raises:
        Exception: If the Overpass API request fails
    """
    headers = {
        "User-Agent": "bss-station-optimization/1.0 (academic-project)",
        "Referer": "https://github.com/Jordigres/bss_station_optimization",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=headers,
                timeout=180,
            )

            if response.status_code == 200:
                return response.json().get("elements", [])

            last_error = Exception(
                f"Overpass request failed on attempt {attempt}: "
                f"{response.status_code}\n{response.text}"
            )

        except requests.RequestException as e:
            last_error = e

        if attempt < max_retries:
            print(f"Retrying Overpass request in {sleep_seconds} seconds... "
                  f"(attempt {attempt}/{max_retries})")
            time.sleep(sleep_seconds)

    raise Exception(f"Overpass request failed after {max_retries} attempts: {last_error}")


def process_bus_stops(elements):
    """
    Extract and process bus stop data from OSM elements.
    
    This function processes bus stops and their associated route references,
    handling cases where multiple bus lines serve the same stop.
    
    Args:
        elements (list): List of OSM elements from Overpass API
        
    Returns:
        pd.DataFrame: DataFrame with columns: stop_id, stop_name, line, lat, lon
        
    Processing Steps:
    1. Extract bus stop nodes with route references
    2. Split multiple route references into separate rows
    3. Remove duplicates and sort by line and stop name
    """
    bus_stops = [
        {
            "stop_id": el["id"],
            "stop_name": el.get("tags", {}).get("name"),
            "line": el.get("tags", {}).get("route_ref"),
            "lat": el.get("lat"),
            "lon": el.get("lon")
        }
        for el in elements if el["type"] == "node" and el.get("tags", {}).get("highway") == "bus_stop"
    ]
    df = pd.DataFrame(bus_stops)

    # Split line into individual rows if they contain multiple lines
    df = df.assign(line=df["line"].str.split())  # Split line into lists
    df = df.explode("line").reset_index(drop=True)  # Expand lists into separate rows
    df = df.drop_duplicates().sort_values(['line', 'stop_name'])
    return df


def process_metro_and_tram_stops_and_lines(elements, pt_modes):
    """
    Extract subway and tram route members with comprehensive data processing.
    
    This function processes metro and tram data by extracting route relationships
    and their associated station nodes. It handles complex OSM relation structures
    and provides clean station-line mappings.
    
    Args:
        elements (list): List of OSM elements from Overpass API
        pt_modes (list or str): Transport modes to process (e.g., ['metro', 'subway', 'tram'])
        
    Returns:
        pd.DataFrame: DataFrame with station and line information
        
    Processing Steps:
    1. Extract route relations for specified transport modes
    2. Map station nodes to their route memberships
    3. Apply train line mapping for long-distance services
    4. Sort and filter results for Barcelona city boundary
    """
    nodes_data = [el for el in elements if el["type"] == "node"]
    relations = [el for el in elements if el["type"] == "relation"]

    # Ensure `pt_modes` is treated as a list (e.g., ['subway', 'train', 'light_rail', 'tram'])
    if isinstance(pt_modes, str):
        pt_modes = [pt_modes]

    # Extract route details for all specified modes
    pt_routes = {
        rel["id"]: {
            "line": rel.get("tags", {}).get("ref"),
            "members": rel.get("members", [])
        }
        for rel in relations
        if rel.get("tags", {}).get("route") in pt_modes
    }

    # Prepare node data with lat and lon
    nodes_data_dict = {node["id"]: (node.get("tags", {}).get("name"), node.get("lat"), node.get("lon")) 
                       for node in nodes_data}

    # Generate members DataFrame
    all_members_data = []
    for route_id, route_data in pt_routes.items():
        line = route_data["line"]
        members = route_data["members"]

        for member in members:
            if member["type"] == "node":
                member_id = member["ref"]
                node_details = nodes_data_dict.get(member_id, (None, None, None))
                stop_name, lat, lon = node_details

                if stop_name:  # If a stop name is found
                    all_members_data.append({
                        "line_id": route_id,
                        "line": line,
                        "stop_name": stop_name,
                        "lat": lat,
                        "lon": lon,
                    })

    all_members_df = pd.DataFrame(all_members_data)

    # Replace long-distance company names by 'Long-distance'
    all_members_df['line'] = all_members_df['line'].apply(
        lambda x: TRAIN_LINE_MAPPING.get(x, x))

    # Sort by line number and stop_name
    all_members_df['letter'] = all_members_df['line'].str.extract(r'([A-Za-z]+)', expand=False)
    all_members_df['number'] = all_members_df['line'].str.extract('(\d+)', expand=False)
    all_members_df['number'] = all_members_df['number'].fillna(0).astype(int)

    # Sort by letter first, then by number
    all_members_df = all_members_df.sort_values(
        by=['letter', 'number', 'stop_name', 'lat', 'lon'], 
        ascending=[True, True, True, True, True]
    ).drop(columns=['letter', 'number'])

    # Select only the stations inside the city boundary
    geometry = [Point(xy) for xy in zip(all_members_df['lon'], all_members_df['lat'])]
    stops_gdf = gpd.GeoDataFrame(all_members_df, geometry=geometry, crs="EPSG:4326")
    boundary_gdf = dl.load_bcn_boundary()
    boundary_gdf = boundary_gdf.to_crs(4326)
    stops_within_boundary = stops_gdf[stops_gdf.within(boundary_gdf.geometry.iloc[0])].copy()
    stops_within_boundary.drop('geometry', axis=1, inplace=True)
    
    # Drop duplicates
    stops_within_boundary = stops_within_boundary.drop_duplicates(
        subset=['line', 'stop_name', 'lat', 'lon'])
    stops_within_boundary.dropna(inplace=True)

    return stops_within_boundary
   

def process_metro_entrances_and_stops(elements):
    """Extract subway entrances and their corresponding stop areas."""
    nodes = [el for el in elements if el["type"] == "node"]
    relations = [el for el in elements if el["type"] == "relation"]

    # Map stop areas to their names
    stop_areas = {
        rel["id"]: {"name": rel.get("tags", {}).get("name")}
        for rel in relations
    }

    # Extract subway entrances and their corresponding stop areas
    entrances_data = []
    for node in nodes:
        if "tags" in node and node["tags"].get('railway') == 'subway_entrance':
            parent_relations = [
                rel for rel in relations
                if any(member["type"] == "node" and member["ref"] == node["id"] for member in rel.get("members", []))
            ]

            for parent in parent_relations:
                stop_area = stop_areas.get(parent["id"], {})
                stop_name = node["tags"].get("name")
                entrances_data.append({
                    "entrance_id": node["id"],
                    "entrance_name": stop_name,
                    "stop_name": stop_area.get("name"),
                    "lat": node.get("lat"),
                    "lon": node.get("lon")
                })
    
    df = pd.DataFrame(entrances_data)

    # Select only the stations inside the city boundary
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    entrances_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    boundary_gdf = dl.load_bcn_boundary()
    boundary_gdf = boundary_gdf.to_crs(4326)
    entrances_within = entrances_gdf[entrances_gdf.within(boundary_gdf.geometry.iloc[0])].copy()
    entrances_within.drop('geometry', axis=1, inplace=True)

    # Remove parentheses and their content from 'name_helper'
    entrances_within['name_helper'] = entrances_within['stop_name'].str.replace(r"\s*\(.*?\)", "", regex=True)

    # Fill missing values in entrance_name with an empty string
    entrances_within['entrance_name'] = entrances_within['entrance_name'].fillna('')

    # Replace missing 'entrance_name' values using 'name_helper'
    def assign_unique_names(group):
        group = group.reset_index(drop=False)

        # Identify rows with empty 'entrance_name'
        missing = group['entrance_name'] == ''
        
        # Assign unique names for missing 'entrance_name'
        group.loc[missing, 'entrance_name'] = [
            f"{row.name_helper}_{i+1}" for i, row in enumerate(group[missing].itertuples())
        ]
        return group

    # Apply the logic to each group
    entrances_within = entrances_within.groupby(
        'stop_name', group_keys=False)[[
            'stop_name', 'entrance_name', 'name_helper', 'entrance_id', 'lat','lon']
            ].apply(assign_unique_names)

    # Reset index to ensure stop_name is preserved
    entrances_within.reset_index(drop=True, inplace=True)

    # Sort and print
    entrances_within = entrances_within.sort_values(['stop_name', 'entrance_name'])

    # Reorder
    entrances_within = entrances_within[['stop_name', 'entrance_name', 'entrance_id', 'lat','lon']]
    entrances_within.sort_values(['stop_name', 'entrance_name'], inplace=True)

    return pd.DataFrame(entrances_data)


def merge_and_clean_metro_entrances_and_stops(stops_lines, entrances_stops):
    """Merge and clean the metro data."""

    # Removed spaces at the begining and end of the column      
    stops_lines['stop_name'] = stops_lines['stop_name'].str.strip()
    entrances_stops['stop_name'] = entrances_stops['stop_name'].str.strip()
    entrances_stops['entrance_name'] = entrances_stops['entrance_name'].str.strip()

    # Merge entrances with members
    merged_df = pd.merge(
        entrances_stops,
        stops_lines,
        on="stop_name",
        how="outer"
    )

    # Update the 'line' column based on 'stop_name' when 'line' is NaN
    merged_df['line'] = merged_df.apply(
        lambda row: re.search(r'\((.*?)\)', row['stop_name']).group(1)
            if pd.isna(row['line']) and re.search(r'\((.*?)\)', row['stop_name'])
            else row['line'],
            axis=1
    )

    # Remove parentheses and their contents from 'stop_name'
    merged_df['stop_name'] = merged_df['stop_name'].str.replace(r'\s*\(.*?\)', '', regex=True)

    # Replace 'line' values "Metro" or "metro" with valid line from 'stop_name'
    merged_df['line'] = merged_df.apply(
        lambda row: row['stop_name'].split(" ")[-1] if row['line'] in ['Metro', 'metro'] else row['line'],
        axis=1
    )

    # Remove line information (e.g., "L2") from 'stop_name'
    merged_df['stop_name'] = merged_df['stop_name'].str.replace(r'\b(L\d+|[Mm]etro)\b', '', regex=True).str.strip()

    # Handle specific cases
    merged_df.loc[merged_df['stop_name'] == 'El Clot', 'stop_name'] = 'el Clot'
    merged_df.loc[merged_df['stop_name'] == 'El Carmel', 'stop_name'] = 'el Carmel'
    merged_df.loc[merged_df['stop_name'] == 'Glòries', 'line'] = 'L1'
    merged_df.loc[merged_df['stop_name'] == 'Barcelona - Plaça de Catalunya', 'line'] = 'L1'
    merged_df.loc[merged_df['stop_name'] == 'Gorg', 'line'] = 'L10 Nord'
    merged_df.loc[merged_df['line'].isin(['L9', 'L9S']), 'line'] = 'L9 Sud'
    
    # Drop duplicates
    merged_df = merged_df.drop_duplicates()

    # Fill NaN values in lat_y and lon_y (entrance) with values from lat_x and lon_x (stations)
    merged_df['lat_y'] = merged_df['lat_y'].fillna(merged_df['lat_x'])
    merged_df['lon_y'] = merged_df['lon_y'].fillna(merged_df['lon_x'])

    # Drop the lat_x and lon_x columns
    merged_df = merged_df.drop(columns=['lat_x', 'lon_x'])

    # Rename lat_y and lon_y to lat and lon
    merged_df = merged_df.rename(columns={'lat_y': 'lat', 'lon_y': 'lon'})

    # There are duplicates because there are two stop_areas for each station, 
    # one for each line direction.
    # Split the DataFrame into two parts: rows with NaN in 'entrance_id' and the rest
    nan_entrance_df = merged_df[merged_df['entrance_id'].isna()] # entrances and stops data
    non_nan_entrance_df = merged_df[~merged_df['entrance_id'].isna()] # lines and stops data

    # Drop duplicates only in the rows with NaN in 'entrance_id'
    nan_entrance_df = (
        nan_entrance_df
        .drop_duplicates(subset=['line', 'stop_name'], keep='first')
    )

    # Combine the two parts back together
    merged_df = pd.concat([nan_entrance_df, non_nan_entrance_df], ignore_index=True)

    # If there are rows with the same line and stop_name, and one has entrance_id and the other no,
    # drop the one that doesn't has it. 
    # Separate rows with and without valid 'entrance_id'
    with_entrance_id = merged_df[~merged_df['entrance_id'].isna()]
    without_entrance_id = merged_df[merged_df['entrance_id'].isna()]

    # Drop duplicates in rows with valid 'entrance_id'
    with_entrance_id = with_entrance_id.drop_duplicates(subset=['line', 'stop_name', 'entrance_name'], keep='first')

    # Handle rows without 'entrance_id' only if there are no valid rows for the same 'line' and 'stop_name'
    fallback_entrances = (
        without_entrance_id[~without_entrance_id[['line', 'stop_name']].apply(
            lambda x: (x['line'], x['stop_name']) in 
            set(zip(with_entrance_id['line'], with_entrance_id['stop_name'])),
            axis=1
        )]
    )

    # Combine both DataFrames back
    merged_df = pd.concat([with_entrance_id, fallback_entrances], ignore_index=True)


    merged_df = merged_df[
        ['stop_name', 'line', 'entrance_name','entrance_id', 'lat','lon']]
    
    # Select only the stations inside the city boundary
    geometry = [Point(xy) for xy in zip(merged_df['lon'], merged_df['lat'])]
    merged_df = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")
    boundary_gdf = dl.load_bcn_boundary()
    boundary_gdf = boundary_gdf.to_crs(4326)
    merged_df = merged_df[merged_df.within(boundary_gdf.geometry.iloc[0])].copy()
    merged_df.drop('geometry', axis=1, inplace=True)
    
    # Group rows with different lines but same stop_name and entrance 
    # Assign unique names to missing 'entrance_name'to avoid errors in the next step
    missing = merged_df['entrance_name'].isna()
    merged_df.loc[missing, 'entrance_name'] = merged_df.loc[missing, 'stop_name']

    grouped_df = (
        merged_df.groupby(['stop_name', 'entrance_name', 'lat', 'lon'], as_index=False)
        .agg({
            'stop_name': 'first',
            'line': lambda x: list(set(x)),  # Combine lists and ensure unique values
            'entrance_name': 'first',
            'entrance_id': 'first',
            'lat': 'first',
            'lon': 'first',
        })
    )
    return grouped_df


def main():
    # Ouput files
    bus_output_file = f"{PR_INPUT}/infrastructure_bus_stops_lines.csv"
    tram_output_file = f"{PR_INPUT}/infrastructure_tram_entrances_lines.csv"
    metro_output_file = f"{PR_INPUT}/infrastructure_metro_entrances_stops_lines.csv"

      #BUS
    if not os.path.exists(bus_output_file):
        bus_elements = fetch_overpass_data(QUERIES['bus'])
        bus_stops_lines = process_bus_stops(bus_elements)
        num_stops = len(bus_stops_lines)
        num_lines = len(bus_stops_lines['line'].unique())
        print(f"BUS -> {num_stops} stops and {num_lines} lines")
        bus_stops_lines.to_csv(bus_output_file, index=False)
    else:
        print(f"Bus dataset already exists in {bus_output_file}")
    
    # TRAM
    #if not os.path.exists(tram_output_file):
     #   tram_members_elements = fetch_overpass_data(QUERIES['tram'][0])
      #  tram_entrances_lines = process_metro_and_tram_stops_and_lines(tram_members_elements, 'tram')
       # num_entrances = len(tram_entrances_lines)
     #   num_stops = len(tram_entrances_lines['stop_name'].unique())
     #   lines = tram_entrances_lines['line'].unique()
     #   num_lines = len(lines)
     #   print(f"TRAM -> {num_entrances} entrances, {num_stops} stops, and {num_lines} lines ({lines})")
     #   tram_entrances_lines.to_csv(tram_output_file, index=False)
    #else:
    #    print(f"Tram dataset already exists in {tram_output_file}")

    # METRO
     #if not os.path.exists(metro_output_file):
      #   metro_elements1 = fetch_overpass_data(QUERIES['metro'][0])
      #   metro_elements2 = fetch_overpass_data(QUERIES['metro'][1])
      #   metro_stops_lines = process_metro_and_tram_stops_and_lines(metro_elements2, ['metro', 'subway', 'train', 'light_rail'])
      #   metro_entrances_stops = process_metro_entrances_and_stops(metro_elements1)

      #   metro_entrances_stops_lines = merge_and_clean_metro_entrances_and_stops(
       #      metro_stops_lines, metro_entrances_stops)

      #   num_entrances = len(metro_entrances_stops_lines)
       #  num_stops = len(metro_entrances_stops_lines['stop_name'].unique())
       #  lines = set(str(item) for sublist in metro_entrances_stops_lines['line'] for item in sublist)
       #  num_lines = len(lines)
       #  print(f"METRO -> {num_entrances} entrances, {num_stops} stops, and {num_lines} lines ({sorted(lines)})")
       #  metro_entrances_stops_lines.to_csv(metro_output_file, index=False)  


if __name__ == "__main__":
    main()
    


