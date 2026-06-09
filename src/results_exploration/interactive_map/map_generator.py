"""
Map generator module for creating the base interactive map.
"""

import os
import sys
from pathlib import Path
import branca.colormap as cm
import folium
import osmnx as ox
# Add project root to the path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import src.data_loader as dl

def generate_map(df_weighted_osm, city_name="Barcelona"):
    # Dictionary of city coordinates
    city_coords = {
        "Barcelona": [41.3851, 2.1734],
        "València": [39.4699, -0.3763],
        "Sevilla": [37.3891, -5.9845],
        "Zaragoza": [41.6488, -0.8891],
        "Palma": [39.5696, 2.6502],
        "Palmas de Gran Canaria": [28.1235, -15.4366],
        "Bilbao": [43.2630, -2.9350],
    }

    # Get city coordinates, default to Barcelona if city not found
    location = city_coords.get(city_name, city_coords["Barcelona"])

    # https://leaflet-extras.github.io/leaflet-providers/preview/
    tiles1 = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
    tiles2 = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png'
    attr1 = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>')
    attr2 = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')

    # Create a base map centered on the selected city
    m = folium.Map(location=location, zoom_start=12, control_scale=True)

    # Add different tile layers
    folium.TileLayer('CartoDB Positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer(tiles2, attr=attr2, name='CaroDB VoyagerNoLabels').add_to(m)
    folium.TileLayer('Cartodb dark_matter', name='CartoDB dark_matter').add_to(m)
    folium.TileLayer(tiles1, attr=attr1, name='CartoDB Light NoLabels').add_to(m)

    # Add city boundary from OSM data
    city_boundary = ox.geocode_to_gdf(f"{city_name}, Spain")
    folium.GeoJson(
        city_boundary.boundary, 
        name=f'{city_name} boundary', 
        opacity=0.75, 
        color='black', 
        weight=3
    ).add_to(m)

    # Create a continuous color map - blue shades, darker for higher values
    colormap = cm.LinearColormap(
        colors=['red', 'yellow', 'green'],
        vmin=df_weighted_osm['norm_score'].min(),
        vmax=df_weighted_osm['norm_score'].max()
    )
    colormap.add_to(m)

    return m, colormap