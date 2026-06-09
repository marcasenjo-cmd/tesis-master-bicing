#!/usr/bin/env python3
"""
Create four-panel comparison maps:
1. Administrative divisions (districts & neighborhoods)
2. Income by neighborhoods  
3. Infrastructure (bike lanes, stations) & topography
4. POI counts by neighborhood
"""

import os
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.ops import unary_union, split
from shapely.geometry import LineString
from matplotlib.path import Path as MplPath

# Add project root to path
project_root = Path().resolve()
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.assign_to_nodes.utils.load_input_datasets as lid

# Configuration
FIGURE_SIZE = (21, 7)
DPI = 300
OUTPUT_DIR = 'visualizations/administrative_maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_four_panel_comparison():
    """Create four-panel comparison map."""
    
    print("Loading data...")
    
    # Load administrative data
    districts = dl.load_bcn_districts()
    neighborhoods = dl.load_bcn_neighborhoods()
    city_boundary = dl.load_bcn_boundary(remove_top_left_part=False)
    
    # Load income data
    income_df = pd.read_csv(f'{RAW_DATA}/idescat/idescat_avg_income.csv')
    
    # Load infrastructure data
    bike_lanes = lid.load_bike_lanes(root='.')
    stations = load_bike_stations()
    altitude = load_altitude_data(city_boundary)
    
    # Load POI data
    pois_df = pd.read_csv(f'{PR_INPUT}/urbanism_pois.csv')
    
    # Convert to WGS84 for basemap
    districts_wgs84 = districts.to_crs(4326)
    neighborhoods_wgs84 = neighborhoods.to_crs(4326)
    city_boundary_wgs84 = city_boundary.to_crs(4326)
    bike_lanes_wgs84 = bike_lanes.to_crs(4326)
    stations_wgs84 = stations.to_crs(4326)
    altitude_wgs84 = altitude.to_crs(4326)
    
    # Verify CRS conversion was successful
    if districts_wgs84 is None or neighborhoods_wgs84 is None or city_boundary_wgs84 is None:
        raise ValueError("Failed to convert data to WGS84 CRS")
    
    # Create figure - now 2x2 layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Panel 1: Administrative divisions
    create_admin_panel(ax1, districts_wgs84, neighborhoods_wgs84, city_boundary_wgs84)
    
    # Panel 2: Income by neighborhoods
    create_income_panel(ax2, income_df, neighborhoods_wgs84, districts_wgs84, city_boundary_wgs84)
    
    # Panel 3: Infrastructure & topography
    create_infrastructure_panel(ax3, bike_lanes_wgs84, stations_wgs84, altitude_wgs84, 
                               districts_wgs84, city_boundary_wgs84)
    
    # Panel 4: POI counts by neighborhood
    create_poi_panel(ax4, pois_df, neighborhoods_wgs84, districts_wgs84, city_boundary_wgs84)
    
    # Add subplot labels (A, B, C, D)
    ax1.text(-0.05, 1.03, 'A', fontsize=12, fontweight='bold', transform=ax1.transAxes, ha='left', va='top')
    ax2.text(-0.05, 1.03, 'B', fontsize=12, fontweight='bold', transform=ax2.transAxes, ha='left', va='top')
    ax3.text(-0.05, 1.03, 'C', fontsize=12, fontweight='bold', transform=ax3.transAxes, ha='left', va='top')
    ax4.text(-0.05, 1.03, 'D', fontsize=12, fontweight='bold', transform=ax4.transAxes, ha='left', va='top')
    
    # Add common elements
    add_north_arrow(ax1)
    add_scalebar(ax1)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Save
    output_file = f'{OUTPUT_DIR}/barcelona_four_panel_comparison.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {output_file}")
    
    plt.show()
    return fig

def load_bike_stations():
    """Load and clean bike station data."""
    df = pd.read_csv('/home/data/raw/bcn/2025_01_Gener_BicingNou_INFORMACIO.csv')
    
    # Clean data
    df = df[['station_id', 'lat', 'lon']].dropna()
    df['station_id'] = df['station_id'].astype(int)
    df = df.groupby(['station_id', 'lat', 'lon']).size().reset_index(name='count')
    df = df.sort_values('count', ascending=False).drop_duplicates()
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
    return gdf.to_crs('EPSG:25831')

def load_altitude_data(city_boundary):
    """Load and clean altitude data."""
    df = pd.read_csv(f'{PR_INPUT}/topology_altitude_slope_50.csv')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=4326)
    gdf = gdf.to_crs(25831).clip(city_boundary.geometry.total_bounds)
    gdf = gdf.dropna(subset=['altitude'])
    gdf.loc[gdf['altitude'] < 0, 'altitude'] = 0
    return gdf

def create_admin_panel(ax, districts, neighborhoods, city_boundary):
    """Create administrative divisions panel."""
    print("Creating administrative panel...")
    
    # Plot districts with colors
    colors = sns.color_palette("Set3", len(districts))
    for i, (idx, district) in enumerate(districts.iterrows()):
        districts[districts.index == idx].plot(ax=ax, color=colors[i], alpha=0.6, 
                                             edgecolor='black', linewidth=2)
    
    # Add boundaries and labels
    neighborhoods.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.8)
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=3)
    
    # Add district labels
    for idx, district in districts.iterrows():
        centroid = district.geometry.centroid
        name = 'l\'Eixample' if district['district'] == 'Eixample' else district['district']
        ax.text(centroid.x, centroid.y, name, fontsize=7, fontweight='semibold', 
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', alpha=0.75))
    
    add_basemap(ax)
    ax.set_axis_off()

def create_income_panel(ax, income_df, neighborhoods, districts, city_boundary):
    """Create income by neighborhoods panel."""
    print("Creating income panel...")
    
    # Process income data
    income_2022 = income_df[
        (income_df['año'] == 2022) & 
        (income_df['municipio'] == 'Barcelona') &
        (income_df['barrio'] != 'total')
    ].copy()
    income_2022['income_2022'] = pd.to_numeric(income_2022['valor'], errors='coerce')
    income_2022 = income_2022.dropna(subset=['income_2022'])
    
    # Add income to neighborhoods
    neighborhoods_with_income = neighborhoods.copy()
    income_mapping = dict(zip(income_2022['barrio'], income_2022['income_2022']))
    neighborhoods_with_income['income_2022'] = neighborhoods_with_income['neighborhood'].map(income_mapping)
    
    # Plot with colorbar
    income_values = neighborhoods_with_income['income_2022'].dropna()
    if len(income_values) > 0:
        vmin, vmax = income_values.min(), income_values.max()
        neighborhoods_with_income.plot(ax=ax, column='income_2022', cmap='plasma', alpha=0.8,
                                     edgecolor='white', linewidth=0.1, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cax = inset_axes(ax, width="22%", height="3%", loc='lower left',
                        bbox_to_anchor=(0.72, 0.07, 1, 0.7), bbox_transform=ax.transAxes)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Avg. net income/capita (€)', fontsize=7)
        cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        cbar.ax.tick_params(labelsize=6)
    
    # Add boundaries
    districts.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.7)
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=3)
    
    add_basemap(ax)
    ax.set_axis_off()

def create_infrastructure_panel(ax, bike_lanes, stations, altitude, districts, city_boundary):
    """Create infrastructure and topography panel."""
    print("Creating infrastructure panel...")
    
    # Plot layers
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=3)
    districts.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    # Plot topography
    plot_altitude_contours(ax, altitude, city_boundary)
    
    # Plot infrastructure
    bike_lanes.plot(ax=ax, color='tab:blue', linewidth=0.6, alpha=1)
    stations.plot(ax=ax, color='red', markersize=8, marker='o', 
                  edgecolor='white', linewidth=0.5, alpha=1)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='tab:blue', linewidth=1.5, label='Bike lanes'),
        Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=6, 
                label='Bike stations', markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0)
    
    add_basemap(ax)
    ax.set_axis_off()

def create_poi_panel(ax, pois_df, neighborhoods, districts, city_boundary):
    """Create POI counts by neighborhood panel."""
    print("Creating POI panel...")
    
    # Convert POI data to GeoDataFrame
    # Extract coordinates from POINT geometry strings
    pois_df['x'] = pois_df['geometry'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')[0].astype(float)
    pois_df['y'] = pois_df['geometry'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')[1].astype(float)
    
    pois_gdf = gpd.GeoDataFrame(
        pois_df, 
        geometry=gpd.points_from_xy(pois_df['x'], pois_df['y']),
        crs='EPSG:25831'
    )
    
    # Convert to WGS84 for consistency
    pois_wgs84 = pois_gdf.to_crs(4326)
    
    # Count non-green POIs per neighborhood
    poi_counts = []
    for idx, neighborhood in neighborhoods.iterrows():
        # Find POIs within this neighborhood
        if neighborhood.geometry is not None:
            poi_in_neighborhood = pois_wgs84[pois_wgs84.geometry.within(neighborhood.geometry)]
            
            # Count non-green POIs (exclude green=1)
            # non_green_pois = poi_in_neighborhood[poi_in_neighborhood['green'] == 0]
            total_pois = len(poi_in_neighborhood)
        else:
            total_pois = 0
        
        poi_counts.append({
            'neighborhood': neighborhood['neighborhood'],
            'poi_count': total_pois
        })
    
    # Create DataFrame with counts
    poi_counts_df = pd.DataFrame(poi_counts)
    
    # Merge with neighborhoods for plotting
    neighborhoods_with_pois = neighborhoods.copy()
    neighborhoods_with_pois = neighborhoods_with_pois.merge(poi_counts_df, on='neighborhood', how='left')
    neighborhoods_with_pois['poi_count'] = neighborhoods_with_pois['poi_count'].fillna(0)
    
    # Plot with colorbar
    poi_values = neighborhoods_with_pois['poi_count'].dropna()
    if len(poi_values) > 0:
        vmin, vmax = poi_values.min(), poi_values.max()
        neighborhoods_with_pois.plot(ax=ax, column='poi_count', cmap='viridis', alpha=0.8,
                                   edgecolor='white', linewidth=0.1, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cax = inset_axes(ax, width="22%", height="3%", loc='lower left',
                        bbox_to_anchor=(0.72, 0.07, 1, 0.7), bbox_transform=ax.transAxes)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('POI count', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    
    # Add boundaries
    districts.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.7)
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=3)
    
    add_basemap(ax)
    ax.set_axis_off()

def plot_altitude_contours(ax, altitude_df, city_boundary, granularity=25):
    """Plot altitude contours."""
    lons = altitude_df['geometry'].x
    lats = altitude_df['geometry'].y
    alts = altitude_df['altitude'].values
    
    min_alt, max_alt = np.min(alts), np.max(alts)
    levels = np.arange(np.floor(min_alt), np.ceil(max_alt), granularity)
    
    if len(levels) > 1:
        lines = ax.tricontour(lons, lats, alts, levels=levels, cmap='terrain',
                             linewidths=0.5, linestyles='dashed')
        
        # Add labels for major elevations
        major_levels = np.arange(np.floor(min_alt), np.ceil(max_alt), 100)
        major_levels = major_levels[major_levels <= max_alt]
        if len(major_levels) > 0:
            ax.clabel(lines, levels=major_levels, inline=True, fontsize=7, 
                     fmt='%d m', colors='gray')
        
        # Clip contours to city boundary
        clip_contours_to_boundary(lines, city_boundary)

def clip_contours_to_boundary(lines, city_boundary):
    """Clip contour lines to city boundary."""
    city_poly = unary_union(city_boundary.geometry)
    city_border = city_poly.boundary
    
    try:
        collections = lines.collections
    except AttributeError:
        collections = [lines]
        
    for coll in collections:
        new_paths = []
        for path in coll.get_paths():
            verts = path.vertices
            codes = path.codes
            split_idxs = np.where(codes == MplPath.MOVETO)[0].tolist() + [len(verts)]
            
            for i in range(len(split_idxs)-1):
                start, end = split_idxs[i], split_idxs[i+1]
                segment_verts = verts[start:end]
                if len(segment_verts) < 2:
                    continue
                
                seg_line = LineString(segment_verts)
                clipped = split(seg_line, city_border)
                
                for piece in clipped.geoms:
                    if piece.is_empty:
                        continue
                    mid = piece.interpolate(0.5, normalized=True)
                    if not city_poly.contains(mid):
                        continue
                    
                    pc = np.array(piece.coords)
                    codes_new = [MplPath.MOVETO] + [MplPath.LINETO]*(len(pc)-1)
                    new_paths.append(MplPath(pc, codes_new))
        
        coll.set_paths(new_paths)

def add_basemap(ax, zoom=12):
    """Add basemap to axis."""
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, 
                   crs='EPSG:4326', zoom=zoom, attribution=False)

def add_north_arrow(ax, size=0.04, location=(0.925, 0.135)):
    """Add north arrow."""
    ax.annotate('', xy=(location[0], location[1]), xytext=(location[0], location[1] - size),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='k', edgecolor='k', width=1, headwidth=8, headlength=12),
                ha='center', va='center')
    ax.text(location[0], location[1] - size - 0.015, 'N', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, fontweight='semibold')

def add_scalebar(ax, length=2000, location=(0.875, 0.05)):
    """Add scale bar."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    length_deg = length / 111000
    x_start = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_start = ylim[0] + location[1] * (ylim[1] - ylim[0])
    x_end = x_start + length_deg
    
    ax.plot([x_start, x_end], [y_start, y_start], color='k', linewidth=4, zorder=100)
    ax.text((x_start + x_end) / 2, ylim[0] + 0.013 * (ylim[1] - ylim[0]), 
            f'{int(length/1000)} km', ha='center', va='bottom', fontsize=9, 
            fontweight='semibold', zorder=101)

if __name__ == "__main__":
    print("Creating Barcelona four-panel comparison maps...")
    create_four_panel_comparison()
    print("Done!")
