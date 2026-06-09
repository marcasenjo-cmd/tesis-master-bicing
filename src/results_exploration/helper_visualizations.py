"""
This module provides utility functions for creating various types of visualizations
for bike-sharing system (BSS) optimization results. It includes functions for:

- Grid-based spatial analysis and point counting
- Node plotting on maps with customizable styling
- Radar charts for multi-dimensional comparisons

Author: Jordi Grau Escolano
"""

import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable  # type: ignore
import contextily as ctx  # type: ignore

# Radar chart
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as Path_
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Add project root to path
project_root = Path().resolve().parents[0]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl
import src.optimization.experiments.helper_experiment as he
import src.optimization.helper_optimization as ho



################################################################################
# Helper functions
################################################################################

def create_grid_cells_and_count_points(bcn_boundary, df_points, grid_size=500):
    """
    This function creates a regular grid over the study area and counts how many
    points (e.g., BSS stations) fall within each grid cell. Useful for spatial
    distribution analysis and density mapping.

    Args:
        bcn_boundary (gpd.GeoDataFrame): The boundary of the study area (e.g., Barcelona city boundary)
        df_points (gpd.GeoDataFrame): The points to count (e.g., station locations)
        grid_size (int, optional): The size of the grid cells in meters. Defaults to 500.
        
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the grid cells and the number 
            of points in each cell. Each row has:
            - geometry: Polygon representing the grid cell
            - n_points: Count of points within that cell
            - CRS: Same as input boundary (EPSG:25831)
    
    Example:
        >>> bcn_boundary = dl.load_bcn_boundary()
        >>> stations = gpd.read_file('stations.geojson')
        >>> grid_cells = create_grid_cells_and_count_points(bcn_boundary, stations, grid_size=1000)
        >>> print(f"Created {len(grid_cells)} grid cells")
        >>> print(f"Max stations per cell: {grid_cells['n_points'].max()}")
    """
    
    # Create grid cells
    xmin, ymin, xmax, ymax = bcn_boundary.total_bounds
    xmin, ymin, xmax, ymax = xmin - 1000, ymin - 1000, xmax + 1000, ymax + 1000
    x_coords = np.arange(xmin, xmax, grid_size)
    y_coords = np.arange(ymin, ymax, grid_size)
    polygons = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            polygons.append(Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size)
            ]))
    grid_cells = gpd.GeoDataFrame(geometry=polygons, crs=bcn_boundary.crs)
    
    # Only keep cells that intersect with bcn_boundary
    grid_cells = grid_cells[grid_cells.intersects(bcn_boundary.unary_union)]

    # Count the number of points in each grid cell
    grid_cells = gpd.GeoDataFrame(
        grid_cells.sjoin(df_points).groupby('geometry').size().reset_index(name='n_points'), 
        crs=25831)
    return grid_cells



################################################################################
# Plotting functions
################################################################################
"""
All the plotting functions below are used to plot on a map with the city boundary and
the basemap of OpenStreetMap.
"""

def plot_nodes_on_map(
        df_w, column, ax, city_boundary, node_size=0.5, ctx_background=True, show_colorbar=True, 
        show_norm_raw_vals=False, attribution=False, attribution_size=8):
    """
    Plot all nodes of df_weighted on a map with customizable styling.

    This function creates a scatter plot of nodes (e.g., potential BSS station locations)
    colored according to a specified column (e.g., normalized scores). It includes
    city boundary overlay, OpenStreetMap basemap, and optional colorbar.

    Args:
        df_w (gpd.GeoDataFrame): GeoDataFrame containing the weighted nodes with geometry
        column (str): The column name to use for coloring the nodes (should contain values 0-1)
        ax (matplotlib.axes.Axes): The matplotlib axes object to plot on
        city_boundary (shapely.geometry.MultiPolygon): The city boundary to overlay
        node_size (float, optional): Size of the node markers. Defaults to 0.5.
        ctx_background (bool, optional): Whether to add OpenStreetMap basemap. Defaults to True.
        show_colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
        show_norm_raw_vals (bool, optional): Whether to show original min/max values in colorbar. Defaults to False.
        attribution (bool, optional): Whether to show map attribution. Defaults to False.
        attribution_size (int, optional): Font size for attribution text. Defaults to 8.
    
    Returns:
        None: The plot is rendered on the provided axes

    
    Notes:
        - Uses a custom colormap that starts with gray for zero values
        - Automatically normalizes values to 0-1 range for consistent coloring
        - Adds CartoDB Positron basemap for clean, professional appearance
        - City boundary is plotted as a black outline with 70% transparency
    """
    if city_boundary is not None:
        city_boundary.boundary.plot(ax=ax, color='black', alpha=0.7)

    # Create a custom colormap with gray for zero values
    viridis_r = plt.get_cmap('viridis_r')
    colors = [(0.7, 0.7, 0.7, 1.0)]  # Start with gray
    colors.extend(viridis_r(np.linspace(0, 1, 255)))  # Add viridis_r colors
    custom_cmap = mpl.colors.ListedColormap(colors)
    
    # Define normalization
    norm = Normalize(vmin=0, vmax=1)

    # Plot stations
    scatter = df_w.plot(
                column=column,
                cmap=custom_cmap,
                ax=ax,
                markersize=node_size,
                legend=False,
                vmin=0, vmax=1)

    # Add colorbar
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        if show_norm_raw_vals:
            original_min, original_max = df_w[column].min(), df_w[column].max()
            cbar.ax.set_ylabel(f'Score (min={original_min:.2f}, max={original_max:.2f})', fontsize=8)

    # Add basemap
    if ctx_background:
        if attribution:
            ctx.add_basemap(
                ax, source=ctx.providers.CartoDB.Positron, crs=df_w.crs.to_string(), zoom=12,
                attribution_size=attribution_size)
        else:
            ctx.add_basemap(
                ax, source=ctx.providers.CartoDB.Positron, crs=df_w.crs.to_string(), 
                zoom=12, attribution=False)

    ax.set_axis_off()


def plot_grid_cells_on_map(bcn_boundary, df_solution, grid_size, ax, show_colorbar=True, vmin=None, vmax=None):
    """
    Plot grid cells showing station density on a map.

    This function creates a choropleth map where each grid cell is colored according
    to the number of stations it contains. Useful for analyzing spatial distribution
    patterns of BSS stations.

    Args:
        bcn_boundary (gpd.GeoDataFrame): The city boundary for overlay
        df_solution (gpd.GeoDataFrame): GeoDataFrame containing station locations
        grid_size (int): Size of grid cells in meters
        ax (matplotlib.axes.Axes): The matplotlib axes object to plot on
        show_colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
        vmin (int, optional): Minimum value for colorbar scale. If None, uses data minimum.
        vmax (int, optional): Maximum value for colorbar scale. If None, uses data maximum.
    
    Returns:
        gpd.GeoDataFrame: The grid cells GeoDataFrame with station counts
    """
    # Count the number of stations in each grid cell
    grid_cells = create_grid_cells_and_count_points(bcn_boundary, df_solution, grid_size)

    # Plot the grid cells
    bcn_boundary.boundary.plot(ax=ax, color='black', alpha=0.7)
    
    # Get bounds - use provided vmin/vmax if available, otherwise use data range
    if vmin is not None and vmax is not None:
        # Use provided bounds for consistent scale across plots
        min_val, max_val = vmin, vmax
        # Create evenly spaced bounds
        n_classes = min(10, max_val - min_val + 1)  # Limit to 10 classes max
        bounds = np.linspace(min_val, max_val, n_classes + 1)
    else:
        # Use data-specific bounds (original behavior)
        unique_values = sorted(grid_cells['n_points'].unique())
        n_classes = len(unique_values)
        bounds = np.linspace(min(unique_values), max(unique_values), n_classes + 1)
        min_val, max_val = min(unique_values), max(unique_values)
    
    # Create custom colormap
    cmap = plt.get_cmap('viridis_r')  # base colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, n_classes)
    
    # Define the normalization
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot grid cells with station counts
    plot = grid_cells.plot(
        column='n_points',
        cmap=cmap,
        norm=norm,
        ax=ax,
        alpha=0.75,
        legend=False,
        linewidth=0.5
    )

    # Add colorbar
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        
        # Create colorbar with integer ticks
        cb = mpl.colorbar.ColorbarBase(
            cax, 
            cmap=cmap,
            norm=norm,
            spacing='proportional',
            ticks=bounds,
            boundaries=bounds,
            format='%1i'
        )
        
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Number of stations', fontsize=8)

    # Add basemap
    ctx.add_basemap(
        ax, source=ctx.providers.CartoDB.Positron, crs=df_solution.crs.to_string(), zoom=12)

    ax.set_axis_off()
    return grid_cells


################################################################################
# Radar chart
################################################################################

def radar_factory(num_vars, frame='circle', font_size=8):
    """
    This function creates a custom matplotlib projection for radar charts.
    It supports both circular and polygonal frame styles and provides
    a clean, professional appearance suitable for publication.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart (number of axes)
    frame : {'circle', 'polygon'}, optional
        Shape of frame surrounding the axes. Defaults to 'circle'.
    font_size : int, optional
        Font size for axis labels and grid. Defaults to 8.
    
    Returns
    -------
   
    Notes:
        - Creates a custom 'radar' projection that can be used with subplot_kw
        - Supports both circular and polygonal frame styles
        - Automatically sets theta zero location to North (top of chart)
        - Includes custom radial grid with 4 levels (0, 0.33, 0.67, 1.0)
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path_(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            # Set custom radial grid
            self.set_rgrids([0, 0.3333, 0.6666, 1], labels=[], angle=0, fontsize=font_size)

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def fill_between(self, *args, **kwargs):
            """Override fill_between to handle closed polygons"""
            return super().fill_between(*args, **kwargs)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=font_size)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                    radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                            spine_type='circle',
                            path=Path_.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                  + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_radar_chart(mean_values, std_values, labels, weights=None, ax=None, 
                     title=None, frame='polygon', fill_alpha=0.25, color='tab:blue',
                     font_size=8, string_replace=None):
    """
    Create a radar chart with mean values and standard deviation bands.

    This function creates a comprehensive radar chart showing mean values as solid lines
    and standard deviation as shaded areas. It's particularly useful for comparing
    multiple scenarios or showing uncertainty in optimization results.

    Parameters
    ----------
    mean_values : pandas.Series or array-like
        Mean values for each variable (should be 0-1 normalized)
    std_values : pandas.Series or array-like
        Standard deviation values for each variable
    labels : list
        Labels for each variable (will be displayed around the chart)
    weights : dict, optional
        Dictionary mapping variable names to their weights (0-1). If provided,
        vertical lines will be drawn at each variable up to their weight value.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes with 'radar' projection.
    title : str, optional
        Title for the chart
    frame : {'circle', 'polygon'}, optional
        Shape of frame surrounding axes. Defaults to 'polygon'.
    fill_alpha : float, optional
        Alpha value for the standard deviation fill area. Defaults to 0.25.
    color : str, optional
        Color for the mean line and fill area. Defaults to 'tab:blue'.
    font_size : int, optional
        Font size for labels and text. Defaults to 8.
    string_replace : dict, optional
        Dictionary for replacing strings in labels for display purposes.
        Example: {'string_to_replace': 'replacement_string'}
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the radar chart

    Notes:
        - Variables are automatically sorted alphabetically for consistent positioning
        - Weight indicators show as vertical black lines with caps
        - Chart is oriented clockwise starting from North (top)
        - Mean values are shown as solid lines, std as semi-transparent fill
    """
    # Sort labels and reorder data accordingly
    if string_replace is not None:
        labels_str = [string_replace.get(label, label) for label in labels]
        sorted_indices = np.argsort(labels_str)
    else:
        sorted_indices = np.argsort(labels)
    
    labels = [labels[i] for i in sorted_indices]
    mean_values = mean_values.iloc[sorted_indices]
    std_values = std_values.iloc[sorted_indices]

    num_vars = len(labels)
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    # Make arrays circular by appending first value
    theta_plot = np.append(theta, theta[0])
    mean_values_plot = np.append(mean_values, mean_values.iloc[0])
    std_values_plot = np.append(std_values, std_values.iloc[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Set the direction to clockwise and start at top
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')

    # Plot mean values with a strong line
    ax.plot(theta_plot, mean_values_plot, '-', linewidth=1, label='Mean', color=color)
    
    # Calculate upper and lower bounds for std area
    upper_bound = np.minimum(mean_values_plot + std_values_plot, 1.0)  # Cap at 1.0
    lower_bound = np.maximum(mean_values_plot - std_values_plot, 0.0)  # Floor at 0.0
    
    # Plot standard deviation as shaded area
    ax.fill_between(theta_plot, lower_bound, upper_bound, alpha=fill_alpha, label='±1 std', color=color)
    
    # Add vertical lines at each variable up to their scenario weight
    if weights is not None:
        # Get original indices before sorting
        if string_replace is not None:
            labels_str = [string_replace.get(label, label) for label in labels]
            original_indices = np.argsort(np.argsort(labels_str))
        else:
            original_indices = np.argsort(np.argsort(labels))
            
        for i, angle in enumerate(theta):
            # Get the original variable name before sorting
            var_name = labels[original_indices[i]]
            # Only draw line if variable exists in scenario weights
            if var_name in weights:
                weight = weights[var_name]
                # Draw vertical line up to weight value
                ax.plot([angle, angle], [0, weight], color='black', linewidth=1, linestyle='-')
                # Draw cap at weight value
                ax.plot([angle-0.07, angle+0.07], [weight, weight], color='black', linewidth=1, linestyle='-')
    
    # Set the labels
    ax.set_xticks(theta)
    ax.set_xticklabels(labels_str, fontsize=font_size)
    ax.tick_params(pad=5)  # Reduce padding between labels and plot
    
    if title:
        ax.set_title(title)
    
    # Add legend
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    return ax