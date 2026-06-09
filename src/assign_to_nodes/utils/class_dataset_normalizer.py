"""
This module provides data normalization capabilities for preparing datasets
for the later BSS optimization. It supports multiple normalization methods and
automatically generates visualizations to compare original and normalized distributions.

Author: Jordi Grau Escolano
"""

import sys
from pathlib import Path
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import geopandas as gpd  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from scipy import stats  # type:ignore

# Define project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import *
import src.data_loader as dl

bcn_bounday = dl.load_bcn_boundary()

class DatasetNormalizer:
    """   
    This class provides data normalization capabilities for preparing datasets
    for the later BSS optimization. It supports multiple normalization methods and
    automatically generates visualizations to compare original and normalized distributions.
    
    Normalization Methods:
    - **minmax**: Scales data to range [0, 1] using min/max values
    - **log**: Natural logarithm transformation for skewed distributions
    - **boxcox**: Box-Cox transformation for non-normal distributions
    - **robust**: Median and IQR-based normalization
    - **zscore**: Standard score normalization with scaling to [0, 1]
    - **0-max**: Scales from 0 to maximum value
    - **inverted_***: Inverted versions (1 - normalized_value) for all methods
    
    Visualization Features:
    - **Distribution Comparison**: Side-by-side plots of original vs normalized data
    - **Multi-plot Groups**: Organized visualization by variable categories
    - **Correlation Matrices**: Before and after normalization correlation analysis
    - **Statistical Summary**: Descriptive statistics for each transformation
    """
    
    def __init__(self, df: pd.DataFrame, normalized_df_filepath: str):
        # Store both the original DataFrame and a copy for normalization.
        self.original_df = df.copy()
        self.normalized_df = df.copy()
        self.normalized_df_filepath = normalized_df_filepath


    def normalize_series(self, column: str, series: pd.Series, method: str) -> pd.Series:
        """
        Normalize a pandas Series using the specified method.
        
        Parameters:
            column (str): Name of the column being normalized
            series (pd.Series): Data to normalize
            method (str): Normalization method to use
                - 'minmax': Scales data to range [0, 1]
                - '0-max': Scales data to range [0, 1] using 0 as minimum
                - 'robust': Normalizes using median and IQR, clipped to [0, 1]
                - 'log': Log transformation followed by scaling to [0, 1]
                - 'boxcox': Box-Cox transformation followed by scaling to [0, 1]
                
                Any method can be prefixed with 'inverted_' to get 1 - normalized_value.
                For example:
                - 'inverted_minmax': 1 - minmax scaling
                - 'inverted_robust': 1 - robust scaling
                etc.
        
        Returns:
            pd.Series: Normalized data in range [0, 1]
        
        Raises:
            ValueError: If method is not supported or if data cannot be normalized
        """
        method = method.lower()
        is_inverted = method.startswith('inverted_')
        base_method = method.replace('inverted_', '') if is_inverted else method

        # Get the normalized series using the base method
        if base_method in ['minmax', '0-max']:
            max_val = series.max()
            min_val = series.min()                
            if base_method == '0-max': 
                min_val = 0
            if max_val == min_val:
                return pd.Series(np.zeros(len(series)), index=series.index)
            normalized = (series - min_val) / (max_val - min_val)
            
        elif base_method == 'zscore':
            # Calculate mean and standard deviation
            mean = series.mean()
            std = series.std()
            
            # Handle case where standard deviation is zero
            if std == 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            
            # Apply z-score normalization
            z_scores = (series - mean) / std
            
            # Scale to [0, 1] range using min-max scaling on z-scores
            min_z, max_z = z_scores.min(), z_scores.max()
            
            if min_z == max_z:
                return pd.Series(np.zeros(len(series)), index=series.index)
            
            # Scale z-scores to [0, 1]
            normalized = (z_scores - min_z) / (max_z - min_z)
    
        elif base_method == 'robust':
            median = series.median()
            q75, q25 = series.quantile(0.75), series.quantile(0.25)
            iqr = q75 - q25
            if iqr == 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            normalized = (series - median) / iqr
            # Clip to 0-1 range
            normalized = normalized.clip(0, 1)
        
        elif base_method == 'log':
            # log1p = log(1+x) -> handles zeros without needing an offset
            log_values = np.log1p(series)
            
            # Scale the log-transformed values to the range [0, 1]
            min_val, max_val = log_values.min(), log_values.max()
            if min_val == max_val:
                return pd.Series(np.zeros(len(series)), index=series.index)
            
            normalized = (log_values - min_val) / (max_val - min_val)

        elif base_method == 'boxcox':
            # Avoid errors when the series contains zeros
            series = series.replace(0, 1e-6)
            # Apply Box-Cox transformation
            transformed = stats.boxcox(series)[0]  # Get only transformed values
            # Min and max of the transformed values
            min_val, max_val = transformed.min(), transformed.max()
            if min_val == max_val:
                return pd.Series(np.zeros(len(series)), index=series.index)
            
            # Min-max scale the transformed values to [0, 1]
            normalized = (transformed - min_val) / (max_val - min_val)
                
        else:
            raise ValueError(f"Normalization method '{method}' is not supported.")

        # Return inverted if requested
        return 1 - normalized if is_inverted else normalized


    def plot_variable_group(self, group_name, columns, methods_dict, ncols=3):
        """
        Create a grid visualization for a group of related variables.
        Each column contains a KDE plot with twin axes (top) and two maps showing 
        original and normalized data (middle and bottom).
        All plots are arranged in a single row.
        
        Args:
            group_name (str or int): Name or identifier for this variable group
            columns (list): List of column names to include in this group
            methods_dict (dict): Dictionary mapping column names to their normalization methods
            ncols (int): Number of columns in the grid (default: 3)
        """
        # Filter columns to only include those that exist in methods_dict
        valid_columns = [col for col in columns if col in methods_dict]
        
        if not valid_columns:
            print(f"Warning: No valid columns found in group {group_name}")
            return
        
        # Create a filtered methods dictionary for just these columns
        group_methods = {col: methods_dict[col] for col in valid_columns}
        
        # Calculate number of columns needed
        n_variables = len(valid_columns)
        ncols = n_variables
        
        # Create figure with appropriate size
        fig_width = 6 * ncols
        fig_height = 15  # Increased height for three rows
        fig = plt.figure(figsize=(fig_width, fig_height))
        markersize = 1
        kde_bw_method = 0.005
        
        # Check that the geometry column exists
        if 'geometry' not in self.normalized_df.columns:
            raise ValueError("The DataFrame does not have a 'geometry' column for mapping.")
        
        # Convert normalized DataFrame to GeoDataFrame
        gdf_normalized = gpd.GeoDataFrame(self.normalized_df)
        
        # Convert original DataFrame to GeoDataFrame
        gdf_original = gpd.GeoDataFrame(self.original_df)
        
        # Create a GridSpec layout with 3 rows of different heights
        # KDEs will be 30%, original maps will be 35%, normalized maps will be 35%
        gs = fig.add_gridspec(3, ncols, height_ratios=[0.12, 0.44, 0.44])
        
        # Plot each variable
        for i, col in enumerate(valid_columns):
            if i >= ncols:
                break  # Don't exceed the number of columns
            
            method = group_methods[col]
            
            # Get data
            orig_values = self.original_df[col].dropna()
            norm_values = self.normalized_df[col].dropna()
            
            # Create KDE subplot (top row)
            ax_kde = fig.add_subplot(gs[0, i])
            
            # Create twin axes for both x and y
            ax_kde2 = ax_kde.twinx().twiny()
            
            # Plot original KDE on original axes with less smoothing
            orig_values.plot.kde(ax=ax_kde, color='blue', linewidth=2, 
                            label=f'Original', bw_method=kde_bw_method)
            # orig_values.hist(ax=ax_kde, bins=100,color='blue', alpha=0.2)
            
            # Add outlier indicators for original data
            q1_orig = orig_values.quantile(0.25)
            q3_orig = orig_values.quantile(0.75)
            iqr_orig = q3_orig - q1_orig
            lower_bound_orig = q1_orig - 1.5 * iqr_orig
            upper_bound_orig = q3_orig + 1.5 * iqr_orig
            
            # Plot vertical lines for original outlier bounds
            ax_kde.axvline(x=lower_bound_orig, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax_kde.axvline(x=upper_bound_orig, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set x-axis limits for original data
            ax_kde.set_xlim(orig_values.min(), orig_values.max())
            
            # Plot normalized KDE on twin axes
            norm_values.plot.kde(ax=ax_kde2, color='green', linewidth=2, 
                            label=f'Normalized', bw_method=kde_bw_method)
            
            # Add outlier indicators for normalized data
            q1_norm = norm_values.quantile(0.25)
            q3_norm = norm_values.quantile(0.75)
            iqr_norm = q3_norm - q1_norm
            lower_bound_norm = q1_norm - 1.5 * iqr_norm
            upper_bound_norm = q3_norm + 1.5 * iqr_norm
            
            # Plot vertical lines for normalized outlier bounds
            ax_kde2.axvline(x=lower_bound_norm, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax_kde2.axvline(x=upper_bound_norm, color='green', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set x-axis limits for normalized data
            ax_kde2.set_xlim(0, 1)
            
            # Configure axes labels
            ax_kde.set_xlabel('Value (Original)', color='blue')
            ax_kde.set_ylabel('Density (Original)', color='blue')
            ax_kde2.set_xlabel(f'Value ({method})', color='green')
            ax_kde2.yaxis.set_label_position('right')  # Position the y-axis label on the right
            ax_kde2.set_ylabel(f'Density ({method})', color='green')
            
            # Set colors for axis labels and ticks
            ax_kde.tick_params(axis='x', labelcolor='blue')
            ax_kde.tick_params(axis='y', labelcolor='blue')
            ax_kde2.tick_params(axis='x', labelcolor='green')
            ax_kde2.tick_params(axis='y', labelcolor='green')
            ax_kde2.yaxis.set_ticks_position('right')  # Position the y-axis label on the right
            
            ax_kde.set_title(f'{col.upper()}', fontsize=12)
            
            # Add legends for both axes
            lines1, labels1 = ax_kde.get_legend_handles_labels()
            lines2, labels2 = ax_kde2.get_legend_handles_labels()
            ax_kde.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
            # Add statistics text box
            stats_text = (
                f"Orig: μ={orig_values.mean():.2f}, σ={orig_values.std():.2f}\n"
                f"Norm: μ={norm_values.mean():.2f}, σ={norm_values.std():.2f}"
            )
            ax_kde.text(0.02, 0.98, stats_text, transform=ax_kde.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)
            
            # Create original map subplot (middle row)
            ax_map_orig = fig.add_subplot(gs[1, i])
            
            # Plot original map
            bcn_bounday.boundary.plot(ax=ax_map_orig, color='black', linewidth=1)
            
            # Create mask for zero values
            zero_mask = gdf_original[col] == 0
            
            # Plot non-zero values
            gdf_original[~zero_mask].plot(
                column=col, ax=ax_map_orig,
                legend=True, cmap='viridis_r',
                markersize=markersize, alpha=0.8
            )
            
            # Plot zero values in gray
            if zero_mask.sum() > 0:
                gdf_original[zero_mask].plot(
                    ax=ax_map_orig, color='gray',
                    markersize=markersize, alpha=0.8
                )
            
            ax_map_orig.set_title("Original", fontsize=10)
            ax_map_orig.axis('off')
            
            # Add north arrow and scale bar only to the first map of the second row
            if i == 0:  # First column only
                # North arrow
                fontsize = 13
                ax_map_orig.annotate(
                    '', 
                    xy=(0.12, 0.295),  # Arrow head position
                    xytext=(0.12, 0.232),  # Arrow stick base position - increased distance
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=10, headlength=15),
                    ha='center', va='center'
                )
                ax_map_orig.text(0.12, 0.21, 'N', transform=ax_map_orig.transAxes, 
                        ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')
                
                # Scale bar
                xlim = ax_map_orig.get_xlim()
                ylim = ax_map_orig.get_ylim()
                
                # Calculate scale bar position
                x_center = xlim[0] + 0.12 * (xlim[1] - xlim[0])  # Center point
                x_start = x_center - 1000  # 1 km to left of center
                x_end = x_center + 1000    # 1 km to right of center
                y_scale = ylim[0] + 0.18 * (ylim[1] - ylim[0])  # Below arrow
                
                # Draw the scale bar
                ax_map_orig.plot([x_start, x_end], [y_scale, y_scale], color='black', linewidth=3, solid_capstyle='butt')
                ax_map_orig.text(x_center, y_scale - 0.01 * (ylim[1] - ylim[0]), '2 km',
                        ha='center', va='top', fontsize=fontsize, color='black', fontweight='bold')
            
            # Create normalized map subplot (bottom row)
            ax_map_norm = fig.add_subplot(gs[2, i])
            
            # Plot normalized map
            bcn_bounday.boundary.plot(ax=ax_map_norm, color='black', linewidth=1)
            
            # Create mask for zero values in normalized data
            zero_mask_norm = gdf_normalized[col] == 0
            
            # Plot non-zero normalized values
            gdf_normalized[~zero_mask_norm].plot(
                column=col, ax=ax_map_norm,
                legend=True, cmap='viridis_r',
                markersize=markersize, alpha=0.8
            )
            
            # Plot zero normalized values in gray
            if zero_mask_norm.sum() > 0:
                gdf_normalized[zero_mask_norm].plot(
                    ax=ax_map_norm, color='gray',
                    markersize=markersize, alpha=0.8
                )
            
            ax_map_norm.set_title("Normalized", fontsize=10)
            ax_map_norm.axis('off')
        # Add a main title for the group
        plt.suptitle(f"{group_name.upper()}", fontsize=16, y=0.98)
        
        # Adjust layout with more space between columns
        plt.tight_layout(rect=(0, 0, 1, 0.98), w_pad=2.0)
        
        # Save figure
        plt.savefig(f'{VISUALIZATIONS}/normalization/variable_group_{group_name}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()


    def normalize_columns(self, columns_mapping: dict, plot_multi=None) -> pd.DataFrame:
        """
        Normalize the specified columns in the DataFrame.
        
        Parameters:
            columns_mapping (dict): A dictionary where the keys are normalization methods 
                                    ('minmax', 'zscore', 'robust', 'log') and the values 
                                    are lists of column names to apply that method.
            plot (bool): If True, generate and display plots
            plot_individual (bool): If True, generate individual plots for each column
            plot_multi: Can be:
                        - True: Plot all variables in one grid
                        - dict: Dictionary mapping group names to lists of column names
                        - None: Don't create multi-variable plots
            multi_ncols (int): Number of columns in the multi-variable grid
                                        
        Returns:
            pd.DataFrame: A new DataFrame with the selected columns normalized.
        """
        # Create a dictionary to track which method was used for each column
        columns_methods = {}
        
        for method, cols in columns_mapping.items():
            for col in cols:
                if col not in self.normalized_df.columns:
                    raise KeyError(f"Column '{col}' not found in the DataFrame.")
                                
                # Normalize the column data
                self.normalized_df[col] = self.normalize_series(col, self.original_df[col], method)
                
                # Track the method used for this column
                columns_methods[col] = method
        
        # Plot multi-variable grid if requested
        if plot_multi is not None:
            # Plot each group of variables
            for group_name, columns in plot_multi.items():
                # if group_name in ['socio', 'age', 'education', 'economic', 'transport', 'public_transport', 'pois1', 'pois2']: continue
                print(f"Plotting {group_name}")
                self.plot_variable_group(group_name, columns, columns_methods)
                
        return self.normalized_df


    def plot_correlation_matrix(self):
        """
        Plots a correlation matrix for the given DataFrame with colors and correlation values using only matplotlib.

        Parameters:
        - df: DataFrame with numeric columns for which the correlation matrix will be calculated and plotted.
        """
        vars_order = [   
            'population',  
            'f', 'm', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+', 
            
            'education_primary', 'education_secondary', 'education_college',
            'household_avg_m2', 'income_2022_pers', 
            'unemployment', 'non_spanish_population', 

            'cars_abs', 'motos_abs', 'others_abs',  
            'bus_lines', 'metro_lines', 'tram_lines',
            'has_bike_lane',


            'n_health_care', 'n_culture', 'n_tourism', 'n_recreation', 'n_sport',
            'n_economic_retail', 'n_industrial', 'n_green', 'n_civic', 'n_worship',
            'n_education', 'n_superpois', 'pois_total', 'pois_entropy',

            'altitude']
        
        for string, df in zip(['raw', 'normalized'], [self.original_df, self.normalized_df]):

            # Calculate the correlation matrix
            df_copy = df.drop(['geometry'], axis=1)[vars_order]
            corr_matrix = df_copy.corr()

            # Create a new figure
            plt.figure(figsize=(20, 20))
            
            # Create the color map for the heatmap
            cax = plt.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            cbar = plt.colorbar(cax, aspect=5)
            cbar.ax.tick_params(labelsize=5)

            # Annotate each cell with the numeric value of the correlation coefficient
            for (i, j), val in np.ndenumerate(corr_matrix.values):
                if val >= 0.995:
                    text = 1
                else:
                    text = f'{val:.2f}'.replace('0.', '.')
                plt.text(j, i, text, ha='center', va='center', color='black', fontsize=3)

            vars = ['70+', 'education_college', 'non_spanish_population', 'has_bike_lane', 'pois_entropy']
            for var in vars:
                # Draw a horizontal line after each variable 
                var_index = corr_matrix.index.get_loc(var)
                plt.plot([-.5, var_index + 0.5], [var_index + 0.5, var_index + 0.5], color='black', alpha=0.7, lw=0.5)
                plt.plot([var_index + 0.5, var_index + 0.5], [-.5, var_index + 0.5], color='black', alpha=0.7, lw=0.5)

            # Set the labels for the x and y axis
            plt.xticks(np.arange(len(corr_matrix.columns)), list(corr_matrix.columns), rotation=90, fontsize=3)
            plt.yticks(np.arange(len(corr_matrix.index)), list(corr_matrix.index), fontsize=3)

            plt.savefig(f"{VISUALIZATIONS}/correlation_matrix_{string}.png", dpi=300)
            plt.close()


    def save_normalized_data(self):
        self.normalized_df = self.normalized_df.round(2)
        self.normalized_df.to_csv(self.normalized_df_filepath)
