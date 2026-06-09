import folium  # type: ignore
import branca.colormap as cm  # type: ignore
import numpy as np
import pandas as pd


def add_variable_areas_to_map(m, df, weights, z_index=None):
    """Add areas for each variable with non-zero weights as separate layers
    
    Args:
        m (folium.Map): The map to add areas to
        df (GeoDataFrame): DataFrame with spatial areas
        weights (dict): Dictionary of weights for variables
        z_index (int, optional): Z-index for the layer (higher values appear on top)
    """
    # Get variables with non-zero weights
    selected_vars = {k: v for k, v in weights.items() if v != 0 and k in df.columns}
    
    # Create a feature group for each variable
    for var, weight in selected_vars.items():
        print(var, weight)
        # Create a feature group for this variable - set show=False to hide by default
        var_group = folium.FeatureGroup(name=f'{var} (w={weight:.3f})', show=False)
        
        # Get min and max values, ensuring they are valid for colormap
        if df[var].notna().any():
            valid_values = df[var].dropna()
            if len(valid_values) > 0:
                vmin = float(valid_values.min())
                vmax = float(valid_values.max())
                
                # Ensure min and max are different to avoid colormap errors
                if vmin == vmax:
                    vmax = vmin + 0.0001
                
                # Create a colormap for this variable - blue shades, darker for higher values
                var_colormap = cm.LinearColormap(
                    colors=['#E6F3FF', '#94C4F5', '#4282C2', '#2C5A8C', '#14325C'],
                    vmin=vmin,
                    vmax=vmax
                )
                
                territory_type = 'neighborhood' if 'neighborhood' in df.columns else 'census_section'


                # Add each area to the map
                for idx, row in df.iterrows():

                    # Skip if value is NaN
                    if pd.isna(row[var]):
                        continue
                        
                    # Create popup with variable value
                    popup_html = f'''
                    <div style="font-family: Arial, sans-serif;">
                        <p>{row[territory_type]}: {row[var]:.3f}</p>
                    </div>
                    '''
                    
                    # Ensure value is within bounds
                    value = float(row[var])
                    value = max(vmin, min(vmax, value))
                    
                    # Define style with z-index if provided
                    style_function = lambda x, value=value, cmap=var_colormap, z=z_index: {
                        'fillColor': cmap(value),
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.5,
                        'zIndex': z if z is not None else None
                    }
                    
                    # Add the area to the feature group with an id for styling
                    area_id = f"var_{var.replace(' ', '_')}_{idx}"
                    geo_json = folium.GeoJson(
                        row.geometry,
                        style_function=style_function,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(var_group)
                
                # Add the feature group to the map
                var_group.add_to(m)
    
    return m