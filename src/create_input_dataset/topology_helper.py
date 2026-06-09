import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import requests

# Set up project root and import project-specific paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from paths import VISUALIZATIONS  # noqa: E402


def create_grid(bounds, resolution):
    """
    Generate a uniform grid of latitude-longitude coordinates within bounds.

    Args:
        bounds (tuple): (min_lon, min_lat, max_lon, max_lat)
        resolution (float): Grid resolution in meters

    Returns:
        list[tuple[float, float]]: [(latitude, longitude), ...]
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Approximate conversion meters -> degrees
    lon_step = resolution / 111320
    lat_step = resolution / 110540

    lon_points = np.arange(min_lon, max_lon, lon_step)
    lat_points = np.arange(min_lat, max_lat, lat_step)

    grid_points = [(lat, lon) for lat in lat_points for lon in lon_points]
    return grid_points


def _fetch_batch_once(batch, api_url, timeout=30):
    query = "|".join(f"{lat},{lon}" for lat, lon in batch)
    response = requests.get(f"{api_url}?locations={query}", timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")

    data = response.json()
    results = data.get("results", [])
    vals = [res.get("elevation", None) for res in results]

    if len(vals) != len(batch):
        raise RuntimeError(
            f"Unexpected number of elevations returned: got {len(vals)}, expected {len(batch)}"
        )

    return vals


def get_altitude_batch(
    coordinates,
    batch_size=100,
    api_url="https://api.opentopodata.org/v1/srtm90m",
    max_retries=4,
    sleep_between_batches=0.5,
):
    """
    Fetch altitude values from OpenTopoData with retries.

    Args:
        coordinates (list[tuple]): [(latitude, longitude), ...]
        batch_size (int): batch size for API requests
        api_url (str): OpenTopoData endpoint
        max_retries (int): retries per batch
        sleep_between_batches (float): pause between successful batches

    Returns:
        list: altitude values aligned with input coordinates
    """
    results = []
    total_batches = (len(coordinates) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(coordinates), batch_size), start=1):
        batch = coordinates[start:start + batch_size]
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                vals = _fetch_batch_once(batch, api_url=api_url, timeout=30)
                results.extend(vals)
                non_null = sum(v is not None for v in vals)
                print(
                    f"[OK] batch {batch_idx}/{total_batches} "
                    f"attempt {attempt}/{max_retries} "
                    f"rows={len(batch)} non_null={non_null}"
                )
                success = True
                break
            except Exception as e:
                print(
                    f"[WARN] batch {batch_idx}/{total_batches} "
                    f"attempt {attempt}/{max_retries} failed: {e}"
                )
                time.sleep(1.5 * attempt)

        if not success:
            print(
                f"[ERROR] batch {batch_idx}/{total_batches} failed completely. "
                f"Filling {len(batch)} rows with None."
            )
            results.extend([None] * len(batch))

        time.sleep(sleep_between_batches)

    return results


def calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Compute slope percentage between two points.

    Returns:
        float: slope percentage, preserving sign
    """
    if pd.isna(alt1) or pd.isna(alt2):
        return np.nan

    R = 6371000  # meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    horizontal_distance = R * c

    if horizontal_distance == 0:
        return 0.0

    vertical_distance = alt2 - alt1
    return (vertical_distance / horizontal_distance) * 100


def compute_avg_max_slope(df):
    """
    Compute average and maximum slope for each grid point from its 4 neighbors.

    Args:
        df (pd.DataFrame): columns latitude, longitude, altitude

    Returns:
        pd.DataFrame: latitude, longitude, altitude, avg_slope, max_slope
    """
    grid = (
        df.pivot(index="latitude", columns="longitude", values="altitude")
        .sort_index(ascending=False)
    )

    if grid.empty:
        raise ValueError(
            "The grid is empty after pivoting. Ensure latitude and longitude values are valid."
        )

    avg_slopes = []
    max_slopes = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            slopes = []

            lat1, lon1, alt1 = grid.index[i], grid.columns[j], grid.iloc[i, j]
            if pd.isna(alt1):
                avg_slopes.append(np.nan)
                max_slopes.append(np.nan)
                continue

            if i > 0:  # North
                lat2, lon2, alt2 = grid.index[i - 1], lon1, grid.iloc[i - 1, j]
                slopes.append(calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2))

            if i < grid.shape[0] - 1:  # South
                lat2, lon2, alt2 = grid.index[i + 1], lon1, grid.iloc[i + 1, j]
                slopes.append(calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2))

            if j > 0:  # West
                lat2, lon2, alt2 = lat1, grid.columns[j - 1], grid.iloc[i, j - 1]
                slopes.append(calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2))

            if j < grid.shape[1] - 1:  # East
                lat2, lon2, alt2 = lat1, grid.columns[j + 1], grid.iloc[i, j + 1]
                slopes.append(calculate_slope_between_2_points(lat1, lon1, alt1, lat2, lon2, alt2))

            slopes = [s for s in slopes if not pd.isna(s)]
            avg_slopes.append(np.mean(slopes) if slopes else 0.0)
            max_slopes.append(np.max(slopes) if slopes else 0.0)

    results_df = pd.DataFrame(
        {
            "latitude": np.repeat(grid.index, grid.shape[1]),
            "longitude": np.tile(grid.columns, grid.shape[0]),
            "altitude": grid.values.flatten(),
            "avg_slope": avg_slopes,
            "max_slope": max_slopes,
        }
    ).sort_values(["latitude", "longitude"])

    return results_df


def visualize_altitude_and_slope(boundary, df):
    """
    Save a 2-panel plot for altitude and max slope.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    boundary.plot(ax=axs[0], color="none", edgecolor="black", linewidth=1)
    boundary.plot(ax=axs[1], color="none", edgecolor="black", linewidth=1)

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["longitude", "latitude"])

    scatter1 = axs[0].scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df["altitude"],
        cmap="viridis",
        s=5,
        alpha=0.7,
    )
    scatter2 = axs[1].scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df["max_slope"],
        cmap="viridis",
        s=5,
        alpha=0.7,
    )

    axs[0].axis("off")
    axs[1].axis("off")

    plt.colorbar(scatter1, ax=axs[0], label="Altitude (m)", shrink=0.6)
    plt.colorbar(scatter2, ax=axs[1], label="Slope (%)", shrink=0.6)

    outdir = Path(VISUALIZATIONS) / "raw_data"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "altitude_slope_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)