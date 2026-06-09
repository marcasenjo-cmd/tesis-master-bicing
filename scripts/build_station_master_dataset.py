from pathlib import Path
import ast

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


ROOT = Path(".")
BICING_CSV = ROOT / "data/raw/bicing/bicing_station_information_2026_03.csv"

CENSUS_SHP = ROOT / "data/raw/ine/seccionado_2022/SECC_CE_20220101.shp"
NEIGHBORHOODS_CSV = ROOT / "data/raw/bcn/BarcelonaCiutat_Barris.csv"

INPUT_VARS = ROOT / "data/processed/input_variables"
OUTDIR = ROOT / "data/processed/stations"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "bicing_station_master_2026_03.csv"
OUT_GPKG = OUTDIR / "bicing_station_master_2026_03.gpkg"

TARGET_CRS = "EPSG:25831"


def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}")
    return None


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def parse_wkt_safe(val):
    if pd.isna(val):
        return None
    try:
        return wkt.loads(val)
    except Exception:
        return None


def to_gdf_generic(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)

    if "geometry" in df.columns:
        geom = df["geometry"].map(parse_wkt_safe)
        gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geom)
        gdf = gdf[~gdf.geometry.isna()].copy()

        bounds = gdf.total_bounds
        if np.all(np.isfinite(bounds)) and -180 <= bounds[0] <= 180 and -90 <= bounds[1] <= 90:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf.set_crs(TARGET_CRS, inplace=True)
        return gdf.to_crs(TARGET_CRS)

    x_col = pick_col(df, ["x", "X", "coord_x", "utm_x"], required=False)
    y_col = pick_col(df, ["y", "Y", "coord_y", "utm_y"], required=False)
    lon_col = pick_col(df, ["lon", "longitude", "lng"], required=False)
    lat_col = pick_col(df, ["lat", "latitude"], required=False)

    if x_col and y_col:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(pd.to_numeric(df[x_col], errors="coerce"),
                                        pd.to_numeric(df[y_col], errors="coerce")),
            crs=TARGET_CRS,
        )
        return gdf

    if lon_col and lat_col:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(pd.to_numeric(df[lon_col], errors="coerce"),
                                        pd.to_numeric(df[lat_col], errors="coerce")),
            crs="EPSG:4326",
        )
        return gdf.to_crs(TARGET_CRS)

    raise ValueError(f"No sé construir geometría para {path}")


def read_bicing_stations(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = clean_cols(df)

    station_id_col = pick_col(df, ["station_id", "stationid", "id"])
    name_col = pick_col(df, ["name"], required=False)
    lat_col = pick_col(df, ["lat", "latitude"])
    lon_col = pick_col(df, ["lon", "longitude", "lng"])

    df["station_id"] = df[station_id_col].astype(str).str.strip()

    # Quedarse con la última observación por estación
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_numeric(df["last_updated"], errors="coerce")
        df = (
            df.sort_values(["station_id", "last_updated"])
              .drop_duplicates(subset="station_id", keep="last")
              .copy()
        )
    else:
        df = df.drop_duplicates(subset="station_id", keep="last").copy()

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    # Corrige los microgrados típicos del CSV histórico de Bicing
    if lat.abs().median() > 1000:
        lat = lat / 1_000_000
    if lon.abs().median() > 1000:
        lon = lon / 10_000_000

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    keep = [c for c in [
        "station_id",
        name_col,
        "physical_configuration",
        "address",
        "cross_street",
        "post_code",
        "capacity",
        "is_charging_station",
        "short_name",
        "nearby_distance",
        "last_updated",
        "geometry",
    ] if c and c in gdf.columns]

    gdf = gdf[keep].copy()
    if name_col and name_col != "name":
        gdf.rename(columns={name_col: "name"}, inplace=True)

    print(f"Estaciones únicas cargadas: {len(gdf)}")
    return gdf


def read_census_sections(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs(TARGET_CRS)
    if "CMUN" in gdf.columns:
        gdf = gdf[gdf["CMUN"].astype(str) == "019"].copy()
    gdf["census_section"] = gdf["CUSEC"].astype(str).str.strip()
    return gdf[["census_section", "geometry"]].copy()


def read_neighborhoods(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)

    geom_col = pick_col(df, ["geometria_etrs89", "geometry"])
    name_col = pick_col(df, ["nom_barri", "neighborhood", "barri"])

    if geom_col == "geometry":
        geom = df["geometry"].map(parse_wkt_safe)
    else:
        geom = df[geom_col].map(parse_wkt_safe)

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=TARGET_CRS)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf["neighborhood"] = gdf[name_col].astype(str).str.strip()

    return gdf[["neighborhood", "geometry"]].copy()


def merge_section_table(stations: gpd.GeoDataFrame, csv_name: str) -> gpd.GeoDataFrame:
    path = INPUT_VARS / csv_name
    if not path.exists():
        print(f"[WARN] No existe {path}")
        return stations

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)
    if "census_section" not in df.columns:
        print(f"[WARN] {csv_name} no tiene census_section")
        return stations

    df["census_section"] = df["census_section"].astype(str).str.strip()
    return stations.merge(df, on="census_section", how="left")


def merge_neighborhood_table(stations: gpd.GeoDataFrame, csv_name: str) -> gpd.GeoDataFrame:
    path = INPUT_VARS / csv_name
    if not path.exists():
        print(f"[WARN] No existe {path}")
        return stations

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_cols(df)
    nb_col = pick_col(df, ["neighborhood", "nom_barri"], required=False)
    if not nb_col:
        print(f"[WARN] {csv_name} no tiene neighborhood")
        return stations

    df["neighborhood"] = df[nb_col].astype(str).str.strip()
    if nb_col != "neighborhood":
        df = df.drop(columns=[nb_col])

    return stations.merge(df, on="neighborhood", how="left")


def buffer_feature_counts(stations, features, radius, prefix, line_candidates=None):
    if features.empty:
        stations[f"{prefix}_count_{radius}m"] = 0
        return stations

    buffers = stations[["station_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    joined = gpd.sjoin(features, buffers, how="inner", predicate="within")
    counts = joined.groupby("station_id").size().rename(f"{prefix}_count_{radius}m").reset_index()
    out = stations.merge(counts, on="station_id", how="left")
    out[f"{prefix}_count_{radius}m"] = out[f"{prefix}_count_{radius}m"].fillna(0).astype(int)

    if line_candidates:
        line_col = pick_col(joined, line_candidates, required=False)
        if line_col:
            unique_lines = (
                joined.groupby("station_id")[line_col]
                .nunique()
                .rename(f"{prefix}_unique_lines_{radius}m")
                .reset_index()
            )
            out = out.merge(unique_lines, on="station_id", how="left")
            out[f"{prefix}_unique_lines_{radius}m"] = out[f"{prefix}_unique_lines_{radius}m"].fillna(0).astype(int)

    return out


def nearest_distance(stations, features, prefix):
    if features.empty:
        stations[f"{prefix}_nearest_m"] = np.nan
        return stations

    left = stations[["station_id", "geometry"]].copy()
    right = features[["geometry"]].copy()

    nearest = gpd.sjoin_nearest(left, right, how="left", distance_col=f"{prefix}_nearest_m")
    nearest = nearest[["station_id", f"{prefix}_nearest_m"]].drop_duplicates(subset=["station_id"])
    return stations.merge(nearest, on="station_id", how="left")


def bike_lane_length_in_buffer(stations, bike_lanes, radius=300):
    if bike_lanes.empty:
        stations[f"bike_lane_len_{radius}m"] = 0.0
        return stations

    buffers = stations[["station_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    pieces = gpd.overlay(bike_lanes[["geometry"]], buffers, how="intersection")
    if pieces.empty:
        stations[f"bike_lane_len_{radius}m"] = 0.0
        return stations

    pieces["seg_len_m"] = pieces.geometry.length
    agg = pieces.groupby("station_id")["seg_len_m"].sum().rename(f"bike_lane_len_{radius}m").reset_index()

    out = stations.merge(agg, on="station_id", how="left")
    out[f"bike_lane_len_{radius}m"] = out[f"bike_lane_len_{radius}m"].fillna(0.0)
    return out


def mean_grid_values_in_buffer(stations, grid_gdf, radius=150):
    if grid_gdf.empty:
        stations[f"mean_altitude_{radius}m"] = np.nan
        stations[f"mean_slope_{radius}m"] = np.nan
        return stations

    alt_col = pick_col(grid_gdf, ["altitude", "elevation"], required=False)
    slope_col = pick_col(grid_gdf, ["slope", "slope_pct"], required=False)

    buffers = stations[["station_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    joined = gpd.sjoin(grid_gdf, buffers, how="inner", predicate="within")
    if joined.empty:
        stations[f"mean_altitude_{radius}m"] = np.nan
        stations[f"mean_slope_{radius}m"] = np.nan
        return stations

    out = stations.copy()

    if alt_col:
        alt = joined.groupby("station_id")[alt_col].mean().rename(f"mean_altitude_{radius}m").reset_index()
        out = out.merge(alt, on="station_id", how="left")

    if slope_col:
        slope = joined.groupby("station_id")[slope_col].mean().rename(f"mean_slope_{radius}m").reset_index()
        out = out.merge(slope, on="station_id", how="left")

    return out


def main():
    stations = read_bicing_stations(BICING_CSV)
    print(f"Estaciones cargadas: {len(stations)}")

    census = read_census_sections(CENSUS_SHP)
    neighborhoods = read_neighborhoods(NEIGHBORHOODS_CSV)

    stations = gpd.sjoin(
        stations,
        census,
        how="left",
        predicate="within"
    ).drop(columns=["index_right"])

    stations = gpd.sjoin(
        stations,
        neighborhoods,
        how="left",
        predicate="within"
    ).drop(columns=["index_right"])

    # ---- Socioeconómico por sección censal ----
    section_csvs = [
        "socioeconomic_census_section_income.csv",
        "socioeconomic_census_section_population_sex_age.csv",
        "socioeconomic_census_section_education.csv",
        "socioeconomic_census_section_household_size.csv",
        "socioeconomic_census_section_non_spanish_population.csv",
        "socioeconomic_census_section_car_ownership.csv",
    ]
    for csv_name in section_csvs:
        stations = merge_section_table(stations, csv_name)

    # ---- Socioeconómico por barrio ----
    stations = merge_neighborhood_table(
        stations,
        "socioeconomic_neighborhood_unemployment.csv"
    )

    # ---- Infraestructura ----
    bus = to_gdf_generic(INPUT_VARS / "infrastructure_bus_stops_lines.csv")
    metro = to_gdf_generic(INPUT_VARS / "infrastructure_metro_entrances_stops_lines.csv")
    tram = to_gdf_generic(INPUT_VARS / "infrastructure_tram_entrances_lines.csv")

    stations = buffer_feature_counts(
        stations, bus, radius=300, prefix="bus", line_candidates=["line", "route"]
    )
    stations = buffer_feature_counts(
        stations, metro, radius=500, prefix="metro", line_candidates=["line", "route"]
    )
    stations = buffer_feature_counts(
        stations, tram, radius=500, prefix="tram", line_candidates=["line", "route"]
    )

    stations = nearest_distance(stations, bus, "bus")
    stations = nearest_distance(stations, metro, "metro")
    stations = nearest_distance(stations, tram, "tram")

    # ---- POIs ----
    pois = to_gdf_generic(INPUT_VARS / "urbanism_pois.csv")
    stations = buffer_feature_counts(stations, pois, radius=300, prefix="poi")

    # ---- Carriles bici ----
    bike_lanes = to_gdf_generic(INPUT_VARS / "urbanism_bike_lanes_epsg_25831.csv")
    stations = bike_lane_length_in_buffer(stations, bike_lanes, radius=300)
    stations = nearest_distance(stations, bike_lanes, "bike_lane")

    # ---- Topografía ----
    topo = to_gdf_generic(INPUT_VARS / "topology_altitude_slope_50.csv")
    stations = mean_grid_values_in_buffer(stations, topo, radius=150)

    # Orden sugerido de columnas
    preferred = [
        "station_id", "name", "address", "post_code", "capacity",
        "census_section", "neighborhood",
        "bus_count_300m", "bus_unique_lines_300m", "bus_nearest_m",
        "metro_count_500m", "metro_unique_lines_500m", "metro_nearest_m",
        "tram_count_500m", "tram_unique_lines_500m", "tram_nearest_m",
        "poi_count_300m",
        "bike_lane_len_300m", "bike_lane_nearest_m",
        "mean_altitude_150m", "mean_slope_150m",
    ]

    others = [c for c in stations.columns if c not in preferred + ["geometry"]]
    stations = stations[[c for c in preferred if c in stations.columns] + others + ["geometry"]]

    # CSV y GPKG
    stations_csv = stations.copy()
    stations_csv["geometry"] = stations_csv.geometry.to_wkt()
    stations_csv.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    stations.to_file(OUT_GPKG, layer="stations", driver="GPKG")

    print(f"\nGuardado CSV:  {OUT_CSV}")
    print(f"Guardado GPKG: {OUT_GPKG}")
    print(f"Filas finales: {len(stations)}")


if __name__ == "__main__":
    main()