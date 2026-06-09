from pathlib import Path
import csv
import re
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


ROOT = Path(".")
TARGET_CRS = "EPSG:25831"

# ---------- INPUTS ----------
BICING_CSV = ROOT / "data/raw/bicing/bicing_station_information_2026_03.csv"

CENSUS_SHP = ROOT / "data/raw/ine/seccionado_2022/SECC_CE_20220101.shp"
NEIGHBORHOODS_CSV = ROOT / "data/raw/bcn/BarcelonaCiutat_Barris.csv"

INPUT_VARS = ROOT / "data/processed/input_variables"

# ---------- OUTPUTS ----------
OUTDIR = ROOT / "data/processed/stations"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "bicing_station_master_2026_03.csv"
OUT_GPKG = OUTDIR / "bicing_station_master_2026_03.gpkg"


SECTION_CSVS = [
    "socioeconomic_census_section_income.csv",
    "socioeconomic_census_section_population_sex_age.csv",
    "socioeconomic_census_section_education.csv",
    "socioeconomic_census_section_household_size.csv",
    "socioeconomic_census_section_non_spanish_population.csv",
    "socioeconomic_census_section_car_ownership.csv",
]

NEIGHBORHOOD_CSV = "socioeconomic_neighborhood_unemployment.csv"


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def read_csv_flexible(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python", low_memory=False)
    return clean_cols(df)


def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}")
    return None


def normalize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def normalize_census_section(x):
    """
    Formato estándar: string de 10 dígitos.
    Ejemplos:
      801902008.0 -> 0801902008
      0801902008  -> 0801902008
      801902008   -> 0801902008
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    try:
        f = float(s.replace(",", "."))
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            s = str(int(round(f)))
    except Exception:
        pass

    s = re.sub(r"\D", "", s)
    if not s:
        return None

    return s.zfill(10)


def parse_wkt_safe(val):
    if pd.isna(val):
        return None
    try:
        return wkt.loads(val)
    except Exception:
        return None


def collapse_duplicates(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if key_col not in df.columns:
        return df

    if not df[key_col].duplicated().any():
        return df

    numeric_cols = [c for c in df.columns if c != key_col and pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in df.columns if c != key_col and c not in numeric_cols]

    agg = {c: "mean" for c in numeric_cols}
    for c in other_cols:
        agg[c] = "first"

    return df.groupby(key_col, as_index=False).agg(agg)


def to_gdf_generic(path: Path) -> gpd.GeoDataFrame:
    df = read_csv_flexible(path)

    # geometry WKT
    if "geometry" in df.columns:
        geom = df["geometry"].map(parse_wkt_safe)
        gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geom)
        gdf = gdf[~gdf.geometry.isna()].copy()

        bounds = gdf.total_bounds
        if np.all(np.isfinite(bounds)) and -180 <= bounds[0] <= 180 and -90 <= bounds[1] <= 90:
            gdf.set_crs("EPSG:4326", inplace=True)
            return gdf.to_crs(TARGET_CRS)

        gdf.set_crs(TARGET_CRS, inplace=True)
        return gdf

    # x/y proyectadas
    x_col = pick_col(df, ["x", "X", "coord_x", "utm_x"], required=False)
    y_col = pick_col(df, ["y", "Y", "coord_y", "utm_y"], required=False)
    if x_col and y_col:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                pd.to_numeric(df[x_col], errors="coerce"),
                pd.to_numeric(df[y_col], errors="coerce"),
            ),
            crs=TARGET_CRS,
        )
        return gdf[~gdf.geometry.isna()].copy()

    # lon/lat
    lon_col = pick_col(df, ["lon", "longitude", "longtude", "lng"], required=False)
    lat_col = pick_col(df, ["lat", "latitude"], required=False)
    if lon_col and lat_col:
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        lat = pd.to_numeric(df[lat_col], errors="coerce")

        # si vienen escaladas
        if lat.abs().median() > 1000:
            lat = lat / 1_000_000
        if lon.abs().median() > 1000:
            lon = lon / 10_000_000

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(lon, lat),
            crs="EPSG:4326",
        )
        return gdf[~gdf.geometry.isna()].to_crs(TARGET_CRS)

    raise ValueError(f"No sé construir geometría para {path}")


def read_bicing_stations(path: Path) -> gpd.GeoDataFrame:
    df = read_csv_flexible(path)

    station_id_col = pick_col(df, ["station_id", "stationid", "id"])
    name_col = pick_col(df, ["name"], required=False)
    lat_col = pick_col(df, ["lat", "latitude"])
    lon_col = pick_col(df, ["lon", "longitude", "lng"])

    df["station_id"] = df[station_id_col].astype(str).str.strip()

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

    if lat.abs().median() > 1000:
        lat = lat / 1_000_000
    if lon.abs().median() > 1000:
        lon = lon / 10_000_000

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    keep = [
        c for c in [
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
        ]
        if c and c in gdf.columns
    ]

    gdf = gdf[keep].copy()
    if name_col and name_col != "name":
        gdf.rename(columns={name_col: "name"}, inplace=True)

    print(f"Estaciones únicas cargadas: {len(gdf)}")
    return gdf.reset_index(drop=True)


def read_census_sections(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs(TARGET_CRS)
    gdf = clean_cols(gdf)

    if "CMUN" in gdf.columns:
        gdf = gdf[gdf["CMUN"].astype(str) == "019"].copy()

    if "CUSEC" not in gdf.columns:
        raise KeyError("El shapefile de secciones no tiene CUSEC")

    gdf["census_section"] = gdf["CUSEC"].map(normalize_census_section)
    return gdf[["census_section", "geometry"]].copy()


def read_neighborhoods(path: Path) -> gpd.GeoDataFrame:
    df = read_csv_flexible(path)

    geom_col = pick_col(df, ["geometria_etrs89", "geometry"])
    name_col = pick_col(df, ["nom_barri", "neighborhood", "barri"])

    geom = df[geom_col].map(parse_wkt_safe)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=TARGET_CRS)
    gdf = gdf[~gdf.geometry.isna()].copy()

    gdf["neighborhood"] = df[name_col].astype(str).str.strip()
    gdf["neighborhood_key"] = gdf["neighborhood"].map(normalize_text)
    return gdf[["neighborhood", "neighborhood_key", "geometry"]].copy()


def _boundary_union(gdf):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def assign_census_section(stations: gpd.GeoDataFrame, census: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    left = stations.reset_index(drop=True).copy()
    right = census.copy()

    joined = gpd.sjoin(left, right, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")

    missing = joined["census_section"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing, ["geometry"]],
            right,
            how="left",
            distance_col="_dist_tmp",
        )
        joined.loc[missing, "census_section"] = nearest["census_section"].values

    joined["census_section"] = joined["census_section"].map(normalize_census_section)
    return joined


def assign_neighborhood(stations: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    left = stations.reset_index(drop=True).copy()
    right = neighborhoods.copy()

    joined = gpd.sjoin(
        left,
        right[["neighborhood", "neighborhood_key", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")

    missing = joined["neighborhood"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing, ["geometry"]],
            right[["neighborhood", "neighborhood_key", "geometry"]],
            how="left",
            distance_col="_dist_tmp",
        )
        joined.loc[missing, "neighborhood"] = nearest["neighborhood"].values
        joined.loc[missing, "neighborhood_key"] = nearest["neighborhood_key"].values

    joined["neighborhood_key"] = joined["neighborhood"].map(normalize_text)
    return joined


def merge_section_table_robust(stations: gpd.GeoDataFrame, csv_name: str) -> gpd.GeoDataFrame:
    path = INPUT_VARS / csv_name
    if not path.exists():
        print(f"[WARN] No existe {path}")
        return stations

    df = read_csv_flexible(path)
    if "census_section" not in df.columns:
        print(f"[WARN] {csv_name} no tiene census_section")
        return stations

    df["census_section_key"] = df["census_section"].map(normalize_census_section)
    df = df[~df["census_section_key"].isna()].copy()
    df = collapse_duplicates(df, "census_section_key")

    value_cols = [c for c in df.columns if c not in ["census_section", "census_section_key"]]

    return stations.merge(
        df[["census_section_key"] + value_cols],
        on="census_section_key",
        how="left"
    )


def merge_neighborhood_table_robust(stations: gpd.GeoDataFrame, csv_name: str) -> gpd.GeoDataFrame:
    path = INPUT_VARS / csv_name
    if not path.exists():
        print(f"[WARN] No existe {path}")
        return stations

    df = read_csv_flexible(path)
    nb_col = pick_col(df, ["neighborhood", "nom_barri"], required=False)
    if not nb_col:
        print(f"[WARN] {csv_name} no tiene neighborhood")
        return stations

    df["neighborhood_key"] = df[nb_col].map(normalize_text)
    df = df[~df["neighborhood_key"].isna()].copy()
    df = collapse_duplicates(df, "neighborhood_key")

    value_cols = [c for c in df.columns if any(k in c.lower() for k in ["unemployment", "atur"])]

    drop_cols = [c for c in stations.columns if any(k in c.lower() for k in ["unemployment", "atur"])]
    if drop_cols:
        stations = stations.drop(columns=drop_cols)

    return stations.merge(
        df[["neighborhood_key"] + value_cols],
        on="neighborhood_key",
        how="left"
    )


def buffer_feature_counts(stations, features, radius, prefix, line_candidates=None):
    if features.empty:
        stations[f"{prefix}_count_{radius}m"] = 0
        if line_candidates:
            stations[f"{prefix}_unique_lines_{radius}m"] = 0
        return stations

    buffers = stations[["station_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    joined = gpd.sjoin(features, buffers, how="inner", predicate="within")
    if joined.empty:
        stations[f"{prefix}_count_{radius}m"] = 0
        if line_candidates:
            stations[f"{prefix}_unique_lines_{radius}m"] = 0
        return stations

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
        else:
            out[f"{prefix}_unique_lines_{radius}m"] = 0

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
    agg = (
        pieces.groupby("station_id")["seg_len_m"]
        .sum()
        .rename(f"bike_lane_len_{radius}m")
        .reset_index()
    )

    out = stations.merge(agg, on="station_id", how="left")
    out[f"bike_lane_len_{radius}m"] = out[f"bike_lane_len_{radius}m"].fillna(0.0)
    return out


def mean_grid_values_in_buffer(stations, grid_gdf, radius=150):
    if grid_gdf.empty:
        stations[f"mean_altitude_{radius}m"] = np.nan
        stations[f"mean_slope_{radius}m"] = np.nan
        return stations

    # altitude viene como altitude
    alt_col = pick_col(grid_gdf, ["altitude", "elevation"], required=False)

    # nombre correcto del slope en tu topology
    slope_col = pick_col(
        grid_gdf,
        ["avg_slope", "slope", "slope_pct", "max_slope"],
        required=False
    )

    buffers = stations[["station_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    joined = gpd.sjoin(grid_gdf, buffers, how="inner", predicate="within")
    out = stations.copy()

    if joined.empty:
        out[f"mean_altitude_{radius}m"] = np.nan
        out[f"mean_slope_{radius}m"] = np.nan
        return out

    if alt_col:
        alt = (
            joined.groupby("station_id")[alt_col]
            .mean()
            .rename(f"mean_altitude_{radius}m")
            .reset_index()
        )
        out = out.merge(alt, on="station_id", how="left")
    else:
        out[f"mean_altitude_{radius}m"] = np.nan

    if slope_col:
        slope = (
            joined.groupby("station_id")[slope_col]
            .mean()
            .rename(f"mean_slope_{radius}m")
            .reset_index()
        )
        out = out.merge(slope, on="station_id", how="left")
    else:
        out[f"mean_slope_{radius}m"] = np.nan

    return out


def main():
    print("Cargando estaciones de Bicing...")
    stations = read_bicing_stations(BICING_CSV)

    print("Asignando sección censal y barrio...")
    census = read_census_sections(CENSUS_SHP)
    neighborhoods = read_neighborhoods(NEIGHBORHOODS_CSV)

    stations = assign_census_section(stations, census)
    stations = assign_neighborhood(stations, neighborhoods)

    stations["census_section_key"] = stations["census_section"].map(normalize_census_section)
    stations["neighborhood_key"] = stations["neighborhood"].map(normalize_text)

    print("Uniendo bloque socioeconómico por sección censal...")
    for csv_name in SECTION_CSVS:
        stations = merge_section_table_robust(stations, csv_name)

    print("Uniendo paro por barrio...")
    stations = merge_neighborhood_table_robust(stations, NEIGHBORHOOD_CSV)

    print("Procesando infraestructura...")
    bus = to_gdf_generic(INPUT_VARS / "infrastructure_bus_stops_lines.csv")
    metro = to_gdf_generic(INPUT_VARS / "infrastructure_metro_entrances_stops_lines.csv")
    tram = to_gdf_generic(INPUT_VARS / "infrastructure_tram_entrances_lines.csv")

    stations = buffer_feature_counts(stations, bus, radius=300, prefix="bus", line_candidates=["line", "route"])
    stations = buffer_feature_counts(stations, metro, radius=500, prefix="metro", line_candidates=["line", "route"])
    stations = buffer_feature_counts(stations, tram, radius=500, prefix="tram", line_candidates=["line", "route"])

    stations = nearest_distance(stations, bus, "bus")
    stations = nearest_distance(stations, metro, "metro")
    stations = nearest_distance(stations, tram, "tram")

    print("Procesando POIs...")
    pois = to_gdf_generic(INPUT_VARS / "urbanism_pois.csv")
    stations = buffer_feature_counts(stations, pois, radius=300, prefix="poi")

    print("Procesando carriles bici...")
    bike_lanes = to_gdf_generic(INPUT_VARS / "urbanism_bike_lanes_epsg_25831.csv")
    stations = bike_lane_length_in_buffer(stations, bike_lanes, radius=300)
    stations = nearest_distance(stations, bike_lanes, "bike_lane")

    print("Procesando topografía...")
    topo = to_gdf_generic(INPUT_VARS / "topology_altitude_slope_50.csv")
    stations = mean_grid_values_in_buffer(stations, topo, radius=150)

    socio_cols = [
        c for c in stations.columns
        if any(k in c.lower() for k in [
            "income", "education", "population", "household",
            "non_spanish", "car", "moto", "others", "unemployment"
        ])
    ]
    if socio_cols:
        print("\nCobertura bloque socioeconómico:")
        coverage = (1 - stations[socio_cols].isna().mean()).sort_values()
        print(coverage.to_string())

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

    technical = ["census_section_key", "neighborhood_key"]

    others = [c for c in stations.columns if c not in preferred + technical + ["geometry"]]
    stations = stations[[c for c in preferred if c in stations.columns] + others + ["geometry"]]

    # Export CSV sin warning de geometry
    stations_csv = pd.DataFrame(stations.drop(columns=["geometry"]).copy())
    stations_csv["geometry"] = stations.geometry.to_wkt()
    stations_csv.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    stations.to_file(OUT_GPKG, layer="stations", driver="GPKG")

    print(f"\nGuardado CSV:  {OUT_CSV}")
    print(f"Guardado GPKG: {OUT_GPKG}")
    print(f"Filas finales: {len(stations)}")


if __name__ == "__main__":
    main()