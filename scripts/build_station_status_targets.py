from pathlib import Path
import pandas as pd


ROOT = Path(".")

INFO_CSV = ROOT / "data/raw/bicing/bicing_station_information_2026_03.csv"
STATUS_CSV = ROOT / "data/raw/bicing/bicing_station_status_2026_03.csv"

OUTDIR = ROOT / "data/processed/stations"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "bicing_station_status_targets_2026_03.csv"


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}")
    return None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    # Primero intenta separador coma
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if len(df.columns) == 1:
        # Si todo quedó en una sola columna, prueba con autodetección
        df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python", low_memory=False)
    return clean_cols(df)


def read_station_info(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)

    station_id_col = pick_col(df, ["station_id", "stationid", "id"])
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

    cap_col = pick_col(df, ["capacity"], required=False)

    keep = ["station_id"]
    if cap_col:
        keep.append(cap_col)

    out = df[keep].copy()

    if cap_col and cap_col != "capacity":
        out.rename(columns={cap_col: "capacity"}, inplace=True)

    if "capacity" in out.columns:
        out["capacity"] = pd.to_numeric(out["capacity"], errors="coerce")

    return out


def read_station_status(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)

    station_id_col = pick_col(df, ["station_id", "stationid", "id"])
    df["station_id"] = df[station_id_col].astype(str).str.strip()

    ts_col = pick_col(df, ["last_reported", "last_updated", "timestamp"], required=False)

    bikes_col = pick_col(
        df,
        [
            "num_bikes_available",
            "num_bikes_available_total",
            "bikes_available",
        ],
        required=False,
    )

    docks_col = pick_col(
        df,
        [
            "num_docks_available",
            "docks_available",
            "num_free_docks",
        ],
        required=False,
    )

    mech_col = pick_col(
        df,
        [
            "num_bikes_available_types.mechanical",
            "num_bikes_available_types_mechanical",
            "mechanical",
            "num_mechanical_bikes",
        ],
        required=False,
    )

    ebike_col = pick_col(
        df,
        [
            "num_bikes_available_types.ebike",
            "num_bikes_available_types_ebike",
            "ebike",
            "num_ebikes",
        ],
        required=False,
    )

    is_installed_col = pick_col(df, ["is_installed"], required=False)
    is_renting_col = pick_col(df, ["is_renting"], required=False)
    is_returning_col = pick_col(df, ["is_returning"], required=False)
    status_col = pick_col(df, ["status"], required=False)

    keep = ["station_id"]
    for c in [
        ts_col,
        bikes_col,
        docks_col,
        mech_col,
        ebike_col,
        is_installed_col,
        is_renting_col,
        is_returning_col,
        status_col,
    ]:
        if c:
            keep.append(c)

    out = df[keep].copy()

    rename_map = {}
    if ts_col and ts_col != "last_reported":
        rename_map[ts_col] = "last_reported"
    if bikes_col and bikes_col != "num_bikes_available":
        rename_map[bikes_col] = "num_bikes_available"
    if docks_col and docks_col != "num_docks_available":
        rename_map[docks_col] = "num_docks_available"
    if mech_col and mech_col != "num_mechanical_bikes":
        rename_map[mech_col] = "num_mechanical_bikes"
    if ebike_col and ebike_col != "num_ebikes":
        rename_map[ebike_col] = "num_ebikes"

    out.rename(columns=rename_map, inplace=True)

    for c in [
        "last_reported",
        "num_bikes_available",
        "num_docks_available",
        "num_mechanical_bikes",
        "num_ebikes",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Quédate solo con estaciones operativas si existe la info
    if "status" in out.columns:
        out["status"] = out["status"].astype(str).str.strip()
    if "is_installed" in out.columns:
        out["is_installed"] = pd.to_numeric(out["is_installed"], errors="coerce")
    if "is_renting" in out.columns:
        out["is_renting"] = pd.to_numeric(out["is_renting"], errors="coerce")
    if "is_returning" in out.columns:
        out["is_returning"] = pd.to_numeric(out["is_returning"], errors="coerce")

    if "status" in out.columns:
        out = out[out["status"].isin(["IN_SERVICE", "ACTIVE", "OPERATIONAL"])].copy()

    if "is_installed" in out.columns:
        out = out[out["is_installed"] == 1].copy()
    if "is_renting" in out.columns:
        out = out[out["is_renting"] == 1].copy()
    if "is_returning" in out.columns:
        out = out[out["is_returning"] == 1].copy()

    dedup_cols = [
        c
        for c in ["station_id", "last_reported", "num_bikes_available", "num_docks_available"]
        if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols).copy()

    return out


def main():
    print("Leyendo station_information...")
    info = read_station_info(INFO_CSV)
    print(f"Estaciones únicas en info: {len(info)}")

    print("Leyendo station_status...")
    status = read_station_status(STATUS_CSV)
    print(f"Observaciones de estado: {len(status)}")

    df = status.merge(info, on="station_id", how="left")

    # Ratios respecto a capacidad
    if "capacity" in df.columns and "num_bikes_available" in df.columns:
        df["bikes_ratio_capacity"] = df["num_bikes_available"] / df["capacity"]

    if "capacity" in df.columns and "num_docks_available" in df.columns:
        df["docks_ratio_capacity"] = df["num_docks_available"] / df["capacity"]

    # Flags operativos
    if "num_bikes_available" in df.columns:
        df["is_empty"] = (df["num_bikes_available"] <= 0).astype(int)
        df["is_near_empty"] = (df["num_bikes_available"] <= 2).astype(int)

    if "num_docks_available" in df.columns:
        df["is_full"] = (df["num_docks_available"] <= 0).astype(int)
        df["is_near_full"] = (df["num_docks_available"] <= 2).astype(int)

    metric_aggs = {
        "last_reported": ["count"],
        "num_bikes_available": ["mean", "std", "min", "max"],
        "num_docks_available": ["mean", "std", "min", "max"],
        "num_mechanical_bikes": ["mean"],
        "num_ebikes": ["mean"],
        "bikes_ratio_capacity": ["mean", "std"],
        "docks_ratio_capacity": ["mean", "std"],
        "is_empty": ["mean"],
        "is_near_empty": ["mean"],
        "is_full": ["mean"],
        "is_near_full": ["mean"],
        "capacity": ["first"],
    }

    usable_aggs = {k: v for k, v in metric_aggs.items() if k in df.columns}

    out = df.groupby("station_id").agg(usable_aggs)

    out.columns = [
        f"{col}_{stat}" if stat else col
        for col, stat in out.columns.to_flat_index()
    ]
    out = out.reset_index()

    rename_final = {
        "last_reported_count": "n_observations",
        "num_bikes_available_mean": "bikes_available_mean",
        "num_bikes_available_std": "bikes_available_std",
        "num_bikes_available_min": "bikes_available_min",
        "num_bikes_available_max": "bikes_available_max",
        "num_docks_available_mean": "docks_available_mean",
        "num_docks_available_std": "docks_available_std",
        "num_docks_available_min": "docks_available_min",
        "num_docks_available_max": "docks_available_max",
        "num_mechanical_bikes_mean": "mechanical_bikes_mean",
        "num_ebikes_mean": "ebikes_mean",
        "bikes_ratio_capacity_mean": "bikes_ratio_capacity_mean",
        "bikes_ratio_capacity_std": "bikes_ratio_capacity_std",
        "docks_ratio_capacity_mean": "docks_ratio_capacity_mean",
        "docks_ratio_capacity_std": "docks_ratio_capacity_std",
        "is_empty_mean": "pct_empty",
        "is_near_empty_mean": "pct_near_empty",
        "is_full_mean": "pct_full",
        "is_near_full_mean": "pct_near_full",
        "capacity_first": "capacity",
    }
    out.rename(columns=rename_final, inplace=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nGuardado en: {OUT_CSV}")
    print(f"Estaciones agregadas: {len(out)}")
    print(out.head())


if __name__ == "__main__":
    main()