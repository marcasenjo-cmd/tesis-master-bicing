from pathlib import Path
import pandas as pd


ROOT = Path(".")

MASTER_CSV = ROOT / "data/processed/stations/bicing_station_master_2026_03.csv"
TARGETS_CSV = ROOT / "data/processed/stations/bicing_station_status_targets_2026_03.csv"

OUTDIR = ROOT / "data/processed/stations"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "bicing_station_modeling_2026_03.csv"


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def main():
    print("Leyendo master dataset...")
    master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    master = clean_cols(master)

    print("Leyendo targets operativos...")
    targets = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig", low_memory=False)
    targets = clean_cols(targets)

    if "station_id" not in master.columns:
        raise KeyError("El master dataset no tiene columna 'station_id'")
    if "station_id" not in targets.columns:
        raise KeyError("El dataset de targets no tiene columna 'station_id'")

    master["station_id"] = master["station_id"].astype(str).str.strip()
    targets["station_id"] = targets["station_id"].astype(str).str.strip()

    print(f"Filas master: {len(master)}")
    print(f"Filas targets: {len(targets)}")
    print(f"Estaciones únicas master: {master['station_id'].nunique()}")
    print(f"Estaciones únicas targets: {targets['station_id'].nunique()}")

    # Nos quedamos solo con estaciones presentes en ambos
    df = master.merge(
        targets,
        on="station_id",
        how="inner",
        suffixes=("", "_target")
    )

    # Para modelado tabular, normalmente no interesa geometry
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    # Quita columnas redundantes si existen
    redundant = [c for c in ["capacity_target"] if c in df.columns]
    if redundant:
        df = df.drop(columns=redundant)

    preferred = [
        "station_id",
        "name",
        "address",
        "post_code",
        "capacity",
        "census_section",
        "neighborhood",
        "bikes_ratio_capacity_mean",
        "pct_empty",
        "pct_near_empty",
        "pct_full",
        "pct_near_full",
        "bikes_available_mean",
        "docks_available_mean",
        "n_observations",
    ]

    existing_preferred = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_preferred]
    df = df[existing_preferred + remaining]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nGuardado en: {OUT_CSV}")
    print(f"Filas finales: {len(df)}")
    print(f"Estaciones finales: {df['station_id'].nunique()}")

    if "bikes_ratio_capacity_mean" in df.columns:
        print("\nTarget principal sugerido: bikes_ratio_capacity_mean")

    print("\nPrimeras columnas:")
    print(df.columns.tolist()[:25])


if __name__ == "__main__":
    main()