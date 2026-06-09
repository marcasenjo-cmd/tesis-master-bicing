from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT_CSV = ROOT / "data/processed/stations/bicing_station_modeling_2026_03.csv"
OUTDIR = ROOT / "data/processed/stations"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "bicing_station_explanatory_score_2026_03.csv"
OUT_FEATURES_CSV = OUTDIR / "bicing_station_explanatory_score_features_2026_03.csv"


# higher_is_better = True  -> más valor = mejor score
# higher_is_better = False -> más valor = peor score, se invierte
DIMENSIONS = {
    "accessibility": {
        "weight": 0.35,
        "features": [
            ("bus_unique_lines_300m", True),
            ("metro_count_500m", True),
            ("metro_nearest_m", False),
            ("tram_nearest_m", False),
            ("bike_lane_len_300m", True),
            ("bus_nearest_m", False),
        ],
    },
    "activity": {
        "weight": 0.25,
        "features": [
            ("poi_count_300m", True),
        ],
    },
    "socioeconomic": {
        "weight": 0.25,
        "features": [
            ("total_population", True),
            ("income_2022_house", True),
            ("education_college", True),
            ("unemployment_percentage", False),
        ],
    },
    "topography": {
        "weight": 0.15,
        "features": [
            ("mean_slope_150m", False),
        ],
    },
}


META_COLS = [
    "station_id",
    "name",
    "address",
    "post_code",
    "capacity",
    "census_section",
    "neighborhood",
]

OPTIONAL_KEEP_COLS = [
    "bikes_ratio_capacity_mean",
    "pct_empty",
    "pct_near_empty",
    "pct_full",
    "pct_near_full",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Escala robusta 0-1 por ranking percentil.
    0 = peor, 1 = mejor
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = pd.to_numeric(series, errors="coerce").dropna()

    if valid.empty:
        return out

    if len(valid) == 1:
        out.loc[valid.index] = 1.0
        return out

    ranks = valid.rank(method="average")
    scaled = (ranks - 1) / (len(valid) - 1)

    if not higher_is_better:
        scaled = 1.0 - scaled

    out.loc[valid.index] = scaled
    return out


def weighted_row_score(df: pd.DataFrame, score_cols: list[str], weights: list[float]) -> pd.Series:
    """
    Media ponderada por fila, renormalizando si falta alguna dimensión.
    """
    if not score_cols:
        return pd.Series(np.nan, index=df.index, dtype=float)

    weights_arr = np.array(weights, dtype=float)
    values = df[score_cols].to_numpy(dtype=float)
    valid = ~np.isnan(values)

    weighted_values = np.where(valid, values * weights_arr, 0.0)
    weighted_sum = weighted_values.sum(axis=1)
    used_weights = np.where(valid, weights_arr, 0.0).sum(axis=1)

    out = np.where(used_weights > 0, weighted_sum / used_weights, np.nan)
    return pd.Series(out, index=df.index, dtype=float)


def categorize_score(score_0_100: pd.Series) -> pd.Series:
    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    return pd.cut(score_0_100, bins=bins, labels=labels)


def main():
    print("Leyendo dataset de modelado...")
    df = read_csv(INPUT_CSV)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")

    feature_report_rows = []
    dimension_score_cols = []
    dimension_weights = []

    # Crear scores por feature
    for dim_name, dim_cfg in DIMENSIONS.items():
        available_feature_score_cols = []

        for col, higher_is_better in dim_cfg["features"]:
            exists = col in df.columns
            non_null_ratio = float(1 - df[col].isna().mean()) if exists else 0.0

            feature_report_rows.append(
                {
                    "dimension": dim_name,
                    "feature": col,
                    "exists_in_dataset": exists,
                    "coverage_ratio": non_null_ratio,
                    "higher_is_better": higher_is_better,
                    "dimension_weight": dim_cfg["weight"],
                }
            )

            if not exists:
                print(f"[WARN] No existe la columna {col}, se omite del score.")
                continue

            score_col = f"score__{col}"
            df[score_col] = percentile_score(df[col], higher_is_better=higher_is_better)
            available_feature_score_cols.append(score_col)

        # Score de dimensión
        dim_score_col = f"dimension_score__{dim_name}"
        if available_feature_score_cols:
            df[dim_score_col] = df[available_feature_score_cols].mean(axis=1, skipna=True)
        else:
            df[dim_score_col] = np.nan

        dimension_score_cols.append(dim_score_col)
        dimension_weights.append(dim_cfg["weight"])

    # Score total explicativo
    df["explanatory_score_0_1"] = weighted_row_score(df, dimension_score_cols, dimension_weights)
    df["explanatory_score_0_100"] = (df["explanatory_score_0_1"] * 100).round(2)

    # Ranking
    df["explanatory_rank"] = (
        df["explanatory_score_0_100"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # Percentil del score final
    final_valid = df["explanatory_score_0_100"].dropna()
    if len(final_valid) > 1:
        final_rank = final_valid.rank(method="average")
        final_pct = (final_rank - 1) / (len(final_valid) - 1)
        df["explanatory_score_percentile"] = np.nan
        df.loc[final_valid.index, "explanatory_score_percentile"] = (final_pct * 100).round(2)
    elif len(final_valid) == 1:
        df["explanatory_score_percentile"] = np.nan
        df.loc[final_valid.index, "explanatory_score_percentile"] = 100.0
    else:
        df["explanatory_score_percentile"] = np.nan

    df["explanatory_score_band"] = categorize_score(df["explanatory_score_0_100"])

    # Correlación con target operativo, si existe
    target_col = "bikes_ratio_capacity_mean"
    if target_col in df.columns:
        pearson = df["explanatory_score_0_100"].corr(df[target_col], method="pearson")
        spearman = df["explanatory_score_0_100"].corr(df[target_col], method="spearman")
        print("\nCorrelación con target operativo:")
        print(f"Pearson  (score vs {target_col}): {pearson:.4f}")
        print(f"Spearman (score vs {target_col}): {spearman:.4f}")

    # Salida
    keep_cols = [c for c in META_COLS if c in df.columns]
    keep_cols += [c for c in OPTIONAL_KEEP_COLS if c in df.columns]
    keep_cols += dimension_score_cols
    keep_cols += [
        "explanatory_score_0_1",
        "explanatory_score_0_100",
        "explanatory_score_percentile",
        "explanatory_score_band",
        "explanatory_rank",
    ]

    # Añadir features crudas utilizadas si existen
    raw_used_features = []
    for dim_cfg in DIMENSIONS.values():
        for col, _ in dim_cfg["features"]:
            if col in df.columns:
                raw_used_features.append(col)

    # Añadir feature scores
    feature_score_cols = [c for c in df.columns if c.startswith("score__")]

    keep_cols += raw_used_features
    keep_cols += feature_score_cols

    # Quitar duplicados conservando orden
    seen = set()
    ordered_keep_cols = []
    for c in keep_cols:
        if c not in seen:
            ordered_keep_cols.append(c)
            seen.add(c)

    out = df[ordered_keep_cols].copy()
    out = out.sort_values(["explanatory_rank", "station_id"], na_position="last")

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    feature_report = pd.DataFrame(feature_report_rows)
    feature_report.to_csv(OUT_FEATURES_CSV, index=False, encoding="utf-8-sig")

    print("\nArchivos guardados:")
    print(f" - {OUT_CSV}")
    print(f" - {OUT_FEATURES_CSV}")

    print("\nTop 15 estaciones por score explicativo:")
    preview_cols = [c for c in ["explanatory_rank", "station_id", "name", "neighborhood", "explanatory_score_0_100"] if c in out.columns]
    print(out[preview_cols].head(15).to_string(index=False))

    print("\nCobertura dimensiones:")
    coverage = (1 - out[dimension_score_cols].isna().mean()).sort_values()
    print(coverage.to_string())


if __name__ == "__main__":
    main()