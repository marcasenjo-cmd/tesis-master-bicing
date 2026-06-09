from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")
SCORE_CSV = ROOT / "data/processed/stations/bicing_station_explanatory_score_2026_03.csv"
MODEL_METRICS_CSV = ROOT / "data/processed/modeling/predictive_model_comparison_2026_03.csv"
OOF_CSV = ROOT / "data/processed/modeling/predictive_oof_predictions_2026_03.csv"

OUTDIR = ROOT / "data/processed/final_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MERGED = OUTDIR / "station_score_vs_prediction_comparison_2026_03.csv"
OUT_SCORE_GAP = OUTDIR / "station_structural_gap_2026_03.csv"
OUT_UNDERVALUED = OUTDIR / "top_undervalued_by_score_2026_03.csv"
OUT_OVERVALUED = OUTDIR / "top_overvalued_by_score_2026_03.csv"
OUT_UNDERPREDICTED = OUTDIR / "top_underpredicted_by_model_2026_03.csv"
OUT_OVERPREDICTED = OUTDIR / "top_overpredicted_by_model_2026_03.csv"

PLOT_SCORE_VS_TARGET = OUTDIR / "plot_explanatory_score_vs_target_2026_03.png"
PLOT_PRED_VS_TARGET = OUTDIR / "plot_prediction_vs_target_2026_03.png"
PLOT_SCORE_GAP = OUTDIR / "plot_structural_gap_hist_2026_03.png"


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def percentile_rank_0_100(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = pd.to_numeric(series, errors="coerce").dropna()

    if valid.empty:
        return out
    if len(valid) == 1:
        out.loc[valid.index] = 100.0
        return out

    ranks = valid.rank(method="average")
    pct = (ranks - 1) / (len(valid) - 1)
    out.loc[valid.index] = pct * 100
    return out.round(2)


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path):
    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(tmp[x], tmp[y], alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_hist(df: pd.DataFrame, col: str, title: str, outpath: Path):
    tmp = pd.to_numeric(df[col], errors="coerce").dropna()
    if tmp.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(tmp, bins=30)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    print("Leyendo score explicativo...")
    score_df = read_csv(SCORE_CSV)

    print("Leyendo métricas predictivas...")
    metrics_df = read_csv(MODEL_METRICS_CSV)
    if "cv_rmse_mean" not in metrics_df.columns or "model" not in metrics_df.columns:
        raise KeyError("No encuentro columnas model / cv_rmse_mean en predictive_model_comparison")

    best_model = metrics_df.sort_values("cv_rmse_mean").iloc[0]["model"]
    pred_col = f"pred__{best_model}"
    abs_err_col = f"abs_error__{best_model}"
    residual_col = f"residual__{best_model}"

    print(f"Mejor modelo detectado: {best_model}")

    print("Leyendo predicciones OOF...")
    oof_df = read_csv(OOF_CSV)
    required_oof = {"station_id", "bikes_ratio_capacity_mean", pred_col}
    missing = required_oof - set(oof_df.columns)
    if missing:
        raise KeyError(f"Faltan columnas en OOF: {missing}")

    # Seleccionar columnas útiles
    score_keep = [
        c for c in [
            "station_id",
            "name",
            "neighborhood",
            "capacity",
            "explanatory_score_0_100",
            "explanatory_score_percentile",
            "explanatory_score_band",
            "explanatory_rank",
            "dimension_score__accessibility",
            "dimension_score__activity",
            "dimension_score__socioeconomic",
            "dimension_score__topography",
        ]
        if c in score_df.columns
    ]

    oof_keep = [
        c for c in [
            "station_id",
            "bikes_ratio_capacity_mean",
            pred_col,
            abs_err_col,
            residual_col,
        ]
        if c in oof_df.columns
    ]

    merged = score_df[score_keep].merge(
        oof_df[oof_keep],
        on="station_id",
        how="inner",
        suffixes=("", "_oof"),
    )

    # Completar name/neighborhood si vinieran del OOF y no del score
    for c in ["name", "neighborhood"]:
        if c not in merged.columns and c in oof_df.columns:
            merged = merged.merge(oof_df[["station_id", c]], on="station_id", how="left")

    # Percentiles comparables
    merged["target_percentile_0_100"] = percentile_rank_0_100(merged["bikes_ratio_capacity_mean"])
    merged["prediction_percentile_0_100"] = percentile_rank_0_100(merged[pred_col])

    # Gap estructural:
    # positivo => score estructural mayor que el desempeño real
    # negativo => desempeño real mayor que lo que sugería el score
    merged["structural_gap_score"] = (
        merged["explanatory_score_percentile"] - merged["target_percentile_0_100"]
    ).round(2)

    # Gap predictivo:
    # positivo => el modelo infrapredijo
    # negativo => el modelo sobrepredijo
    merged["predictive_gap"] = (
        merged["bikes_ratio_capacity_mean"] - merged[pred_col]
    ).round(6)

    # Etiquetas interpretables
    merged["score_interpretation"] = np.where(
        merged["structural_gap_score"] >= 0,
        "Sobrevalorada por score",
        "Infravalorada por score",
    )
    merged["predictive_interpretation"] = np.where(
        merged["predictive_gap"] >= 0,
        "Infrapredicha por modelo",
        "Sobrepredicha por modelo",
    )

    # Correlaciones
    score_target_pearson = merged["explanatory_score_0_100"].corr(
        merged["bikes_ratio_capacity_mean"], method="pearson"
    )
    score_target_spearman = merged["explanatory_score_0_100"].corr(
        merged["bikes_ratio_capacity_mean"], method="spearman"
    )

    pred_target_pearson = merged[pred_col].corr(
        merged["bikes_ratio_capacity_mean"], method="pearson"
    )
    pred_target_spearman = merged[pred_col].corr(
        merged["bikes_ratio_capacity_mean"], method="spearman"
    )

    mae_best = np.abs(merged["bikes_ratio_capacity_mean"] - merged[pred_col]).mean()
    rmse_best = np.sqrt(((merged["bikes_ratio_capacity_mean"] - merged[pred_col]) ** 2).mean())

    print("\nResumen comparación final:")
    print(f"Filas comparables: {len(merged)}")
    print(f"Pearson  score vs target: {score_target_pearson:.4f}")
    print(f"Spearman score vs target: {score_target_spearman:.4f}")
    print(f"Pearson  pred  vs target: {pred_target_pearson:.4f}")
    print(f"Spearman pred  vs target: {pred_target_spearman:.4f}")
    print(f"MAE  mejor modelo ({best_model}): {mae_best:.4f}")
    print(f"RMSE mejor modelo ({best_model}): {rmse_best:.4f}")

    # Ordenaciones
    sort_cols = [c for c in ["station_id", "name", "neighborhood"] if c in merged.columns]

    undervalued = merged.sort_values("structural_gap_score").copy()
    overvalued = merged.sort_values("structural_gap_score", ascending=False).copy()

    underpredicted = merged.sort_values("predictive_gap", ascending=False).copy()
    overpredicted = merged.sort_values("predictive_gap").copy()

    # Guardados principales
    merged.sort_values("explanatory_rank", na_position="last").to_csv(
        OUT_MERGED, index=False, encoding="utf-8-sig"
    )

    structural_gap_cols = [
        c for c in [
            "station_id",
            "name",
            "neighborhood",
            "explanatory_score_0_100",
            "explanatory_score_percentile",
            "bikes_ratio_capacity_mean",
            "target_percentile_0_100",
            "structural_gap_score",
            "score_interpretation",
            pred_col,
            "prediction_percentile_0_100",
            "predictive_gap",
            "predictive_interpretation",
        ]
        if c in merged.columns
    ]
    merged[structural_gap_cols].sort_values(
        "structural_gap_score"
    ).to_csv(OUT_SCORE_GAP, index=False, encoding="utf-8-sig")

    top_cols = [
        c for c in [
            "station_id",
            "name",
            "neighborhood",
            "capacity",
            "explanatory_score_0_100",
            "explanatory_score_percentile",
            "bikes_ratio_capacity_mean",
            "target_percentile_0_100",
            pred_col,
            "prediction_percentile_0_100",
            "structural_gap_score",
            "predictive_gap",
        ]
        if c in merged.columns
    ]

    undervalued[top_cols].head(25).to_csv(OUT_UNDERVALUED, index=False, encoding="utf-8-sig")
    overvalued[top_cols].head(25).to_csv(OUT_OVERVALUED, index=False, encoding="utf-8-sig")
    underpredicted[top_cols].head(25).to_csv(OUT_UNDERPREDICTED, index=False, encoding="utf-8-sig")
    overpredicted[top_cols].head(25).to_csv(OUT_OVERPREDICTED, index=False, encoding="utf-8-sig")

    # Plots
    save_scatter(
        merged,
        "explanatory_score_0_100",
        "bikes_ratio_capacity_mean",
        "Explanatory score vs target",
        PLOT_SCORE_VS_TARGET,
    )
    save_scatter(
        merged,
        pred_col,
        "bikes_ratio_capacity_mean",
        f"Prediction ({best_model}) vs target",
        PLOT_PRED_VS_TARGET,
    )
    save_hist(
        merged,
        "structural_gap_score",
        "Structural gap score distribution",
        PLOT_SCORE_GAP,
    )

    print("\nTop 10 infravaloradas por score:")
    print(
        undervalued[
            [c for c in ["station_id", "name", "neighborhood", "structural_gap_score"] if c in undervalued.columns]
        ].head(10).to_string(index=False)
    )

    print("\nTop 10 sobrevaloradas por score:")
    print(
        overvalued[
            [c for c in ["station_id", "name", "neighborhood", "structural_gap_score"] if c in overvalued.columns]
        ].head(10).to_string(index=False)
    )

    print("\nTop 10 más infrapredichas por el modelo:")
    print(
        underpredicted[
            [c for c in ["station_id", "name", "neighborhood", "predictive_gap"] if c in underpredicted.columns]
        ].head(10).to_string(index=False)
    )

    print("\nTop 10 más sobrepredichas por el modelo:")
    print(
        overpredicted[
            [c for c in ["station_id", "name", "neighborhood", "predictive_gap"] if c in overpredicted.columns]
        ].head(10).to_string(index=False)
    )

    print("\nArchivos guardados:")
    print(f" - {OUT_MERGED}")
    print(f" - {OUT_SCORE_GAP}")
    print(f" - {OUT_UNDERVALUED}")
    print(f" - {OUT_OVERVALUED}")
    print(f" - {OUT_UNDERPREDICTED}")
    print(f" - {OUT_OVERPREDICTED}")
    print(f" - {PLOT_SCORE_VS_TARGET}")
    print(f" - {PLOT_PRED_VS_TARGET}")
    print(f" - {PLOT_SCORE_GAP}")


if __name__ == "__main__":
    main()