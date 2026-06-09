from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")

# Inputs
EXPLANATORY_CSV = ROOT / "data/processed/stations/bicing_station_explanatory_score_2026_03.csv"
MODEL_COMPARISON_CSV = ROOT / "data/processed/modeling/predictive_model_comparison_2026_03.csv"
MODEL_FEATURES_CSV = ROOT / "data/processed/modeling/predictive_feature_importance_2026_03.csv"
OOF_CSV = ROOT / "data/processed/modeling/predictive_oof_predictions_2026_03.csv"
FINAL_COMPARE_CSV = ROOT / "data/processed/final_analysis/station_score_vs_prediction_comparison_2026_03.csv"

# Output
OUTDIR = ROOT / "data/processed/visual_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_model_comparison(metrics: pd.DataFrame):
    if metrics.empty:
        return

    metrics = metrics.sort_values("cv_rmse_mean").copy()

    # RMSE
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["cv_rmse_mean"], yerr=metrics.get("cv_rmse_std"))
    plt.title("Comparativa de modelos - CV RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "01_model_comparison_rmse.png")

    # R2
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["model"], metrics["cv_r2_mean"], yerr=metrics.get("cv_r2_std"))
    plt.title("Comparativa de modelos - CV R²")
    plt.ylabel("R²")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "02_model_comparison_r2.png")


def plot_feature_importance(fi: pd.DataFrame, top_n: int = 20):
    if fi.empty or "importance" not in fi.columns:
        return

    fi = fi.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(top_n).copy()
    if fi.empty:
        return

    fi = fi.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(fi["feature"], fi["importance"])
    title = "Top variables del mejor modelo"
    if "model" in fi.columns and fi["model"].notna().any():
        title += f" ({fi['model'].iloc[0]})"
    plt.title(title)
    plt.xlabel("Importancia")
    plt.grid(True, axis="x", alpha=0.3)
    save_fig(OUTDIR / "03_top_feature_importance.png")


def plot_explanatory_score_distribution(score_df: pd.DataFrame):
    col = "explanatory_score_0_100"
    if col not in score_df.columns:
        return

    s = pd.to_numeric(score_df[col], errors="coerce").dropna()
    if s.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title("Distribución del score explicativo")
    plt.xlabel("Score explicativo (0-100)")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "04_explanatory_score_distribution.png")


def plot_score_vs_target(compare_df: pd.DataFrame):
    needed = ["explanatory_score_0_100", "bikes_ratio_capacity_mean"]
    if not all(c in compare_df.columns for c in needed):
        return

    df = safe_numeric(compare_df, needed).dropna(subset=needed).copy()
    if df.empty:
        return

    corr_p = df["explanatory_score_0_100"].corr(df["bikes_ratio_capacity_mean"], method="pearson")
    corr_s = df["explanatory_score_0_100"].corr(df["bikes_ratio_capacity_mean"], method="spearman")

    plt.figure(figsize=(7, 6))
    plt.scatter(df["explanatory_score_0_100"], df["bikes_ratio_capacity_mean"], alpha=0.7)
    plt.title(f"Score explicativo vs ocupación\nPearson={corr_p:.3f} | Spearman={corr_s:.3f}")
    plt.xlabel("Score explicativo (0-100)")
    plt.ylabel("bikes_ratio_capacity_mean")
    plt.grid(True, alpha=0.3)
    save_fig(OUTDIR / "05_score_vs_target.png")


def detect_best_prediction_col(compare_df: pd.DataFrame) -> str | None:
    pred_cols = [c for c in compare_df.columns if c.startswith("pred__")]
    return pred_cols[0] if pred_cols else None


def plot_prediction_vs_target(compare_df: pd.DataFrame):
    pred_col = detect_best_prediction_col(compare_df)
    target_col = "bikes_ratio_capacity_mean"
    if pred_col is None or target_col not in compare_df.columns:
        return

    df = safe_numeric(compare_df, [pred_col, target_col]).dropna(subset=[pred_col, target_col]).copy()
    if df.empty:
        return

    corr_p = df[pred_col].corr(df[target_col], method="pearson")
    corr_s = df[pred_col].corr(df[target_col], method="spearman")

    plt.figure(figsize=(7, 6))
    plt.scatter(df[pred_col], df[target_col], alpha=0.7)
    mn = min(df[pred_col].min(), df[target_col].min())
    mx = max(df[pred_col].max(), df[target_col].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title(f"Predicción vs ocupación real\nPearson={corr_p:.3f} | Spearman={corr_s:.3f}")
    plt.xlabel(pred_col)
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    save_fig(OUTDIR / "06_prediction_vs_target.png")


def plot_residual_distribution(compare_df: pd.DataFrame):
    residual_cols = [c for c in compare_df.columns if c.startswith("residual__")]
    if not residual_cols:
        return

    col = residual_cols[0]
    s = pd.to_numeric(compare_df[col], errors="coerce").dropna()
    if s.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title(f"Distribución de residuos ({col})")
    plt.xlabel("Residual")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "07_residual_distribution.png")


def plot_structural_gap_distribution(compare_df: pd.DataFrame):
    col = "structural_gap_score"
    if col not in compare_df.columns:
        return

    s = pd.to_numeric(compare_df[col], errors="coerce").dropna()
    if s.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title("Distribución del structural gap")
    plt.xlabel("Score percentile - target percentile")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "08_structural_gap_distribution.png")


def plot_top_bars(compare_df: pd.DataFrame, sort_col: str, filename: str, title: str, n: int = 15):
    if sort_col not in compare_df.columns:
        return

    cols = [c for c in ["name", "station_id", sort_col] if c in compare_df.columns]
    df = compare_df[cols].copy()
    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
    df = df.dropna(subset=[sort_col])

    if df.empty:
        return

    if "name" in df.columns:
        label_col = "name"
    else:
        label_col = "station_id"

    if "under" in filename or "infra" in filename:
        df = df.sort_values(sort_col, ascending=False).head(n)
    else:
        df = df.sort_values(sort_col).head(n)

    df = df.sort_values(sort_col, ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(df[label_col].astype(str), df[sort_col])
    plt.title(title)
    plt.xlabel(sort_col)
    plt.grid(True, axis="x", alpha=0.3)
    save_fig(OUTDIR / filename)


def plot_dimension_scores(score_df: pd.DataFrame):
    dim_cols = [c for c in score_df.columns if c.startswith("dimension_score__")]
    if not dim_cols:
        return

    means = score_df[dim_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values()

    plt.figure(figsize=(8, 5))
    plt.bar(means.index, means.values)
    plt.title("Media de score por dimensión")
    plt.ylabel("Media score (0-1)")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(OUTDIR / "09_dimension_score_means.png")


def plot_target_vs_top_features(compare_df: pd.DataFrame, fi: pd.DataFrame, n_features: int = 6):
    target_col = "bikes_ratio_capacity_mean"
    if target_col not in compare_df.columns or fi.empty or "feature" not in fi.columns:
        return

    top_feats = (
        fi.dropna(subset=["feature"])
        .sort_values("importance", ascending=False)["feature"]
        .astype(str)
        .tolist()[:n_features]
    )

    for i, feat in enumerate(top_feats, start=1):
        if feat not in compare_df.columns:
            continue

        df = safe_numeric(compare_df, [feat, target_col]).dropna(subset=[feat, target_col]).copy()
        if df.empty:
            continue

        corr = df[feat].corr(df[target_col], method="spearman")

        plt.figure(figsize=(7, 6))
        plt.scatter(df[feat], df[target_col], alpha=0.6)
        plt.title(f"{feat} vs ocupación\nSpearman={corr:.3f}")
        plt.xlabel(feat)
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        save_fig(OUTDIR / f"10_feature_vs_target_{i:02d}_{feat}.png")


def build_visual_summary_csv(compare_df: pd.DataFrame, fi: pd.DataFrame):
    rows = []

    if "explanatory_score_0_100" in compare_df.columns and "bikes_ratio_capacity_mean" in compare_df.columns:
        df = safe_numeric(compare_df, ["explanatory_score_0_100", "bikes_ratio_capacity_mean"])
        rows.append({
            "analysis": "score_vs_target_pearson",
            "value": df["explanatory_score_0_100"].corr(df["bikes_ratio_capacity_mean"], method="pearson")
        })
        rows.append({
            "analysis": "score_vs_target_spearman",
            "value": df["explanatory_score_0_100"].corr(df["bikes_ratio_capacity_mean"], method="spearman")
        })

    pred_col = detect_best_prediction_col(compare_df)
    if pred_col and "bikes_ratio_capacity_mean" in compare_df.columns:
        df = safe_numeric(compare_df, [pred_col, "bikes_ratio_capacity_mean"])
        rows.append({
            "analysis": "pred_vs_target_pearson",
            "value": df[pred_col].corr(df["bikes_ratio_capacity_mean"], method="pearson")
        })
        rows.append({
            "analysis": "pred_vs_target_spearman",
            "value": df[pred_col].corr(df["bikes_ratio_capacity_mean"], method="spearman")
        })

    if not fi.empty:
        top_feat = fi.sort_values("importance", ascending=False).iloc[0]
        rows.append({
            "analysis": "top_feature_name",
            "value": top_feat["feature"]
        })
        rows.append({
            "analysis": "top_feature_importance",
            "value": top_feat["importance"]
        })

    pd.DataFrame(rows).to_csv(
        OUTDIR / "00_visual_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )


def main():
    print("Leyendo ficheros...")
    score_df = read_csv(EXPLANATORY_CSV)
    metrics_df = read_csv(MODEL_COMPARISON_CSV)
    fi_df = read_csv(MODEL_FEATURES_CSV)
    oof_df = read_csv(OOF_CSV)
    compare_df = read_csv(FINAL_COMPARE_CSV)

    print("Generando gráficos...")
    plot_model_comparison(metrics_df)
    plot_feature_importance(fi_df)
    plot_explanatory_score_distribution(score_df)
    plot_score_vs_target(compare_df)
    plot_prediction_vs_target(compare_df)
    plot_residual_distribution(compare_df)
    plot_structural_gap_distribution(compare_df)
    plot_dimension_scores(score_df)

    plot_top_bars(
        compare_df,
        sort_col="structural_gap_score",
        filename="11_top_infravaloradas_score.png",
        title="Top estaciones infravaloradas por el score",
        n=15,
    )
    plot_top_bars(
        compare_df,
        sort_col="structural_gap_score",
        filename="12_top_sobrevaloradas_score.png",
        title="Top estaciones sobrevaloradas por el score",
        n=15,
    )
    plot_top_bars(
        compare_df,
        sort_col="predictive_gap",
        filename="13_top_infrapredichas_modelo.png",
        title="Top estaciones infrapredichas por el modelo",
        n=15,
    )
    plot_top_bars(
        compare_df,
        sort_col="predictive_gap",
        filename="14_top_sobrepredichas_modelo.png",
        title="Top estaciones sobrepredichas por el modelo",
        n=15,
    )

    plot_target_vs_top_features(compare_df, fi_df, n_features=6)
    build_visual_summary_csv(compare_df, fi_df)

    print(f"\nGráficos y resumen guardados en: {OUTDIR}")


if __name__ == "__main__":
    main()