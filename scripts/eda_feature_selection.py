from pathlib import Path
import numpy as np
import pandas as pd

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


ROOT = Path(".")
INPUT_CSV = ROOT / "data/processed/stations/bicing_station_modeling_2026_03.csv"

OUTDIR = ROOT / "data/processed/eda"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_TARGET_CORR = OUTDIR / "feature_target_correlations.csv"
OUT_FEATURE_CORR = OUTDIR / "feature_feature_spearman_matrix.csv"
OUT_HIGH_CORR_PAIRS = OUTDIR / "high_correlation_feature_pairs.csv"
OUT_VIF = OUTDIR / "feature_vif.csv"


TARGET = "bikes_ratio_capacity_mean"
ALT_TARGETS = ["pct_empty", "pct_near_empty", "pct_full", "pct_near_full"]

HIGH_CORR_THRESHOLD = 0.85
TOP_N_PRINT = 20


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    target_cols = {TARGET, *ALT_TARGETS}
    id_cols = {
        "station_id",
        "name",
        "address",
        "post_code",
        "census_section",
        "neighborhood",
        "geometry",
        "short_name",
        "cross_street",
        "last_updated",
    }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in numeric_cols if c not in target_cols and c not in id_cols]
    return features


def compute_target_correlations(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")

        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < 10:
            continue

        pearson = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson")
        spearman = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")

        rows.append(
            {
                "feature": col,
                "n_valid": len(valid),
                "pearson": pearson,
                "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                "spearman": spearman,
                "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    return out


def compute_high_corr_pairs(df: pd.DataFrame, features: list[str], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[features].corr(method="spearman")
    pairs = []

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append(
                    {
                        "feature_1": c1,
                        "feature_2": c2,
                        "spearman_corr": val,
                        "abs_spearman_corr": abs(val),
                    }
                )

    pairs_df = pd.DataFrame(pairs).sort_values("abs_spearman_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_1", "feature_2", "spearman_corr", "abs_spearman_corr"]
    )
    return corr, pairs_df


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame(columns=["feature", "vif"])

    X = df[features].copy()

    # imputación simple para VIF
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # quita columnas constantes
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep].copy()

    if X.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])

    X_np = X.astype(float).values
    vif_rows = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = variance_inflation_factor(X_np, i)
        except Exception:
            vif_val = np.nan
        vif_rows.append({"feature": col, "vif": vif_val})

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


def main():
    print("Leyendo dataset de modelado...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df = clean_cols(df)

    if TARGET not in df.columns:
        raise KeyError(f"No encuentro el target principal: {TARGET}")

    features = get_numeric_features(df)

    print(f"Filas: {len(df)}")
    print(f"Número de features numéricas candidatas: {len(features)}")

    print("\nCalculando correlación feature-target...")
    target_corr = compute_target_correlations(df, features, TARGET)
    target_corr.to_csv(OUT_TARGET_CORR, index=False, encoding="utf-8-sig")

    print(f"\nTop {TOP_N_PRINT} variables por correlación Spearman con {TARGET}:")
    print(target_corr.head(TOP_N_PRINT).to_string(index=False))

    print("\nCalculando correlación entre features...")
    feature_corr_matrix, high_corr_pairs = compute_high_corr_pairs(df, features, HIGH_CORR_THRESHOLD)
    feature_corr_matrix.to_csv(OUT_FEATURE_CORR, index=True, encoding="utf-8-sig")
    high_corr_pairs.to_csv(OUT_HIGH_CORR_PAIRS, index=False, encoding="utf-8-sig")

    print(f"\nPares de variables con |Spearman| >= {HIGH_CORR_THRESHOLD}: {len(high_corr_pairs)}")
    if len(high_corr_pairs) > 0:
        print(high_corr_pairs.head(TOP_N_PRINT).to_string(index=False))

    print("\nCalculando VIF...")
    vif_df = compute_vif(df, features)
    vif_df.to_csv(OUT_VIF, index=False, encoding="utf-8-sig")

    if len(vif_df) > 0:
        print(f"\nTop {TOP_N_PRINT} variables por VIF:")
        print(vif_df.head(TOP_N_PRINT).to_string(index=False))
    else:
        print("No se ha podido calcular VIF o no hay suficientes variables.")

    print("\nArchivos generados:")
    print(f" - {OUT_TARGET_CORR}")
    print(f" - {OUT_FEATURE_CORR}")
    print(f" - {OUT_HIGH_CORR_PAIRS}")
    print(f" - {OUT_VIF}")

    print("\nInterpretación rápida:")
    print(" - Para el índice explicativo: quédate con variables urbanísticamente interpretables y evita pares muy redundantes.")
    print(" - Para el modelo predictivo: prioriza variables con buena relación con el target, pero sin meter demasiadas muy correlacionadas entre sí.")


if __name__ == "__main__":
    main()