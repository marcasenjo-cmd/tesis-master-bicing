from pathlib import Path
from typing import Dict, List, Tuple

import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


ROOT = Path(".")
INPUT_CSV = ROOT / "data/processed/stations/bicing_station_modeling_2026_03.csv"
OUTDIR = ROOT / "data/processed/modeling"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "bikes_ratio_capacity_mean"
RANDOM_STATE = 42
N_SPLITS = 5
EXCLUDE_TESTING_STATIONS = True

OUT_COMPARISON = OUTDIR / "demographic_feature_set_comparison_2026_03.csv"
OUT_FEATURES_BY_SCENARIO = OUTDIR / "demographic_feature_set_features_used_2026_03.csv"
OUT_BEST_FEATURE_IMPORTANCE = OUTDIR / "demographic_best_model_feature_importance_2026_03.csv"


META_COLS = [
    "station_id",
    "name",
    "address",
    "cross_street",
    "census_section",
    "neighborhood",
    "geometry",
]

MANUAL_DROP_COLS = [
    "short_name",
    "nearby_distance",
    "last_updated",
    "post_code",
    "unemployment",
]

LEAKAGE_COLS = [
    "bikes_ratio_capacity_mean",
    "bikes_ratio_capacity_std",
    "docks_ratio_capacity_mean",
    "docks_ratio_capacity_std",
    "bikes_available_mean",
    "bikes_available_std",
    "bikes_available_min",
    "bikes_available_max",
    "docks_available_mean",
    "docks_available_std",
    "docks_available_min",
    "docks_available_max",
    "mechanical_bikes_mean",
    "ebikes_mean",
    "pct_empty",
    "pct_near_empty",
    "pct_full",
    "pct_near_full",
    "n_observations",
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


def filter_testing_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDE_TESTING_STATIONS or "name" not in df.columns:
        return df
    mask = ~df["name"].astype(str).str.contains("TESTING", case=False, na=False)
    return df.loc[mask].copy()


def detect_age_columns(df: pd.DataFrame) -> List[str]:
    cols = df.columns.tolist()
    age_patterns = [
        r"^\d{1,2}-\d{1,2}$",   # 10-19
        r"^\d{1,2}\+$",         # 70+
        r"^age[_\-].*",
        r".*age[_\-].*",
    ]
    out = []
    for c in cols:
        for pat in age_patterns:
            if re.match(pat, c, flags=re.IGNORECASE):
                out.append(c)
                break
    return sorted(set(out))


def detect_sex_columns(df: pd.DataFrame) -> List[str]:
    sex_candidates = {
        "f", "m",
        "female", "male",
        "women", "men",
        "population_female", "population_male",
        "sex_f", "sex_m",
    }
    out = [c for c in df.columns if c.lower() in sex_candidates]
    return sorted(set(out))


def make_models() -> Dict[str, Pipeline]:
    numeric_transformer_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_transformer_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            steps=[
                ("prep", ColumnTransformer(
                    transformers=[("num", numeric_transformer_scaled, slice(0, None))],
                    remainder="drop",
                )),
                ("model", LinearRegression()),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("prep", ColumnTransformer(
                    transformers=[("num", numeric_transformer_scaled, slice(0, None))],
                    remainder="drop",
                )),
                ("model", ElasticNet(
                    alpha=0.05,
                    l1_ratio=0.5,
                    random_state=RANDOM_STATE,
                    max_iter=20000,
                )),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", ColumnTransformer(
                    transformers=[("num", numeric_transformer_tree, slice(0, None))],
                    remainder="drop",
                )),
                ("model", RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("prep", ColumnTransformer(
                    transformers=[("num", numeric_transformer_tree, slice(0, None))],
                    remainder="drop",
                )),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = Pipeline(
            steps=[
                ("prep", ColumnTransformer(
                    transformers=[("num", numeric_transformer_tree, slice(0, None))],
                    remainder="drop",
                )),
                ("model", XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.04,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        )

    return models


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    scenario_drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise KeyError(f"No existe el target '{target_col}'")

    df = df[df[target_col].notna()].copy()

    drop_cols = set(META_COLS + LEAKAGE_COLS + MANUAL_DROP_COLS + scenario_drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c not in drop_cols]
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    if not feature_cols:
        raise ValueError("No quedan features válidas")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


def evaluate_models(X: pd.DataFrame, y: pd.Series, models: Dict[str, Pipeline]) -> pd.DataFrame:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=False,
        )

        rows.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(-scores["test_rmse"].mean()),
                "cv_rmse_std": float(scores["test_rmse"].std()),
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_mae_std": float(scores["test_mae"].std()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "cv_r2_std": float(scores["test_r2"].std()),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)


def extract_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model: Pipeline,
    scenario_name: str,
) -> pd.DataFrame:
    model.fit(X, y)
    est = model.named_steps["model"]
    feature_names = X.columns.tolist()

    if hasattr(est, "coef_"):
        vals = np.ravel(est.coef_)
        out = pd.DataFrame(
            {
                "scenario": scenario_name,
                "model": model_name,
                "feature": feature_names,
                "importance": np.abs(vals),
                "signed_value": vals,
                "importance_type": "abs_coefficient",
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(est, "feature_importances_"):
        vals = np.ravel(est.feature_importances_)
        out = pd.DataFrame(
            {
                "scenario": scenario_name,
                "model": model_name,
                "feature": feature_names,
                "importance": vals,
                "signed_value": np.nan,
                "importance_type": "feature_importance",
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame(
        {
            "scenario": [scenario_name],
            "model": [model_name],
            "feature": ["N/A"],
            "importance": [np.nan],
            "signed_value": [np.nan],
            "importance_type": ["not_available"],
        }
    )


def main():
    df = read_csv(INPUT_CSV)
    df = filter_testing_rows(df)

    age_cols = detect_age_columns(df)
    sex_cols = detect_sex_columns(df)

    print(f"Filas tras excluir TESTING si aplica: {len(df)}")
    print(f"Columnas edad detectadas: {age_cols}")
    print(f"Columnas sexo detectadas: {sex_cols}")

    scenarios = {
        "all_demographics": [],
        "no_age_keep_sex": age_cols,
        "no_age_no_sex": age_cols + sex_cols,
    }

    models = make_models()
    if not HAS_XGBOOST:
        print("\n[INFO] xgboost no está instalado. Se omite ese modelo.")
        print(r"Instalación: python -m pip install xgboost")

    results_all = []
    features_rows = []
    best_overall = None
    best_overall_importance = None

    for scenario_name, scenario_drop_cols in scenarios.items():
        print(f"\n--- Escenario: {scenario_name} ---")
        X, y, feature_cols = build_feature_matrix(df, TARGET_COL, scenario_drop_cols)

        print(f"Features usadas ({len(feature_cols)}):")
        print(feature_cols)

        for f in feature_cols:
            features_rows.append(
                {
                    "scenario": scenario_name,
                    "feature": f,
                    "used": True,
                    "is_age": f in age_cols,
                    "is_sex": f in sex_cols,
                }
            )

        metrics = evaluate_models(X, y, models)
        metrics.insert(0, "scenario", scenario_name)
        metrics.insert(1, "n_features", len(feature_cols))
        metrics.insert(2, "n_rows", len(X))
        results_all.append(metrics)

        print(metrics.to_string(index=False))

        best_row = metrics.iloc[0].copy()
        best_model_name = best_row["model"]
        best_model = models[best_model_name]

        fi = extract_feature_importance(
            X=X,
            y=y,
            model_name=best_model_name,
            model=best_model,
            scenario_name=scenario_name,
        )

        if best_overall is None or best_row["cv_rmse_mean"] < best_overall["cv_rmse_mean"]:
            best_overall = best_row
            best_overall_importance = fi

    comparison = pd.concat(results_all, ignore_index=True)
    comparison.to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")

    features_used = pd.DataFrame(features_rows).sort_values(["scenario", "feature"])
    features_used.to_csv(OUT_FEATURES_BY_SCENARIO, index=False, encoding="utf-8-sig")

    if best_overall_importance is not None:
        best_overall_importance.to_csv(OUT_BEST_FEATURE_IMPORTANCE, index=False, encoding="utf-8-sig")

    print("\n=== Resumen final por escenario ===")
    summary = (
        comparison.sort_values(["scenario", "cv_rmse_mean"])
        .groupby("scenario", as_index=False)
        .first()
        .sort_values("cv_rmse_mean")
        .reset_index(drop=True)
    )
    print(summary.to_string(index=False))

    if best_overall is not None:
        print("\n=== Mejor combinación global ===")
        print(best_overall.to_string())

    print("\nArchivos guardados:")
    print(f" - {OUT_COMPARISON}")
    print(f" - {OUT_FEATURES_BY_SCENARIO}")
    print(f" - {OUT_BEST_FEATURE_IMPORTANCE}")


if __name__ == "__main__":
    main()