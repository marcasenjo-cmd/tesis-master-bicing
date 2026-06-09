from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
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

OUT_METRICS = OUTDIR / "predictive_model_comparison_2026_03.csv"
OUT_OOF = OUTDIR / "predictive_oof_predictions_2026_03.csv"
OUT_FEATURES = OUTDIR / "predictive_feature_importance_2026_03.csv"


META_COLS = [
    "station_id",
    "name",
    "address",
    "cross_street",
    "post_code",
    "census_section",
    "neighborhood",
    "geometry",
]

# Variables operativas derivadas del estado de estaciones.
# No deben entrar si el target es ocupación media / ratio de ocupación.
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
    "unemployment",  # está vacío en tu caso
]

# Si no quieres usar topografía, capacidad o cualquier otra,
# añádela aquí.
MANUAL_DROP_COLS: List[str] = []


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise KeyError(f"No existe el target '{target_col}' en el dataset")

    # Solo filas con target válido
    df = df[df[target_col].notna()].copy()

    drop_cols = set(META_COLS + LEAKAGE_COLS + MANUAL_DROP_COLS)

    # Mantener solo numéricas y no target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c not in drop_cols]

    # Quitar columnas completamente vacías
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    if not feature_cols:
        raise ValueError("No quedan features numéricas válidas tras filtrar metadatos y leakage")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y, feature_cols


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
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=20000)),
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

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    return results


def create_oof_predictions(
    full_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Pipeline],
) -> pd.DataFrame:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    out = full_df.loc[y.index, [c for c in ["station_id", "name", "neighborhood", TARGET_COL] if c in full_df.columns]].copy()

    for model_name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        out[f"pred__{model_name}"] = preds
        out[f"abs_error__{model_name}"] = (y.values - preds).astype(float)
        out[f"sq_error__{model_name}"] = (y.values - preds) ** 2

    return out


def extract_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    best_model_name: str,
    best_model: Pipeline,
) -> pd.DataFrame:
    best_model.fit(X, y)

    model = best_model.named_steps["model"]

    # ColumnTransformer + un único bloque numérico -> conserva orden de X.columns
    feature_names = X.columns.tolist()

    if hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
        out = pd.DataFrame(
            {
                "model": best_model_name,
                "feature": feature_names,
                "importance": importance,
                "signed_value": np.ravel(model.coef_),
                "importance_type": "abs_coefficient",
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        importance = np.ravel(model.feature_importances_)
        out = pd.DataFrame(
            {
                "model": best_model_name,
                "feature": feature_names,
                "importance": importance,
                "signed_value": np.nan,
                "importance_type": "feature_importance",
            }
        )
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame(
        {
            "model": [best_model_name],
            "feature": ["N/A"],
            "importance": [np.nan],
            "signed_value": [np.nan],
            "importance_type": ["not_available"],
        }
    )


def main() -> None:
    print("Leyendo dataset de modelado...")
    df = read_csv(INPUT_CSV)

    X, y, feature_cols = build_feature_matrix(df, TARGET_COL)

    print(f"Filas válidas para modelado: {len(X)}")
    print(f"Número de features: {len(feature_cols)}")
    print("\nPrimeras features:")
    print(feature_cols[:25])

    models = make_models()
    if not HAS_XGBOOST:
        print("\n[INFO] xgboost no está instalado. Se omite ese modelo.")

    print("\nEvaluando modelos con cross-validation...")
    metrics = evaluate_models(X, y, models)
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    print("\nComparativa de modelos:")
    print(metrics.to_string(index=False))

    print("\nGenerando predicciones out-of-fold...")
    oof = create_oof_predictions(df, X, y, models)
    oof.to_csv(OUT_OOF, index=False, encoding="utf-8-sig")

    best_model_name = metrics.iloc[0]["model"]
    best_model = models[best_model_name]

    print(f"\nMejor modelo por CV RMSE: {best_model_name}")

    feature_importance = extract_feature_importance(X, y, best_model_name, best_model)
    feature_importance.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

    print("\nTop 20 features del mejor modelo:")
    print(feature_importance.head(20).to_string(index=False))

    print("\nArchivos guardados:")
    print(f" - {OUT_METRICS}")
    print(f" - {OUT_OOF}")
    print(f" - {OUT_FEATURES}")


if __name__ == "__main__":
    main()