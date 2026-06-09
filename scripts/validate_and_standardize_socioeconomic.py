from pathlib import Path
import re
import unicodedata
import itertools
import numpy as np
import pandas as pd

ROOT = Path(".")

INPUT_DIR = ROOT / "data/processed/input_variables"
OUT_DIR = ROOT / "data/processed/socioeconomic_validation"
STD_DIR = OUT_DIR / "standardized_csvs"
DUP_DIR = OUT_DIR / "duplicate_key_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
STD_DIR.mkdir(parents=True, exist_ok=True)
DUP_DIR.mkdir(parents=True, exist_ok=True)

SECTION_FILES = [
    "socioeconomic_census_section_income.csv",
    "socioeconomic_census_section_population_sex_age.csv",
    "socioeconomic_census_section_education.csv",
    "socioeconomic_census_section_household_size.csv",
    "socioeconomic_census_section_non_spanish_population.csv",
    "socioeconomic_census_section_car_ownership.csv",
]

NEIGHBORHOOD_FILES = [
    "socioeconomic_neighborhood_unemployment.csv",
]


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    return df


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
    Devuelve la sección censal como string de 10 dígitos.
    Ejemplos:
      801902008.0   -> 0801902008
      0801902008    -> 0801902008
      '0801902008 ...' -> 0801902008
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Si viene como float-string tipo 801902008.0
    try:
        f = float(s.replace(",", "."))
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            s = str(int(round(f)))
    except Exception:
        pass

    # Quédate solo con dígitos
    s = re.sub(r"\D", "", s)

    if not s:
        return None

    # Formato estándar CUSEC: 10 dígitos
    return s.zfill(10)


def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}")
    return None


def summarize_lengths(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return {
            "min_len": np.nan,
            "max_len": np.nan,
            "n_lengths": 0,
            "length_values": "",
        }
    lengths = s.str.len()
    uniq = sorted(lengths.unique().tolist())
    return {
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "n_lengths": len(uniq),
        "length_values": ",".join(map(str, uniq[:10])),
    }


def duplicate_report(df: pd.DataFrame, key_col: str, file_name: str):
    if key_col not in df.columns:
        return

    dups = df[df[key_col].duplicated(keep=False)].copy()
    if dups.empty:
        return

    out = DUP_DIR / f"{Path(file_name).stem}__duplicates_on_{key_col}.csv"
    dups.to_csv(out, index=False, encoding="utf-8-sig")


def process_section_file(file_name: str):
    path = INPUT_DIR / file_name
    if not path.exists():
        return None, set()

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = clean_cols(df)

    if "census_section" not in df.columns:
        return {
            "file": file_name,
            "type": "section",
            "exists": True,
            "has_census_section": False,
        }, set()

    df["census_section_raw_str"] = df["census_section"].astype(str)
    df["census_section_std"] = df["census_section"].map(normalize_census_section)

    # Guardar versión estandarizada
    out_std = STD_DIR / file_name
    df.to_csv(out_std, index=False, encoding="utf-8-sig")

    # Reporte de duplicados en clave estandarizada
    duplicate_report(df, "census_section_std", file_name)

    raw_lengths = summarize_lengths(df["census_section"])
    std_lengths = summarize_lengths(df["census_section_std"])

    summary = {
        "file": file_name,
        "type": "section",
        "exists": True,
        "has_census_section": True,
        "rows": len(df),
        "raw_non_null": int(df["census_section"].notna().sum()),
        "std_non_null": int(df["census_section_std"].notna().sum()),
        "raw_unique": int(df["census_section"].astype(str).nunique(dropna=True)),
        "std_unique": int(df["census_section_std"].nunique(dropna=True)),
        "std_duplicates_rows": int(df["census_section_std"].duplicated(keep=False).sum()),
        "std_duplicate_keys": int(df["census_section_std"].duplicated(keep=False).groupby(df["census_section_std"]).any().sum())
        if df["census_section_std"].notna().any() else 0,
        "raw_min_len": raw_lengths["min_len"],
        "raw_max_len": raw_lengths["max_len"],
        "raw_length_values": raw_lengths["length_values"],
        "std_min_len": std_lengths["min_len"],
        "std_max_len": std_lengths["max_len"],
        "std_length_values": std_lengths["length_values"],
        "sample_raw_1": df["census_section"].dropna().astype(str).head(1).tolist()[0] if df["census_section"].notna().any() else None,
        "sample_std_1": df["census_section_std"].dropna().astype(str).head(1).tolist()[0] if df["census_section_std"].notna().any() else None,
    }

    key_set = set(df["census_section_std"].dropna().astype(str).unique().tolist())
    return summary, key_set


def process_neighborhood_file(file_name: str):
    path = INPUT_DIR / file_name
    if not path.exists():
        return None, set()

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = clean_cols(df)

    nb_col = pick_col(df, ["neighborhood", "nom_barri"], required=False)
    if not nb_col:
        return {
            "file": file_name,
            "type": "neighborhood",
            "exists": True,
            "has_neighborhood": False,
        }, set()

    df["neighborhood_std"] = df[nb_col].map(normalize_text)

    out_std = STD_DIR / file_name
    df.to_csv(out_std, index=False, encoding="utf-8-sig")

    duplicate_report(df, "neighborhood_std", file_name)

    summary = {
        "file": file_name,
        "type": "neighborhood",
        "exists": True,
        "has_neighborhood": True,
        "rows": len(df),
        "raw_non_null": int(df[nb_col].notna().sum()),
        "std_non_null": int(df["neighborhood_std"].notna().sum()),
        "raw_unique": int(df[nb_col].astype(str).nunique(dropna=True)),
        "std_unique": int(df["neighborhood_std"].nunique(dropna=True)),
        "std_duplicates_rows": int(df["neighborhood_std"].duplicated(keep=False).sum()),
        "sample_raw_1": df[nb_col].dropna().astype(str).head(1).tolist()[0] if df[nb_col].notna().any() else None,
        "sample_std_1": df["neighborhood_std"].dropna().astype(str).head(1).tolist()[0] if df["neighborhood_std"].notna().any() else None,
    }

    key_set = set(df["neighborhood_std"].dropna().astype(str).unique().tolist())
    return summary, key_set


def build_pairwise_overlap(file_keys: dict, key_name: str, out_name: str):
    rows = []
    files = list(file_keys.keys())

    for a, b in itertools.combinations(files, 2):
        set_a = file_keys[a]
        set_b = file_keys[b]
        inter = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a

        rows.append({
            "file_a": a,
            "file_b": b,
            f"{key_name}_a": len(set_a),
            f"{key_name}_b": len(set_b),
            "intersection": len(inter),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "intersection_pct_vs_a": len(inter) / len(set_a) if set_a else np.nan,
            "intersection_pct_vs_b": len(inter) / len(set_b) if set_b else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / out_name, index=False, encoding="utf-8-sig")
    return out


def main():
    section_summaries = []
    section_keys = {}

    print("Analizando CSV por census_section...")
    for file_name in SECTION_FILES:
        summary, keys = process_section_file(file_name)
        if summary is not None:
            section_summaries.append(summary)
            section_keys[file_name] = keys

    nb_summaries = []
    nb_keys = {}

    print("Analizando CSV por neighborhood...")
    for file_name in NEIGHBORHOOD_FILES:
        summary, keys = process_neighborhood_file(file_name)
        if summary is not None:
            nb_summaries.append(summary)
            nb_keys[file_name] = keys

    section_summary_df = pd.DataFrame(section_summaries)
    nb_summary_df = pd.DataFrame(nb_summaries)

    section_summary_df.to_csv(OUT_DIR / "section_file_summary.csv", index=False, encoding="utf-8-sig")
    nb_summary_df.to_csv(OUT_DIR / "neighborhood_file_summary.csv", index=False, encoding="utf-8-sig")

    if section_keys:
        pair_sections = build_pairwise_overlap(
            section_keys,
            key_name="n_keys",
            out_name="section_pairwise_overlap.csv"
        )
    else:
        pair_sections = pd.DataFrame()

    if nb_keys:
        pair_nb = build_pairwise_overlap(
            nb_keys,
            key_name="n_keys",
            out_name="neighborhood_pairwise_overlap.csv"
        )
    else:
        pair_nb = pd.DataFrame()

    print("\nResumen section files:")
    if not section_summary_df.empty:
        print(section_summary_df.to_string(index=False))
    else:
        print("No se encontraron section files.")

    print("\nResumen neighborhood files:")
    if not nb_summary_df.empty:
        print(nb_summary_df.to_string(index=False))
    else:
        print("No se encontraron neighborhood files.")

    print("\nArchivos generados en:")
    print(OUT_DIR)
    print("\nTe interesa mirar especialmente:")
    print(" - section_file_summary.csv")
    print(" - section_pairwise_overlap.csv")
    print(" - standardized_csvs/")
    print(" - duplicate_key_reports/")


if __name__ == "__main__":
    main()