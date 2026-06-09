from pathlib import Path
import csv
import io
import re
import unicodedata
import pandas as pd


ROOT = Path(".")
RAW_CSV = ROOT / "data/raw/bcn/bcn_atur.csv"
OUT_CSV = ROOT / "data/processed/input_variables/socioeconomic_neighborhood_unemployment.csv"


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


def read_bcn_atur_broken_csv(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n\r") for line in f if line.strip()]

    if not lines:
        raise ValueError("El fichero bcn_atur.csv está vacío")

    # Header normal
    header = next(csv.reader([lines[0]]))

    rows = []
    expected_len = len(header)

    for raw in lines[1:]:
        fixed = raw

        # Quitar comilla exterior de toda la fila
        if fixed.startswith('"') and fixed.endswith('"'):
            fixed = fixed[1:-1]

        # Convertir comillas dobles escapadas a comillas normales
        fixed = fixed.replace('""', '"')

        row = next(csv.reader([fixed]))

        if len(row) == expected_len:
            rows.append(row)
        else:
            # Si alguna fila sigue rara, intentamos ajustar
            if len(row) < expected_len:
                row = row + [None] * (expected_len - len(row))
                rows.append(row)
            else:
                row = row[:expected_len]
                rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    return clean_cols(df)


def main():
    df = read_bcn_atur_broken_csv(RAW_CSV)

    # Detectar columnas clave
    territory_col = None
    territory_type_col = None

    for c in df.columns:
        cl = c.lower()
        if cl in ["territori", "territorio"]:
            territory_col = c
        elif cl in ["tipus de territori", "tipo de territorio"]:
            territory_type_col = c

    if territory_col is None:
        raise KeyError("No encuentro la columna de territorio en bcn_atur.csv")
    if territory_type_col is None:
        raise KeyError("No encuentro la columna de tipo de territorio en bcn_atur.csv")

    # La última columna temporal es la más reciente
    date_cols = [c for c in df.columns if c not in [territory_col, territory_type_col]]
    if not date_cols:
        raise ValueError("No encuentro columnas temporales en bcn_atur.csv")

    date_col = date_cols[-1]

    print(f"Columna temporal usada: {date_col}")

    df = df.rename(columns={
        territory_col: "neighborhood",
        territory_type_col: "territory_type",
        date_col: "unemployment_percentage",
    }).copy()

    df["territory_type"] = df["territory_type"].astype(str).str.strip().str.lower()

    # Solo barrios
    df = df[df["territory_type"].str.contains("barri", na=False)].copy()

    df["neighborhood"] = df["neighborhood"].map(normalize_text)

    df["unemployment_percentage"] = (
        df["unemployment_percentage"]
        .astype(str)
        .str.replace('"', "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["unemployment_percentage"] = pd.to_numeric(df["unemployment_percentage"], errors="coerce")

    # El raw no trae el absoluto claramente separado
    df["unemployment"] = pd.NA

    out = df[["neighborhood", "unemployment_percentage", "unemployment"]].copy()
    out = out[~out["neighborhood"].isna()].copy()

    out = (
        out.groupby("neighborhood", as_index=False)
        .agg({
            "unemployment_percentage": "mean",
            "unemployment": "first",
        })
    )

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Guardado en: {OUT_CSV}")
    print(f"Barrios: {len(out)}")
    print((1 - out.isna().mean()).to_string())
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()