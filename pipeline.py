"""
Data loading, cleaning, feature engineering and modeling pipeline.

Reconstructed from Motorcycle_Price_Prediction.ipynb so the Streamlit app is
self-contained: it trains from the raw per-brand CSVs in data/, with every
heavy step wrapped in Streamlit caching by the app itself.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

RAW_FILES = {
    "BMW": "BMW_bike.csv",
    "Ducati": "ducatti_bike.csv",
    "KTM": "KTM_bike.csv",
    "Royal Enfield": "Royal_Enfield_Standard_bike.csv",
    "Suzuki": "Suzuki_bike.csv",
    "Yamaha": "Yamaha_bike.csv",
}

MODEL_COL_BY_BRAND = {
    "BMW": "Bike", "Ducati": "Bike name ", "KTM": "Bike",
    "Royal Enfield": "bike", "Suzuki": "BIke name", "Yamaha": "Bike name",
}
TYPE_COL_BY_BRAND = {
    "BMW": "Types and Used Time", "Ducati": "Time of USed", "KTM": "Types and Used Time",
    "Royal Enfield": "Types ", "Suzuki": "Types and Used Time", "Yamaha": "Types and Used  Time",
}

REFERENCE_YEAR = 2024  # matches the year the original dataset was scraped/labeled
RARE_CATEGORY_THRESHOLD = 50

# Brand display order used consistently across every chart in the app
BRAND_ORDER = ["BMW", "Ducati", "KTM", "Royal Enfield", "Suzuki", "Yamaha"]


def _parse_type_field(text, brand):
    if pd.isna(text):
        return pd.Series([np.nan, np.nan, np.nan])
    text = str(text).strip()
    condition = "Used"
    if text.lower().startswith("new"):
        condition, text = "New", text[3:].strip()
    elif text.lower().startswith("used"):
        condition, text = "Used", text[4:].strip()
    match = re.match(r"(\d{4})\s+(.*)", text)
    if not match:
        return pd.Series([condition, np.nan, np.nan])
    year = int(match.group(1))
    rest = match.group(2).strip()
    category = re.sub(re.escape(brand), "", rest, flags=re.IGNORECASE).strip() or rest
    return pd.Series([condition, year, category])


def _parse_price(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _parse_mileage(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace("miles", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def load_and_engineer_features() -> pd.DataFrame:
    """Load the six brand CSVs, clean them, and engineer the modeling features.

    Returns the full DataFrame (still carries `model`, `category`, `year` for
    storytelling) with every engineered column already added.
    """
    raw_dfs = {}
    for brand, filename in RAW_FILES.items():
        raw_dfs[brand] = pd.read_csv(
            DATA_DIR / filename, encoding="latin-1", engine="pyarrow", on_bad_lines="skip",
        )

    frames = []
    for brand, raw in raw_dfs.items():
        d = raw.rename(columns={
            MODEL_COL_BY_BRAND[brand]: "model",
            TYPE_COL_BY_BRAND[brand]: "type_raw",
        })
        d["brand"] = brand
        d[["condition", "year", "category"]] = d["type_raw"].apply(lambda x: _parse_type_field(x, brand))
        d["price"] = d["price"].apply(_parse_price)
        d["mileage"] = d["mileage"].apply(_parse_mileage)
        frames.append(d[["brand", "model", "year", "condition", "category", "mileage", "price", "description"]])

    df = pd.concat(frames, ignore_index=True)

    # --- clean ---
    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])
    df.loc[df["condition"] == "New", "mileage"] = df.loc[df["condition"] == "New", "mileage"].fillna(0)
    df.loc[df["mileage"] > 500_000, "mileage"] = np.nan  # data-entry placeholder, not a real reading
    df = df.reset_index(drop=True)

    # --- feature engineering ---
    category_counts = df["category"].value_counts()
    rare_categories = category_counts[category_counts < RARE_CATEGORY_THRESHOLD].index.tolist()
    df["category_grouped"] = df["category"].apply(lambda c: "Other" if c in rare_categories else c)

    group_median = df.groupby(["brand", "category_grouped"])["mileage"].transform("median")
    brand_median = df.groupby("brand")["mileage"].transform("median")
    df["mileage"] = df["mileage"].fillna(group_median).fillna(brand_median)

    df["bike_age"] = REFERENCE_YEAR - df["year"]
    df["mileage_per_year"] = df["mileage"] / df["bike_age"].replace(0, np.nan)
    df["mileage_per_year"] = df["mileage_per_year"].fillna(0)
    df["has_description"] = df["description"].notna().astype(int)
    df["description_length"] = df["description"].fillna("").apply(lambda t: len(str(t).split()))

    return df


def to_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop free-text/leaky columns and one-hot encode categoricals."""
    drop_cols = ["model", "description", "category", "year"]
    model_df = df.drop(columns=drop_cols)
    model_df = pd.get_dummies(model_df, columns=["brand", "category_grouped", "condition"], drop_first=True)
    return model_df


def feature_groups(columns) -> dict:
    """Map each one-hot/raw column to the human-readable feature it belongs to."""
    mapping = {}
    for c in columns:
        if c.startswith("brand_"):
            mapping[c] = "Brand"
        elif c.startswith("category_grouped_"):
            mapping[c] = "Category"
        elif c.startswith("condition_"):
            mapping[c] = "Condition (new/used)"
        elif c == "bike_age":
            mapping[c] = "Bike age"
        elif c == "mileage":
            mapping[c] = "Mileage"
        elif c == "mileage_per_year":
            mapping[c] = "Mileage per year"
        elif c == "has_description":
            mapping[c] = "Has description"
        elif c == "description_length":
            mapping[c] = "Description length"
        else:
            mapping[c] = c
    return mapping


def build_input_row(feature_columns, *, brand, category_grouped, condition, year, mileage, description_length):
    """Build a single-row, correctly one-hot-encoded feature vector for a
    user-specified bike, aligned to `feature_columns` (the training columns).
    """
    bike_age = REFERENCE_YEAR - year
    mileage_per_year = (mileage / bike_age) if bike_age else 0.0
    has_description = 1 if description_length > 0 else 0

    row = {c: 0 for c in feature_columns}
    row["mileage"] = mileage
    row["bike_age"] = bike_age
    row["mileage_per_year"] = mileage_per_year
    row["has_description"] = has_description
    row["description_length"] = description_length

    brand_col = f"brand_{brand}"
    if brand_col in row:
        row[brand_col] = 1  # if brand is the drop_first reference level, all brand_* stay 0 (correct)

    cat_col = f"category_grouped_{category_grouped}"
    if cat_col in row:
        row[cat_col] = 1

    cond_col = f"condition_{condition}"
    if cond_col in row:
        row[cond_col] = 1

    return pd.DataFrame([row], columns=feature_columns)
