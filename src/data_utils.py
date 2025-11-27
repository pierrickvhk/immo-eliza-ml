import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Price"

DROP_COLS = ["url", "Property ID"]

SURFACE_COLS = [
    "Livable surface",
    "Surface garden",
    "Surface terrace",
    "Total land surface",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Immo Eliza CSV file."""
    return pd.read_csv(path)


def clean_price(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Clean the Price column and convert it to float."""
    price_str = df[target_col].astype(str)
    price_str = price_str.str.replace(r"[^\d,\.]", "", regex=True)
    price_str = price_str.str.replace(".", "", regex=False)
    price_str = price_str.str.replace(",", ".", regex=False)
    price_str = price_str.replace("", np.nan)
    df[target_col] = price_str.astype(float)
    df = df.dropna(subset=[target_col])
    return df


def convert_surface_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert surface-related columns from text to numeric floats."""
    for col in SURFACE_COLS:
        if col in df.columns:
            s = df[col].astype(str)
            s = s.str.replace(r"[^\d,\.]", "", regex=True)
            s = s.str.replace(".", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
            s = s.replace("", np.nan)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps used in the notebook/train script."""
    df = df.drop_duplicates()
    df = clean_price(df, TARGET_COL)
    df = convert_surface_columns(df)
    return df


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X (features) and y (target)."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    y = y.loc[X.index]
    return X, y


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create ColumnTransformer for numeric and categorical features."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def save_clean_data(df: pd.DataFrame, path: str) -> None:
    """Save a cleaned version of the dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
