import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_utils import (
    load_raw_data,
    basic_clean,
    split_features_target,
    build_preprocessor,
)


def train_and_save_model(data_path: str, model_path: str) -> None:
    df_raw = load_raw_data(data_path)
    df_clean = basic_clean(df_raw)
    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE : {mae:,.2f}")
    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R²  : {r2:,.4f}")

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Immo Eliza price model.")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/raw/immo_eliza_raw.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../models/immo_eliza_random_forest.joblib",
    )
    args = parser.parse_args()

    train_and_save_model(args.data, args.out)


if __name__ == "__main__":
    main()
