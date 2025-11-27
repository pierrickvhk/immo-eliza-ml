import argparse
import pandas as pd
import joblib


def load_model(model_path: str):
    """Load the trained pipeline (preprocessing + model)."""
    return joblib.load(model_path)


def predict_price(model, input_data: dict) -> float:
    """
    Predict the price for a single property.

    input_data should be a dict with the SAME keys as the training features.
    Missing keys will be treated via the preprocessors' imputers.
    """
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return float(prediction)


def main():
    parser = argparse.ArgumentParser(description="Predict house price with Immo Eliza model.")
    parser.add_argument(
        "--model",
        type=str,
        default="../models/immo_eliza_random_forest.joblib",
        help="Path to the trained model."
    )
    args = parser.parse_args()

    model = load_model(args.model)

    # Example dummy property – adapt these values to something realistic
    new_house = {
        "State of the property": "good",
        "Availability": "immediately",
        "Number of bedrooms": 3,
        "Livable surface": 120,
        "Furnished": "no",
        "Attic": "no",
        "Garage": "yes",
        "Number of garages": 1,
        "Kitchen equipment": "installed",
        "Kitchen type": "semi equipped",
        "Number of bathrooms": 1,
        "Number of showers": 1,
        "Number of toilets": 2,
        "Type of heating": "gas",
        "Type of glazing": "double",
        "Elevator": "no",
        "Number of facades": 2,
        "Garden": "yes",
        "Surface garden": 80,
        "Terrace": "yes",
        "Surface terrace": 15,
        "Total land surface": 200,
        "Swimming pool": "no",
    }

    price = predict_price(model, new_house)
    print(f"Predicted price for dummy house: € {price:,.2f}")


if __name__ == "__main__":
    main()
