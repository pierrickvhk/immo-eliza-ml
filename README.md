# Immo Eliza – Belgian Real Estate Price Prediction (Regression)

A reusable and evaluated machine learning regression project built for the real estate company **Immo Eliza**, focused on predicting property prices in Belgium.

---

## 🎯 Goal

Predict the selling price (**Price**) of a real-estate property using classical ML models, with a strong emphasis on reusable preprocessing pipelines and honest model evaluation.

---

## 📁 Project Structure

```text
immo-eliza-ml/
├─ data/
│  ├─ raw/
│  │  └─ immo_eliza_raw.csv              # original scraped dataset
│  └─ processed/
│     └─ immo_eliza_clean.csv            # cleaned & numeric-converted dataset (generated)
├─ models/
│  └─ immo_eliza_random_forest.joblib             # final trained Pipeline (preprocessing + Random Forest)
├─ notebooks/
│  └─ 01_baseline_regression.ipynb     # experiments v1 → v3 with evaluation
├─ src/
│  ├─ data_utils.py                     # reusable loading + cleaning + preprocessing logic
│  ├─ make_clean_data.py                # generates cleaned CSV for reuse
│  ├─ train.py                          # trains & exports model pipeline
│  └─ predict.py                        # loads model and predicts on dummy data
├─ requirements.txt
└─ README.md
```

---

## 🧼 Data Cleaning & Preprocessing

The project consumes raw scraped listings and converts them into an ML-ready format. Key reusable steps implemented in `data_utils.py` include:

### ✔ Cleaning
- Remove duplicates
- Convert `Price` from noisy text → `float`
- Extract and convert hidden numerical surface features
- Drop rows where Price could not be converted

### 🔁 Reusable Preprocessor Pipeline
A `ColumnTransformer` embedded within a `sklearn.Pipeline` applies:

| Feature type | Transformation |
|---|---|
| Numeric | Median imputation + `StandardScaler()` |
| Categorical | Most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` |

All transformations are encapsulated so the trained model can be reused safely during inference.

---

## 🤖 Model Training & Performance (v3)

Three regression models were evaluated in the notebook:

1. `LinearRegression` (baseline)
2. `RandomForestRegressor` (non-linear)
3. `SVR` (non-linear comparison)

### 🏆 Final Selected Model
Random Forest performed best on unseen data after tuning. The final pipeline includes both preprocessing and a tuned `RandomForestRegressor`.

| Metric | Train | Unseen Test |
|---|---:|---:|
| **MAE** | 66,782.55 | 90,544.47 |
| **RMSE** | 168,208.26 | 194,357.23 |
| **R²** | 0.738 | 0.663 |

✅ Strong generalization  
✅ No heavy overfitting  
✅ Realistic price error range for Belgian housing listings

The final pipeline model is saved as:

```text
models/rf_price_model.joblib
```

---

## 🔍 How to Run

### 1. Create & Activate Virtual Environment

```bash
cd immo-eliza-ml
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate a Cleaned Dataset for Reuse

```bash
cd src
python make_clean_data.py --data ../data/raw/immo_eliza_raw.csv --out ../data/processed/immo_eliza_clean.csv
```

### 4. Train the Random Forest Pipeline

```bash
cd src
python train.py --data ../data/processed/immo_eliza_clean.csv --out ../models/immo_eliza_random_forest.joblib
```

### 5. Predict Price for Dummy House Data

```bash
cd src
python predict.py --model ../models/immo_eliza_random_forest.joblib
```

You can edit the `new_house` dictionary inside `predict.py` to test new property data.

---

## 🚀 Road to Mastery

This project represents a personal milestone for me.

I didn’t start by trying to build the most impressive model immediately —  
instead, I invested a large portion of my time upfront into studying regression, preprocessing, pipelines, and model evaluation through structured courses and applied reading.

That learning-first decision cost me time at the start, but it paid off:

✔ I now understand every preprocessing transformation we applied  
✔ I can explain why each model performs differently  
✔ I built a reusable pipeline with confidence instead of trial-and-error  
✔ The result is not only functional, but meaningful to me because I understand _how_ we got here  

I’m genuinely happy with the outcome now, not only because the model performs well, but because I built it with understanding.

---

## ⭐ Future Improvements (Optional)

If this project grows further, here are impactful refinements I would explore next:

- **Feature engineering:**
  - Extract structured province or postcode information from raw addresses or URLs
  - Derive location clusters or median province prices as extra predictors
  - Add year built, renovation status, energy score, proximity indicators, etc.

- **Model expansion:**
  - Reintroduce **XGBoost** or `GradientBoostingRegressor` as an exported 3rd model for saved model comparison
  - Explore stacking or ensemble blending strategies

- **Pipeline improvements:**
  - Test `log1p` transform on target or skewed numeric fields for extra stability
  - Replace `StandardScaler` with `RobustScaler` on surface fields if heavy outliers persist

- **Evaluation improvements:**
  - More exhaustive hyperparameter search using `GridSearchCV`
  - Add cross-validation metric reporting to the README

- **Deployment / UX:**
  - Expose the model behind a lightweight API or Streamlit UI
  - Build an interactive prediction page for customers to test new listings rapidly
  - Export a small explainability dashboard using SHAP or permutation importance


