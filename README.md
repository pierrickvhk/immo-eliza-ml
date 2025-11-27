# Immo Eliza – House Price Prediction (Regression)

This repository contains a production-like, end-to-end regression pipeline to predict Belgian real-estate property prices for **Immo Eliza**.

The project was built as a solo consolidation exercise with a clean, layered approach:

- **v1** → load and inspect raw data
- **v2** → first ML pipelines, model comparison
- **v3** → tuned Random Forest with strong generalization

---

## 1. Project Overview

**Goal:** Predict the selling price of a property in Belgium based on listing features such as number of bedrooms, equipment types, and various living surface areas.

✅ The final model includes reusable preprocessing wrapped together with the model as a single `scikit-learn Pipeline`.

---

## 2. Repository Structure

```text
immo-eliza-ml/
├─ data/
│  ├─ raw/
│  │  └─ immo_eliza_raw.csv            # raw input data
│  └─ processed/                       # (optional) cleaned exports
├─ models/
│  └─ immo_eliza_random_forest.joblib  # final pipeline (preprocessing + model)
├─ notebooks/
│  └─ 01_immo_eliza_regression.ipynb   # notebook with v1, v2 and v3 experiments
├─ src/
│  ├─ train.py                         # model training + saving script
│  └─ predict.py                       # inference script with new dummy data
├─ requirements.txt
└─ README.md
```

---

## 3. Data & Preprocessing

The dataset contains scraped Belgian real-estate listings with features such as:

- Comfort features → `Number of bedrooms`, `Number of bathrooms`, `Number of toilets`, `Number of facades`
- Surface areas → `Livable surface`, `Surface garden`, `Surface terrace`, `Total land surface`
- Equipment → `Kitchen type`, `Type of heating`, `Type of glazing`, `Furnished`, `Terrace`, `Garden`, `Garage`, `Elevator`, `Swimming pool`

### Key Preprocessing Steps
1. **Clean target (`Price`)**
   - Remove `€` signs, spaces, text noise
   - Normalize decimal separators
   - Convert to `float`
   - Drop rows where price is missing or invalid

2. **Convert hidden text-based numerical features**
   ✔ Livable, garden, terrace and land surfaces converted from text → float  
   ✔ Non-numeric characters removed (e.g., `m²`, spaces, symbols)

3. **Drop low-signal identifier columns**
   - `url`
   - `Property ID`

4. **Reusable preprocessing system**
   - Numeric features -> imputed using `median` + standardized
   - Categorical features -> imputed using `most_frequent` + One-Hot encoded
   - Everything wrapped in a reusable sklearn `Pipeline` for reuse in scripts

---

## 4. Model Comparison & Final Results

Three regression models were trained using the same pipeline (v2):

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Support Vector Regression (SVR)**

Random Forest performed best on unseen test data and was selected and tuned (v3).

### Final Tuned Model — Random Forest (`v3`)
| Metric | Train | Test |
|---|---:|---:|
| **MAE** | ~66,783 | ~97,186 |
| **RMSE** | ~168,208 | ~206,821 |
| **R²** | 0.738 | 0.618 |

### Conclusion
- Random Forest shows the strongest performance on unseen data (R² ≈ 0.62)
- Train/test gap is healthy -> **no heavy overfitting**
- The model extracts meaningful signal despite noisy real-estate data
- Linear Regression and SVR work as baselines but do not generalize as strongly

---

## 5. How to Run the Project

### 5.1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/immo-eliza-ml.git
cd immo-eliza-ml

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Make sure your dataset is placed in:

```text
data/raw/immo_eliza_raw.csv
```

---

### 5.2. Train the Model (Notebook)

Open:

```text
notebooks/01_immo_eliza_regression.ipynb
```

Run all cells.  
The notebook will:

1. load raw data
2. clean the target
3. convert surfaces to numeric
4. build reusable preprocessing
5. train and tune a Random Forest pipeline
6. save the best pipeline to the `models/` folder

---

### 5.3. Train the Model (CLI)

```bash
cd src
python train.py --data ../data/raw/immo_eliza_raw.csv --out ../models/immo_eliza_random_forest.joblib
```

This will:

✅ Train the Random Forest pipeline  
✅ Evaluate on held-out test data  
✅ Save the pipeline (preprocessing + model) to disk

---

### 5.4. Run a Prediction on a Dummy House

```bash
cd src
python predict.py --model ../models/immo_eliza_random_forest.joblib
```

Inside `predict.py` there is a `new_house` dictionary — you can edit it to test any new listing.

---

## 6. Future Improvements

- Add more real-world feature engineering (postcode, province, build year, ...)
- Try `log1p` transform on target or skewed numeric predictors
- More extensive hyperparameter tuning (e.g., `GridSearchCV`)
- Build an API or UI (e.g., Streamlit) on top of this model
- Expand model comparison with Gradient Boosting regressors
