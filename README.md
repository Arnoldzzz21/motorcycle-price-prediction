# Motorcycle Price Prediction

Predicting the resale price of used motorcycles from brand, category, year, condition, and mileage — a regression project built on ~7,600 real marketplace listings across six brands.

## Overview

| | |
|---|---|
| **Goal** | Predict the price of a used motorcycle listing |
| **Data** | ~7,600 real motorcycle listings scraped from a marketplace (BMW, Ducati, KTM, Royal Enfield, Suzuki, Yamaha) |
| **Stack** | Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Plotly, Streamlit |
| **Best model** | XGBoost (tuned) — RMSE ≈ $4,600–4,700, R² ≈ 0.58–0.60 (varies slightly run to run) |

## Interactive app

`app.py` is a narrative Streamlit app that replaces the old static `dashboard.html`. It trains live
(cached) straight from the raw CSVs in `data/`, and walks through the full story:

1. **Datos y EDA** — price distribution, price by brand/category, feature engineering notes.
2. **Modelos** — Linear Regression, Random Forest, Gradient Boosting, and XGBoost (tuned via
   `RandomizedSearchCV`), compared on the same test set.
3. **Interpretabilidad (SHAP)** — global feature importance, plus a deep dive into the model's worst
   single miss: a $99,950 Ducati Diavel 1260 Lamborghini (a limited-edition collab bike) predicted at
   ~$21K. SHAP shows the model *does* recognize "Ducati" and "Sportbike" as premium signals — it just
   has no feature that captures this specific listing is a rare special edition, because the exact
   model name was dropped during feature engineering.
4. **Probar el modelo** — build a hypothetical listing and get a live prediction with its own SHAP
   waterfall explanation.
5. **Conclusiones** — business interpretation and recommended next step.

### Running it locally

```bash
git clone https://github.com/Arnoldzzz21/motorcycle-price-prediction.git
cd motorcycle-price-prediction
pip install -r requirements.txt
streamlit run app.py
```

### Deploying (Streamlit Community Cloud)

1. Push this repo (with `app.py`, `pipeline.py`, `requirements.txt`, `.streamlit/config.toml`, and
   `data/`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and point it at `app.py`.
3. First load trains the model live (a few seconds) and caches it for the rest of the session.

## Project workflow (also in the notebook)

1. **Load & consolidate** — six brand-level CSVs merged into a single DataFrame, with condition/year/category parsed out of a free-text field.
2. **EDA** — structure, missing values, duplicates, price distribution and skew, correlation analysis, and price breakdowns by brand/category.
3. **Cleaning** — drop duplicates and rows with missing price; handle mileage placeholders.
4. **Feature engineering** — group rare categories, impute missing mileage by brand/category median, and engineer `bike_age`, `mileage_per_year`, and description-based features.
5. **Modeling** — compare Linear Regression, Random Forest, Gradient Boosting, and XGBoost (tuned via `RandomizedSearchCV`, 5-fold CV) on an 80/20 stratified split.
6. **Evaluation** — predicted vs. actual, residual analysis, feature importance, SHAP, and error breakdown by brand/category.

**Key price drivers:** brand is the dominant signal, followed by category and bike age; raw mileage contributes less than expected — confirmed independently by both feature importance and SHAP.

**Where the model struggles:** premium and limited-edition motorcycles — see the SHAP deep-dive in the app for the full Ducati Diavel 1260 Lamborghini story.

## Repository structure

```
motorcycle-price-prediction/
├── app.py                               # narrative Streamlit app (SHAP interpretability, live predictor)
├── pipeline.py                          # shared load/clean/feature-engineering pipeline used by app.py
├── Motorcycle_Price_Prediction.ipynb    # full analysis notebook: EDA -> feature engineering -> modeling -> evaluation
├── data/                                # raw per-brand listing CSVs
│   ├── BMW_bike.csv
│   ├── ducatti_bike.csv
│   ├── KTM_bike.csv
│   ├── Royal_Enfield_Standard_bike.csv
│   ├── Suzuki_bike.csv
│   └── Yamaha_bike.csv
├── .streamlit/config.toml               # app theme
├── requirements.txt
└── README.md
```

## Next steps

- Re-introduce a grouped version of the `model` column (e.g. a "premium tier" flag) to close the gap on rare, high-value motorcycles — the SHAP section quantifies exactly why this gap exists.

## Author

**Arnoldo** — [github.com/Arnoldzzz21](https://github.com/Arnoldzzz21)
