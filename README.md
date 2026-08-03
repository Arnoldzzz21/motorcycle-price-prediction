# Motorcycle Price Prediction

Predicting the resale price of used motorcycles from brand, category, year, condition, and mileage — a regression project built on ~7,600 real marketplace listings across six brands.

## Overview

| | |
|---|---|
| **Goal** | Predict the price of a used motorcycle listing |
| **Data** | ~7,600 real motorcycle listings scraped from a marketplace (BMW, Ducati, KTM, Royal Enfield, Suzuki, Yamaha) |
| **Stack** | Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost |
| **Best model** | XGBoost (tuned) — RMSE **$4,627**, R² **0.595** |

## Project workflow

1. **Load & consolidate** — six brand-level CSVs merged into a single DataFrame, with condition/year/category parsed out of a free-text field.
2. **EDA** — structure, missing values, duplicates, price distribution and skew, correlation analysis, and price breakdowns by brand/category.
3. **Cleaning** — drop duplicates and rows with missing price; handle mileage placeholders and IQR-based outliers.
4. **Feature engineering** — group rare categories, impute missing mileage by brand/category median, and engineer `bike_age`, `mileage_per_year`, and description-based features.
5. **Modeling** — compare Linear Regression, Random Forest, Gradient Boosting, and XGBoost (tuned via `RandomizedSearchCV`, 5-fold CV) on an 80/20 stratified split.
6. **Evaluation** — predicted vs. actual, residual analysis, feature importance, and error breakdown by brand/category.

## Results

| Model | RMSE | R² |
|---|---|---|
| **XGBoost (tuned)** | **$4,627** | **0.595** |
| Random Forest (tuned) | higher | lower |
| Gradient Boosting | higher | lower |
| Linear Regression | higher | lower |

**Key price drivers:** brand is the dominant signal, followed by category and bike age; raw mileage contributes less than expected.

**Where the model struggles:** premium and limited-edition motorcycles (e.g. a $99,950 Ducati Sportbike predicted at ~$22K) — the model is reliable for mainstream, high-volume segments (Royal Enfield, Standard/Cruiser categories) but systematically undervalues rare, high-end listings.

## Repository structure

```
motorcycle-price-prediction/
├── Motorcycle_Price_Prediction.ipynb   # full analysis: EDA -> feature engineering -> modeling -> evaluation
├── data/                                # raw per-brand listing CSVs
│   ├── BMW_bike.csv
│   ├── ducatti_bike.csv
│   ├── KTM_bike.csv
│   ├── Royal_Enfield_Standard_bike.csv
│   ├── Suzuki_bike.csv
│   └── Yamaha_bike.csv
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/Arnoldzzz21/motorcycle-price-prediction.git
cd motorcycle-price-prediction
pip install -r requirements.txt
jupyter notebook Motorcycle_Price_Prediction.ipynb
```

## Next steps

- Re-introduce a grouped version of the `model` column (e.g. a "premium tier" flag) to close the gap on rare, high-value motorcycles.
- Expose the trained model through an interactive Streamlit dashboard for a live demo.

## Author

**Arnoldo** — [github.com/Arnoldzzz21](https://github.com/Arnoldzzz21)
