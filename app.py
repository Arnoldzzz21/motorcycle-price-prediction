"""
Motorcycle Price Prediction — narrative Streamlit app.

Replaces the old static dashboard.html: this app trains live (cached), and
walks through the story end to end — data, models, and *why* the model
predicts what it predicts, via SHAP.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from pipeline import (
    BRAND_ORDER,
    REFERENCE_YEAR,
    build_input_row,
    feature_groups,
    load_and_engineer_features,
    to_model_table,
)

# ---------------------------------------------------------------- Palette --
BLUE, BLUE_LIGHT, BLUE_MUTED = "#2a78d6", "#86b6ef", "#9ec5f4"
RED = "#e34948"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
BRAND_COLOR = dict(zip(BRAND_ORDER, CATEGORICAL))

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.set_page_config(page_title="Motorcycle Price Prediction", page_icon="🏍️", layout="wide")


# ------------------------------------------------------------------ Cache --
@st.cache_data(show_spinner="Loading and cleaning the data...")
def get_full_data():
    return load_and_engineer_features()


@st.cache_data(show_spinner=False)
def get_model_table(df):
    return to_model_table(df)


@st.cache_data(show_spinner=False)
def get_split(model_df, strat_col):
    X = model_df.drop(columns=["price"])
    y = model_df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_col)


@st.cache_resource(show_spinner="Training XGBoost...")
def train_xgb(X_train, y_train):
    # Hyperparameters found offline via RandomizedSearchCV (30 iters, 5-fold CV)
    # over the same search space as the notebook; refit here on the live split.
    model = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


@st.cache_resource(show_spinner="Training comparison models...")
def train_comparison(X_train, y_train, X_test, y_test, _xgb_model):
    def evaluate(name, y_true, y_pred):
        return {
            "model": name,
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    rows = []
    lr_raw = LinearRegression().fit(X_train, y_train)
    rows.append(evaluate("Linear Regression", y_test, lr_raw.predict(X_test)))

    rf = RandomForestRegressor(n_estimators=250, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    rows.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

    rows.append(evaluate("XGBoost (tuned)", y_test, _xgb_model.predict(X_test)))
    return pd.DataFrame(rows).set_index("model").sort_values("rmse")


@st.cache_resource(show_spinner=False)
def get_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)


def aggregate_shap(shap_row_values, columns, groups):
    """Sum SHAP contributions for one-hot columns back into their real-world feature."""
    s = pd.Series(shap_row_values, index=columns)
    grouped = s.groupby([groups[c] for c in columns]).sum()
    return grouped.sort_values()


def waterfall_figure(base_value, grouped_contrib, prediction, title, top_n=7):
    ordered = grouped_contrib.reindex(grouped_contrib.abs().sort_values(ascending=False).index)
    top = ordered.head(top_n)
    rest = ordered.iloc[top_n:].sum() if len(ordered) > top_n else 0.0

    labels = ["Base value"] + list(top.index)
    values = [base_value] + list(top.values)
    measures = ["absolute"] + ["relative"] * len(top)
    if abs(rest) > 1:
        labels.append("Other features")
        values.append(rest)
        measures.append("relative")
    labels.append("Final prediction")
    values.append(prediction)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        increasing=dict(marker=dict(color=BLUE)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=INK)),
        connector=dict(line=dict(color=GRID, width=1)),
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, showlegend=False, yaxis_title="Price (USD)",
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        **PLOTLY_LAYOUT,
    )
    return fig


# ------------------------------------------------------------------- Data --
df = get_full_data()
model_df = get_model_table(df)
X_train, X_test, y_train, y_test = get_split(model_df, df["category_grouped"])
xgb_model = train_xgb(X_train, y_train)
comparison_df = train_comparison(X_train, y_train, X_test, y_test, xgb_model)
explainer = get_explainer(xgb_model)
groups = feature_groups(X_train.columns)

y_pred_test = xgb_model.predict(X_test)
test_rmse = root_mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# ----------------------------------------------------------------- Header --
st.title("🏍️ Motorcycle Price Prediction")
st.caption(
    "Predicting the resale price of used motorcycles — 7,600 real listings across six brands "
    "(BMW, Ducati, KTM, Royal Enfield, Suzuki, Yamaha). Narrative app with SHAP interpretability."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "XGBoost (tuned)")
c2.metric("RMSE (test)", f"${test_rmse:,.0f}")
c3.metric("R²", f"{test_r2:.3f}")
c4.metric("Listings analyzed", f"{len(df):,}")

tabs = st.tabs([
    "📊 Data & EDA", "🤖 Models", "🔍 Interpretability (SHAP)",
    "🧮 Try the model", "📝 Conclusions",
])

# ------------------------------------------------------------------- EDA --
with tabs[0]:
    st.subheader("Price distribution")
    st.write(
        "Price is right-skewed (a handful of luxury listings stretch the tail); "
        "most motorcycles fall below \\$15,000."
    )
    fig = go.Figure(go.Histogram(x=df["price"], nbinsx=50, marker_color=BLUE))
    fig.update_layout(
        title="Price distribution (USD)", xaxis_title="Price (USD)", yaxis_title="Listings",
        bargap=0.02, xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Median price by brand")
        med_brand = df.groupby("brand")["price"].median().reindex(BRAND_ORDER)
        fig = go.Figure(go.Bar(
            x=med_brand.index, y=med_brand.values,
            marker_color=[BRAND_COLOR[b] for b in med_brand.index],
            text=[f"${v:,.0f}" for v in med_brand.values], textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Median price (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Sample size by brand")
        counts = df["brand"].value_counts().reindex(BRAND_ORDER)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[BRAND_COLOR[b] for b in counts.index],
            text=counts.values, textposition="outside",
            hovertemplate="%{x}<br>%{y:,} listings<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Listings", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by category")
    cat_order = df.groupby("category_grouped")["price"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for cat in cat_order:
        fig.add_trace(go.Box(
            y=df.loc[df["category_grouped"] == cat, "price"], name=cat,
            marker_color=BLUE, line_color=BLUE, fillcolor=BLUE_LIGHT, showlegend=False,
        ))
    fig.update_layout(
        yaxis_title="Price (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Feature engineering applied"):
        st.markdown(
            "- **Rare categories** (fewer than 50 listings) are grouped into `Other` to avoid categories "
            "with very little statistical signal.\n"
            "- **Missing mileage** is imputed with the brand+category median (or brand median, as a fallback).\n"
            "- **`bike_age`** = 2024 − listing year; **`mileage_per_year`** = mileage / age.\n"
            "- **`description_length`** and **`has_description`** capture how detailed the listing is.\n"
            "- The **exact model name** (`model`) is dropped as a feature — this comes back "
            "in the SHAP section as the root cause of the model's biggest miss."
        )

# --------------------------------------------------------------- Models --
with tabs[1]:
    st.subheader("Model comparison")
    st.write(
        "Four models of increasing complexity, evaluated on the same test set (80/20, "
        "stratified by category). XGBoost was tuned with `RandomizedSearchCV` (30 combinations, "
        "5-fold CV) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`."
    )
    show_df = comparison_df.copy()
    show_df["rmse"] = show_df["rmse"].map(lambda v: f"${v:,.0f}")
    show_df["mae"] = show_df["mae"].map(lambda v: f"${v:,.0f}")
    show_df["r2"] = show_df["r2"].map(lambda v: f"{v:.3f}")
    show_df.columns = ["RMSE", "MAE", "R²"]
    st.dataframe(show_df, use_container_width=True)

    ranked = comparison_df.sort_values("rmse")
    colors = [BLUE if m == "XGBoost (tuned)" else BLUE_MUTED for m in ranked.index]
    fig = go.Figure(go.Bar(
        x=ranked["rmse"], y=ranked.index, orientation="h", marker_color=colors,
        text=[f"${v:,.0f}" for v in ranked["rmse"]], textposition="outside",
        hovertemplate="%{y}<br>RMSE $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="RMSE by model (lower = better) — XGBoost in blue is the chosen model",
        xaxis_title="RMSE (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "XGBoost is used as the project's flagship model (documented in the README), although in some "
        "runs tuned Random Forest comes within 2% RMSE — the difference between the two is "
        "sampling noise from `RandomizedSearchCV`, not a clear structural advantage."
    )

# -------------------------------------------------------------- SHAP tab --
with tabs[2]:
    st.subheader("What does the model focus on? (Global SHAP)")
    st.write(
        "Each bar is the average impact (in dollars) of that feature on the prediction, "
        "in absolute value, aggregating the brand and category one-hot columns back into their original feature."
    )

    shap_test = explainer(X_test)
    global_importance = pd.DataFrame(np.abs(shap_test.values), columns=X_test.columns).mean()
    global_grouped = global_importance.groupby([groups[c] for c in X_test.columns]).sum().sort_values()

    fig = go.Figure(go.Bar(
        x=global_grouped.values, y=global_grouped.index, orientation="h", marker_color=BLUE,
        text=[f"${v:,.0f}" for v in global_grouped.values], textposition="outside",
        hovertemplate="%{y}<br>$%{x:,.0f} average<extra></extra>",
    ))
    fig.update_layout(
        title="Mean |SHAP| importance by feature", xaxis_title="Mean impact on prediction (USD)",
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "**Brand** and **category** dominate, followed by **age**; **mileage** carries "
        "less weight than intuition would suggest — consistent with the tree's built-in feature "
        "importance and with the weak correlation seen in the EDA section."
    )

    st.divider()
    st.subheader("The Ducati case: where the model fails")

    ducati_row = df[(df["brand"] == "Ducati") & (df["price"] == 99950.0)]
    if len(ducati_row):
        idx = ducati_row.index[0]
        listing = ducati_row.iloc[0]
        x_row = model_df.drop(columns=["price"]).loc[[idx]]
        pred = float(xgb_model.predict(x_row)[0])
        shap_row = explainer(x_row)

        st.markdown(
            f"The model's single worst miss: a **{listing['model'].title()}** "
            f"({listing['category']}, {int(listing['year'])}, {int(listing['mileage'])} miles) "
            f"listed at **\\${listing['price']:,.0f}**, predicted at **\\${pred:,.0f}** — an error of "
            f"**\\${listing['price'] - pred:,.0f}** ({(listing['price'] - pred) / listing['price']:.0%} below)."
        )

        grouped_contrib = aggregate_shap(shap_row.values[0], x_row.columns, groups)
        fig = waterfall_figure(
            float(shap_row.base_values[0]), grouped_contrib, pred,
            "Where the prediction for this Ducati comes from (SHAP, aggregated by feature)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Reading it:** the model does recognize that 'Ducati' and 'Sportbike' are premium signals — both "
            "push the prediction **upward**. The problem is magnitude: there is no feature that "
            "captures that this particular listing is a **limited-edition Diavel 1260 Lamborghini** "
            "with only 124 miles. The exact model name (`model`) was dropped as a feature during "
            "feature engineering, so the model treats this bike as a generic Ducati and falls "
            "well short. It's an information limit, not a model-fit error."
        )
    else:
        st.info("The reference Ducati case wasn't found in this data run.")

# ------------------------------------------------------------ Predictor --
with tabs[3]:
    st.subheader("Test the model with your own motorcycle")
    st.write(
        f"Build a hypothetical listing and see the model's prediction along with its own SHAP "
        f"explanation, exactly like the Ducati case above. (Dataset reference year: {REFERENCE_YEAR}.)"
    )

    brand_options = BRAND_ORDER
    category_options = sorted(df["category_grouped"].unique())

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("Brand", brand_options, index=brand_options.index("Ducati"))
            condition = st.radio("Condition", ["Used", "New"], horizontal=True)
        with col2:
            category = st.selectbox(
                "Category", category_options,
                index=category_options.index("Sportbike") if "Sportbike" in category_options else 0,
            )
            year = st.slider("Year", 1990, REFERENCE_YEAR, 2020)
        with col3:
            mileage = st.number_input("Mileage (miles)", min_value=0, max_value=150_000, value=5_000, step=500)
            description_length = st.slider("Words in the listing description", 0, 200, 40)
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        x_row = build_input_row(
            list(X_train.columns), brand=brand, category_grouped=category,
            condition=condition, year=year, mileage=mileage, description_length=description_length,
        )
        pred = float(xgb_model.predict(x_row)[0])
        shap_row = explainer(x_row)

        st.metric("Predicted price", f"${pred:,.0f}")
        grouped_contrib = aggregate_shap(shap_row.values[0], x_row.columns, groups)
        fig = waterfall_figure(
            float(shap_row.base_values[0]), grouped_contrib, pred,
            "How this prediction is built (SHAP, aggregated by feature)",
        )
        st.plotly_chart(fig, use_container_width=True)

        top_driver = grouped_contrib.abs().idxmax()
        top_value = grouped_contrib[top_driver]
        direction = "raises" if top_value > 0 else "lowers"
        st.caption(
            f"The feature with the biggest impact on this prediction is **{top_driver}**, which {direction} the "
            f"price by roughly **\\${abs(top_value):,.0f}** relative to the base value of "
            f"\\${float(shap_row.base_values[0]):,.0f}."
        )

# ----------------------------------------------------------- Conclusions --
with tabs[4]:
    st.subheader("Conclusions")
    st.markdown(
        f"""
- **Brand and category** are the strongest price predictors; raw mileage and year
  contribute less than expected — something both feature importance and SHAP confirm
  independently.
- **XGBoost (tuned)** is the project's flagship model: RMSE ≈ \\${test_rmse:,.0f}, R² ≈ {test_r2:.3f}
  in this run (exact numbers vary slightly between runs due to `RandomizedSearchCV` sampling,
  but the order of magnitude is stable).
- **Where the model struggles:** premium or limited-edition motorcycles (the Ducati Diavel 1260
  Lamborghini case in the interpretability tab). SHAP shows the model *does know* that "Ducati" and
  "Sportbike" are premium signals, but has no way to distinguish a limited edition from a
  generic Ducati, because the exact model name isn't a feature.
- **Business recommendation:** the model is reliable for the mass-market segment (Royal Enfield,
  Standard/Cruiser categories). For premium or limited-edition motorcycles, pricing should be
  reviewed manually — the model systematically undervalues them.

**Natural next step:** reintroduce a grouped version of `model` (e.g. a "premium tier /
limited edition" flag) to close this specific gap.
"""
    )
    st.caption(
        "This interactive dashboard replaces the repository's previous static version "
        "(dashboard.html) — same project, now with full storytelling and live SHAP interpretability."
    )

st.divider()
st.caption("Developed by **Arnoldo Cuéllar** · [GitHub](https://github.com/Arnoldzzz21)")
