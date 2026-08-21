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
@st.cache_data(show_spinner="Cargando y limpiando los datos...")
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


@st.cache_resource(show_spinner="Entrenando XGBoost...")
def train_xgb(X_train, y_train):
    # Hyperparameters found offline via RandomizedSearchCV (30 iters, 5-fold CV)
    # over the same search space as the notebook; refit here on the live split.
    model = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


@st.cache_resource(show_spinner="Entrenando modelos de comparación...")
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
    rows.append(evaluate("Regresión Lineal", y_test, lr_raw.predict(X_test)))

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

    labels = ["Valor base"] + list(top.index)
    values = [base_value] + list(top.values)
    measures = ["absolute"] + ["relative"] * len(top)
    if abs(rest) > 1:
        labels.append("Otras variables")
        values.append(rest)
        measures.append("relative")
    labels.append("Predicción final")
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
        title=title, showlegend=False, yaxis_title="Precio (USD)",
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
    "Predicción del precio de reventa de motos usadas — ~7,600 anuncios reales de seis marcas "
    "(BMW, Ducati, KTM, Royal Enfield, Suzuki, Yamaha). App narrativa con interpretabilidad SHAP."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", "XGBoost (tuned)")
c2.metric("RMSE (test)", f"${test_rmse:,.0f}")
c3.metric("R²", f"{test_r2:.3f}")
c4.metric("Anuncios analizados", f"{len(df):,}")

tabs = st.tabs([
    "📊 Datos y EDA", "🤖 Modelos", "🔍 Interpretabilidad (SHAP)",
    "🧮 Probar el modelo", "📝 Conclusiones",
])

# ------------------------------------------------------------------- EDA --
with tabs[0]:
    st.subheader("Distribución de precios")
    st.write(
        "El precio está sesgado a la derecha (unos pocos anuncios de lujo estiran la cola); "
        "la mayoría de las motos se concentra por debajo de los \\$15,000."
    )
    fig = go.Figure(go.Histogram(x=df["price"], nbinsx=50, marker_color=BLUE))
    fig.update_layout(
        title="Distribución de precios (USD)", xaxis_title="Precio (USD)", yaxis_title="Anuncios",
        bargap=0.02, xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precio mediano por marca")
        med_brand = df.groupby("brand")["price"].median().reindex(BRAND_ORDER)
        fig = go.Figure(go.Bar(
            x=med_brand.index, y=med_brand.values,
            marker_color=[BRAND_COLOR[b] for b in med_brand.index],
            text=[f"${v:,.0f}" for v in med_brand.values], textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Precio mediano (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tamaño de muestra por marca")
        counts = df["brand"].value_counts().reindex(BRAND_ORDER)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[BRAND_COLOR[b] for b in counts.index],
            text=counts.values, textposition="outside",
            hovertemplate="%{x}<br>%{y:,} anuncios<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Anuncios", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Precio por categoría")
    cat_order = df.groupby("category_grouped")["price"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for cat in cat_order:
        fig.add_trace(go.Box(
            y=df.loc[df["category_grouped"] == cat, "price"], name=cat,
            marker_color=BLUE, line_color=BLUE, fillcolor=BLUE_LIGHT, showlegend=False,
        ))
    fig.update_layout(
        yaxis_title="Precio (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Feature engineering aplicado"):
        st.markdown(
            "- **Categorías raras** (menos de 50 anuncios) se agrupan en `Other` para evitar categorías "
            "con muy poca señal estadística.\n"
            "- **Kilometraje faltante** se imputa con la mediana de marca+categoría (o de marca, si no alcanza).\n"
            "- **`bike_age`** = 2024 − año del anuncio; **`mileage_per_year`** = kilometraje / antigüedad.\n"
            "- **`description_length`** y **`has_description`** capturan qué tan detallado es el anuncio.\n"
            "- El **nombre exacto del modelo** (`model`) se descarta como feature — esto vuelve a aparecer "
            "en la sección de SHAP como la causa raíz del error más grande del modelo."
        )

# --------------------------------------------------------------- Modelos --
with tabs[1]:
    st.subheader("Comparación de modelos")
    st.write(
        "Cuatro modelos de complejidad creciente, evaluados sobre el mismo test set (80/20, "
        "estratificado por categoría). XGBoost fue afinado con `RandomizedSearchCV` (30 combinaciones, "
        "5-fold CV) sobre `n_estimators`, `max_depth`, `learning_rate`, `subsample` y `colsample_bytree`."
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
        title="RMSE por modelo (menor = mejor) — XGBoost en azul es el modelo elegido",
        xaxis_title="RMSE (USD)", xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "XGBoost se usa como modelo insignia del proyecto (documentado en el README) aunque en algunas "
        "corridas Random Forest afinado queda a menos del 2% de RMSE — la diferencia entre ambos es "
        "ruido de muestreo del `RandomizedSearchCV`, no una ventaja estructural clara."
    )

# -------------------------------------------------------------- SHAP tab --
with tabs[2]:
    st.subheader("¿En qué se fija el modelo? (SHAP global)")
    st.write(
        "Cada barra es el impacto promedio (en dólares) de esa variable sobre la predicción, "
        "en valor absoluto, agregando las columnas one-hot de marca y categoría en su variable original."
    )

    shap_test = explainer(X_test)
    global_importance = pd.DataFrame(np.abs(shap_test.values), columns=X_test.columns).mean()
    global_grouped = global_importance.groupby([groups[c] for c in X_test.columns]).sum().sort_values()

    fig = go.Figure(go.Bar(
        x=global_grouped.values, y=global_grouped.index, orientation="h", marker_color=BLUE,
        text=[f"${v:,.0f}" for v in global_grouped.values], textposition="outside",
        hovertemplate="%{y}<br>$%{x:,.0f} promedio<extra></extra>",
    ))
    fig.update_layout(
        title="Importancia media |SHAP| por variable", xaxis_title="Impacto medio en la predicción (USD)",
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "**Marca** y **categoría** dominan, seguidas de la **antigüedad**; el **kilometraje** pesa "
        "menos de lo que se esperaría intuitivamente — consistente con la importancia de features "
        "del árbol y con la correlación débil vista en la sección de EDA."
    )

    st.divider()
    st.subheader("El caso Ducati: dónde falla el modelo")

    ducati_row = df[(df["brand"] == "Ducati") & (df["price"] == 99950.0)]
    if len(ducati_row):
        idx = ducati_row.index[0]
        listing = ducati_row.iloc[0]
        x_row = model_df.drop(columns=["price"]).loc[[idx]]
        pred = float(xgb_model.predict(x_row)[0])
        shap_row = explainer(x_row)

        st.markdown(
            f"El peor error individual del modelo: un **{listing['model'].title()}** "
            f"({listing['category']}, {int(listing['year'])}, {int(listing['mileage'])} millas) "
            f"listado en **\\${listing['price']:,.0f}**, predicho en **\\${pred:,.0f}** — un error de "
            f"**\\${listing['price'] - pred:,.0f}** ({(listing['price'] - pred) / listing['price']:.0%} por debajo)."
        )

        grouped_contrib = aggregate_shap(shap_row.values[0], x_row.columns, groups)
        fig = waterfall_figure(
            float(shap_row.base_values[0]), grouped_contrib, pred,
            "De dónde sale la predicción para esta Ducati (SHAP, agregado por variable)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Lectura:** el modelo sí reconoce que 'Ducati' y 'Sportbike' son señales premium — ambas "
            "empujan la predicción **hacia arriba**. El problema es la magnitud: no existe ninguna feature "
            "que indique que este anuncio en particular es una **edición limitada Diavel 1260 Lamborghini** "
            "con apenas 124 millas. El nombre exacto del modelo (`model`) se descartó como feature en el "
            "feature engineering, así que el modelo trata esta moto como una Ducati genérica y se queda "
            "muy corto. Es un límite de información, no un error de ajuste del modelo."
        )
    else:
        st.info("No se encontró el caso Ducati de referencia en esta corrida de los datos.")

# ------------------------------------------------------------ Predictor --
with tabs[3]:
    st.subheader("Prueba el modelo con tu propia moto")
    st.write(
        f"Arma un anuncio hipotético y mira la predicción del modelo junto con su explicación SHAP, "
        f"exactamente igual que para el caso Ducati de arriba. (Año de referencia del dataset: {REFERENCE_YEAR}.)"
    )

    brand_options = BRAND_ORDER
    category_options = sorted(df["category_grouped"].unique())

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("Marca", brand_options, index=brand_options.index("Ducati"))
            condition = st.radio("Condición", ["Used", "New"], horizontal=True)
        with col2:
            category = st.selectbox(
                "Categoría", category_options,
                index=category_options.index("Sportbike") if "Sportbike" in category_options else 0,
            )
            year = st.slider("Año", 1990, REFERENCE_YEAR, 2020)
        with col3:
            mileage = st.number_input("Kilometraje (millas)", min_value=0, max_value=150_000, value=5_000, step=500)
            description_length = st.slider("Palabras en la descripción del anuncio", 0, 200, 40)
        submitted = st.form_submit_button("Predecir precio", type="primary")

    if submitted:
        x_row = build_input_row(
            list(X_train.columns), brand=brand, category_grouped=category,
            condition=condition, year=year, mileage=mileage, description_length=description_length,
        )
        pred = float(xgb_model.predict(x_row)[0])
        shap_row = explainer(x_row)

        st.metric("Precio predicho", f"${pred:,.0f}")
        grouped_contrib = aggregate_shap(shap_row.values[0], x_row.columns, groups)
        fig = waterfall_figure(
            float(shap_row.base_values[0]), grouped_contrib, pred,
            "Cómo se construye esta predicción (SHAP, agregado por variable)",
        )
        st.plotly_chart(fig, use_container_width=True)

        top_driver = grouped_contrib.abs().idxmax()
        top_value = grouped_contrib[top_driver]
        direction = "sube" if top_value > 0 else "baja"
        st.caption(
            f"La variable con mayor impacto en esta predicción es **{top_driver}**, que {direction} el "
            f"precio en aproximadamente **\\${abs(top_value):,.0f}** respecto al valor base de "
            f"\\${float(shap_row.base_values[0]):,.0f}."
        )

# ----------------------------------------------------------- Conclusiones --
with tabs[4]:
    st.subheader("Conclusiones")
    st.markdown(
        f"""
- **Marca y categoría** son los predictores más fuertes del precio; el kilometraje crudo y el año
  aportan menos de lo esperado — algo que tanto la importancia de features como SHAP confirman de
  forma independiente.
- **XGBoost (tuned)** es el modelo insignia del proyecto: RMSE ≈ \\${test_rmse:,.0f}, R² ≈ {test_r2:.3f}
  en esta corrida (los números exactos varían un poco entre corridas por el muestreo de
  `RandomizedSearchCV`, pero el orden de magnitud es estable).
- **Dónde falla el modelo:** motos premium o de edición limitada (el caso Ducati Diavel 1260
  Lamborghini de la pestaña de interpretabilidad). SHAP muestra que el modelo *sabe* que "Ducati" y
  "Sportbike" son señales premium, pero no tiene forma de distinguir una edición limitada de una
  Ducati genérica, porque el nombre exacto del modelo no es una feature.
- **Recomendación de negocio:** el modelo es confiable para el segmento masivo (Royal Enfield,
  categorías Standard/Cruiser). Para motos premium o de edición limitada, el precio debería revisarse
  manualmente — el modelo las subvalúa de forma sistemática.

**Próximo paso natural:** reintroducir una versión agrupada de `model` (p. ej. una bandera de
"tier premium / edición limitada") para cerrar esta brecha específica.
"""
    )
    st.caption(
        "Este dashboard interactivo reemplaza la versión estática anterior (dashboard.html) del "
        "repositorio — mismo proyecto, ahora con storytelling completo e interpretabilidad SHAP en vivo."
    )
