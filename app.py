import streamlit as st
import numpy as np
import pandas as pd
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import plotly.express as px


# Streamlit setup

st.set_page_config(
    page_title="Fuzzy Grocery Recommendation",
    layout="wide"
)

st.title("🛒 Fuzzy Grocery Recommendation System")


# Load dataset

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# Sidebar – User Inputs

st.sidebar.header("User Preferences")

selected_products = st.sidebar.multiselect(
    "Select products",
    options=sorted(df["product"].unique()),
    default=[df["product"].unique()[0]]
)

u_price = st.sidebar.slider("Price importance", 1, 10, 5)
u_freshness = st.sidebar.slider("Freshness importance", 1, 10, 5)
u_distance = st.sidebar.slider("Distance importance", 1, 10, 5)
u_availability = st.sidebar.slider("Availability importance", 1, 10, 5)

st.sidebar.header("Your Location")

user_lat = st.sidebar.number_input("Latitude", value=46.9480)
user_lon = st.sidebar.number_input("Longitude", value=7.4474)

if len(selected_products) == 0:
    st.warning("Please select at least one product.")
    st.stop()


# Distance calculation (Haversine)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

store_coords = df[["store", "store_latitude", "store_longitude"]].drop_duplicates()

store_distance_km = {
    row.store: haversine(user_lat, user_lon, row.store_latitude, row.store_longitude)
    for _, row in store_coords.iterrows()
}

df_eval = df.copy()
df_eval["distance_km"] = df_eval["store"].map(store_distance_km)

MAX_DISTANCE_KM = 150
df_eval["store_distance"] = np.clip(
    (df_eval["distance_km"] / MAX_DISTANCE_KM) * 10, 0, 10
)

# Filter for MULTIPLE products
df_eval = df_eval[df_eval["product"].isin(selected_products)].copy()


# Fuzzy variables

store_freshness = ctrl.Antecedent(np.arange(0, 11, 1), "store_freshness")
store_distance = ctrl.Antecedent(np.arange(0, 11, 1), "store_distance")
product_price = ctrl.Antecedent(np.arange(0, 11, 1), "product_price")
availability = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "availability_score")

recommendation = ctrl.Consequent(np.arange(0, 26, 1), "recommendation")

# Membership functions
store_freshness["low"] = fuzz.trimf(store_freshness.universe, [0, 0, 4])
store_freshness["medium"] = fuzz.trimf(store_freshness.universe, [3, 5, 7])
store_freshness["high"] = fuzz.trimf(store_freshness.universe, [6, 10, 10])

store_distance["near"] = fuzz.trimf(store_distance.universe, [0, 0, 4])
store_distance["medium"] = fuzz.trimf(store_distance.universe, [3, 5, 7])
store_distance["far"] = fuzz.trimf(store_distance.universe, [6, 10, 10])

product_price["low"] = fuzz.trimf(product_price.universe, [0, 0, 4])
product_price["medium"] = fuzz.trimf(product_price.universe, [3, 5, 7])
product_price["high"] = fuzz.trimf(product_price.universe, [6, 10, 10])

availability["low"] = fuzz.trimf(availability.universe, [0, 0, 0.4])
availability["medium"] = fuzz.trimf(availability.universe, [0.3, 0.5, 0.7])
availability["high"] = fuzz.trimf(availability.universe, [0.6, 1, 1])

recommendation["low"] = fuzz.trimf(recommendation.universe, [0, 0, 10])
recommendation["medium"] = fuzz.trimf(recommendation.universe, [8, 13, 18])
recommendation["high"] = fuzz.trimf(recommendation.universe, [16, 25, 25])


# Fuzzy rules (with fallback)

rules = [
    ctrl.Rule(availability["high"] & store_freshness["high"], recommendation["high"]),
    ctrl.Rule(product_price["high"], recommendation["low"]),
    ctrl.Rule(store_distance["far"], recommendation["low"]),
    ctrl.Rule(product_price["low"] & store_distance["near"], recommendation["high"]),
    ctrl.Rule(store_freshness["medium"], recommendation["medium"]),
    ctrl.Rule(
        store_freshness["low"] | store_freshness["medium"] | store_freshness["high"],
        recommendation["medium"]
    )
]

recom_ctrl = ctrl.ControlSystem(rules)


# Compute recommendation scores

scores = []

for _, row in df_eval.iterrows():
    sim = ctrl.ControlSystemSimulation(recom_ctrl)

    sim.input["store_freshness"] = np.clip(row.store_freshness * (u_freshness / 10), 0, 10)
    sim.input["store_distance"] = np.clip(row.store_distance * (u_distance / 10), 0, 10)
    sim.input["product_price"] = np.clip(row.product_price * (u_price / 10), 0, 10)
    sim.input["availability_score"] = np.clip(row.availability_score * (u_availability / 10), 0, 1)

    sim.compute()
    scores.append(sim.output.get("recommendation", np.nan))

df_eval["recommendation_score"] = np.round(scores, 2)


# Top-5 Recommendations PER PRODUCT

st.subheader("🏆 Top 5 Store Recommendations per Product")

for product in selected_products:
    st.markdown(f"### 📦 {product}")

    top5 = (
        df_eval[df_eval["product"] == product]
        .sort_values("recommendation_score", ascending=False)
        .head(5)
    )

    st.dataframe(
        top5[
            [
                "store",
                "recommendation_score",
                "store_distance",
                "store_freshness",
                "product_price",
                "availability_score",
            ]
        ],
        use_container_width=True
    )

    fig = px.bar(
        top5,
        x="store",
        y="recommendation_score",
        color="store",
        title=f"Top 5 Stores for {product}"
    )

    st.plotly_chart(fig, use_container_width=True)


# Overall ranking across selected products

st.subheader("⭐ Overall Best Stores (Across Selected Products)")

overall_top = (
    df_eval
    .groupby("store")["recommendation_score"]
    .mean()
    .reset_index()
    .sort_values("recommendation_score", ascending=False)
    .head(5)
)

st.dataframe(overall_top, use_container_width=True)
