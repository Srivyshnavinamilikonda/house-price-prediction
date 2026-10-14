import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_assets():
    model = joblib.load("models/lasso_house_price.joblib")
    with open("models/defaults.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_assets()
defaults = meta["defaults"]

st.title("🏠 House Price Predictor")
st.write(
    "Lasso regression trained on the Ames, Iowa housing dataset "
    "(test R² = 0.92). Enter a few details to get an estimate. "
    "All other features are set to typical values."
)

col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall quality (1-10)", 1, 10, int(defaults["OverallQual"]))
    gr_liv_area = st.number_input(
        "Above-ground living area (sq ft)", 300, 6000, int(defaults["GrLivArea"]), step=50
    )
    year_built = st.number_input("Year built", 1870, 2010, int(defaults["YearBuilt"]))
    neighborhood = st.selectbox(
        "Neighborhood",
        meta["neighborhoods"],
        index=meta["neighborhoods"].index(defaults["Neighborhood"]),
    )
with col2:
    garage_cars = st.slider("Garage capacity (cars)", 0, 4, int(defaults["GarageCars"]))
    total_bsmt = st.number_input(
        "Total basement area (sq ft)", 0, 4000, int(defaults["TotalBsmtSF"]), step=50
    )
    full_bath = st.slider("Full bathrooms", 0, 4, int(defaults["FullBath"]))

if st.button("Predict price", type="primary"):
    row = dict(defaults)
    row.update(
        {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "YearBuilt": year_built,
            "Neighborhood": neighborhood,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt,
            "FullBath": full_bath,
        }
    )
    X_new = pd.DataFrame([row])
    X_new["MSSubClass"] = X_new["MSSubClass"].astype(str)

    price = float(np.expm1(model.predict(X_new))[0])
    st.success(f"Estimated price: ${price:,.0f}")
    st.caption(
        "On the test set the model's average error was about $14,000. "
        "This is a demo, not a valuation."
    )