
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "california_house_price_model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.json"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.write("Enter neighborhood details to estimate the house price.")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

FIELDS = {
    "Median income ($10,000s)": ("MedInc", 4.5, 0.01),
    "Median house age (years)": ("HouseAge", 25.0, 1.0),
    "Average rooms": ("AveRooms", 5.5, 0.1),
    "Average bedrooms": ("AveBedrms", 1.05, 0.01),
    "Population": ("Population", 1500.0, 1.0),
    "Average household size": ("AveOccup", 3.0, 0.1),
    "Latitude": ("Latitude", 34.05, 0.01),
    "Longitude": ("Longitude", -118.25, 0.01),
}

CITIES = {
    "Los Angeles": (34.05, -118.25),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
}

try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"Could not load the model or metadata: {e}")
    st.info(
        "Check that california_house_price_model.pkl "
        "and model_metadata.json exist in your GitHub repository."
    )
    st.stop()

city = st.selectbox("Choose a California city", list(CITIES))
latitude, longitude = CITIES[city]

values = {}
cols = st.columns(2)

for i, (label, (name, default, step)) in enumerate(FIELDS.items()):
    with cols[i % 2]:
        if name == "Latitude":
            default = latitude
        elif name == "Longitude":
            default = longitude

        values[name] = st.number_input(
            label,
            value=float(default),
            step=float(step),
            format="%.4f",
        )

if st.button("Predict House Price", type="primary"):
    try:
        features = metadata["features"]
        frame = pd.DataFrame([values])[features]
        price = max(float(model.predict(frame)[0]), 0.0)

        rmse = float(metadata["cv_mean_rmse"])

        st.success(f"Estimated house price: ${price * 100000:,.2f}")
        st.caption(
            "The model's target scale determines the price conversion. "
            "Verify whether the model predicts units of $100,000."
        )
        st.write(f"Model: {metadata['best_algorithm']}")
        st.write(f"Typical prediction error (model units): {rmse:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
