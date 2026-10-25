"""
Flask web app for the California House Price Prediction model (Assignment 3).

Routes
------
GET  /              HTML form
POST /               form submission -> same page with the predicted price
POST /api/predict    JSON in, JSON out  ({"MedInc": 4.5, ...})
GET  /api/info       model information (algorithm, cross-validation scores, feature ranges)
GET  /health         health check used by the cloud platform
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "california_house_price_model.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "model_metadata.json")

app = Flask(__name__)

# ---- Load the trained model and its metadata once, when the server starts ----
model = None
metadata = None
load_error = None
try:
    model = joblib.load(MODEL_FILE)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
except Exception as error:  # the page explains what is missing instead of crashing
    load_error = f"{type(error).__name__}: {error}"

# Field definitions: label, help text, the value shown first, and hard limits that reject nonsense input
FIELDS = [
    {"name": "MedInc", "label": "Median income", "unit": "in $10,000s (3.5 = $35,000)",
     "default": 4.5, "step": "0.01", "hard": (0.0, 100.0)},
    {"name": "HouseAge", "label": "Median house age", "unit": "years",
     "default": 25, "step": "1", "hard": (0.0, 150.0)},
    {"name": "AveRooms", "label": "Average rooms per household", "unit": "rooms",
     "default": 5.5, "step": "0.01", "hard": (0.1, 500.0)},
    {"name": "AveBedrms", "label": "Average bedrooms per household", "unit": "bedrooms",
     "default": 1.05, "step": "0.01", "hard": (0.1, 200.0)},
    {"name": "Population", "label": "Population of the neighbourhood", "unit": "people",
     "default": 1500, "step": "1", "hard": (1.0, 1_000_000.0)},
    {"name": "AveOccup", "label": "Average household size", "unit": "people per household",
     "default": 3.0, "step": "0.01", "hard": (0.1, 2000.0)},
    {"name": "Latitude", "label": "Latitude", "unit": "degrees north",
     "default": 34.05, "step": "0.01", "hard": (32.0, 43.0)},
    {"name": "Longitude", "label": "Longitude", "unit": "degrees east (negative in California)",
     "default": -118.25, "step": "0.01", "hard": (-125.0, -114.0)},
]

CITIES = {
    "Los Angeles": (34.05, -118.25),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
}


def parse_and_validate(values):
    """Return (clean_values, errors, warnings) for a dict of raw input values."""
    clean, errors, warnings = {}, {}, []
    for field in FIELDS:
        name = field["name"]
        raw = values.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = "Enter a value."
            continue
        try:
            number = float(str(raw).strip())
        except ValueError:
            errors[name] = "Enter a number."
            continue
        low, high = field["hard"]
        if number != number or not (low <= number <= high):
            errors[name] = f"Enter a value between {low:g} and {high:g}."
            continue
        clean[name] = number
        # Soft check: a value outside the range seen in training makes the prediction less reliable
        if metadata:
            seen = metadata["feature_ranges"].get(name)
            if seen and not (seen["min"] <= number <= seen["max"]):
                warnings.append(f"{field['label']} ({number:g}) is outside the range the model was "
                                f"trained on ({seen['min']:g} to {seen['max']:g}).")
    return clean, errors, warnings


def predict_price(clean):
    features = metadata["features"]
    frame = pd.DataFrame([clean])[features]          # same columns, same order as in training
    price = float(model.predict(frame)[0])
    return max(price, 0.0)                           # a price cannot be negative


def result_payload(price):
    rmse = metadata["cv_mean_rmse"]
    return {
        "predicted_price": round(price, 2),
        "typical_error": round(rmse, 2),
        "likely_range": [round(max(price - rmse, 0.0), 2), round(price + rmse, 2)],
        "algorithm": metadata["best_algorithm"],
    }


@app.route("/", methods=["GET", "POST"])
def home():
    values = {f["name"]: f["default"] for f in FIELDS}
    errors, warnings, result = {}, [], None

    if request.method == "POST":
        values = {f["name"]: request.form.get(f["name"], "") for f in FIELDS}
        if load_error:
            errors["_model"] = "The model is not available on the server."
        else:
            clean, errors, warnings = parse_and_validate(values)
            if not errors:
                result = result_payload(predict_price(clean))

    return render_template("index.html", fields=FIELDS, cities=CITIES, values=values, errors=errors,
                           warnings=warnings, result=result, metadata=metadata, load_error=load_error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if load_error:
        return jsonify({"error": "Model not available", "detail": load_error}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with the 8 input values."}), 400
    clean, errors, warnings = parse_and_validate(payload)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400
    response = result_payload(predict_price(clean))
    response["warnings"] = warnings
    return jsonify(response)


@app.route("/api/info")
def api_info():
    if load_error:
        return jsonify({"error": "Model not available", "detail": load_error}), 503
    return jsonify(metadata)


@app.route("/health")
def health():
    if load_error:
        return jsonify({"status": "error", "detail": load_error}), 503
    return jsonify({"status": "ok", "algorithm": metadata["best_algorithm"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
