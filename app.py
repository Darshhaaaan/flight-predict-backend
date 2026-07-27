from flask import Flask, request, jsonify
import json
import os
import numpy as np
import pandas as pd
from flask_cors import CORS

frontend_url = os.environ.get("FRONTEND_URL")

app = Flask(__name__)

CORS(
    app,
    origins=[
        "http://localhost:3000",
        frontend_url
    ]
)

with open("model.json", "r") as file:
    model = json.load(file)

weights = np.array(model["weight"], dtype=float)
bias = float(model["bias"])

duration_mean = float(model["duration_mean"])
duration_std = float(model["duration_std"])

days_left_mean = float(model["days_left_mean"])
days_left_std = float(model["days_left_std"])

order = model["order"]


@app.route("/")
def home():
    return jsonify({
        "message": "Flight Price Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received"
            }), 400

        required_fields = [
            "airline",
            "source_city",
            "departure_time",
            "stops",
            "arrival_time",
            "destination_city",
            "class",
            "duration",
            "days_left"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] in [None, ""]
        ]

        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing_fields)}"
            }), 400

        if data["source_city"] == data["destination_city"]:
            return jsonify({
                "error": "Source city and destination city cannot be the same"
            }), 400

        duration = float(data["duration"])
        days_left = float(data["days_left"])

        if duration < 0 or duration > 50:
            return jsonify({
                "error": "Duration must be between 0 and 50 hours"
            }), 400

        if days_left < 0 or days_left > 49:
            return jsonify({
                "error": "Days left must be between 0 and 49"
            }), 400

        input_df = pd.DataFrame([data])

        input_df["class"] = input_df["class"].map({
            "Economy": 0,
            "Business": 1
        })

        input_df["stops"] = input_df["stops"].map({
            "0": 0,
            "1": 1,
            "2": 2
        })

        if input_df["class"].isna().any():
            return jsonify({
                "error": "Invalid class value"
            }), 400

        if input_df["stops"].isna().any():
            return jsonify({
                "error": "Invalid stops value"
            }), 400

        input_df["duration"] = (
            input_df["duration"] - duration_mean
        ) / duration_std

        input_df["days_left"] = (
            input_df["days_left"] - days_left_mean
        ) / days_left_std

        categorical_columns = [
            "airline",
            "source_city",
            "departure_time",
            "arrival_time",
            "destination_city"
        ]

        input_df = pd.get_dummies(
            input_df,
            columns=categorical_columns
        )

        input_df = input_df.reindex(
            columns=order,
            fill_value=0
        )

        X = input_df.to_numpy(dtype=float)

        if X.shape[1] != len(weights):
            return jsonify({
                "error": "Feature count does not match model weights"
            }), 400

        if np.isnan(X).any() or np.isinf(X).any():
            return jsonify({
                "error": "Invalid numerical value found in processed input"
            }), 400

        if np.isnan(weights).any() or np.isinf(weights).any():
            return jsonify({
                "error": "Model weights contain invalid values"
            }), 500

        if not np.isfinite(bias):
            return jsonify({
                "error": "Model bias contains an invalid value"
            }), 500

        prediction = np.dot(X, weights) + bias
        prediction = float(prediction[0])

        if not np.isfinite(prediction):
            return jsonify({
                "error": "Model returned an invalid prediction"
            }), 500

        return jsonify({
            "predicted_price": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)