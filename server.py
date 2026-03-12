"""
SCBI Classifier Web Server
Serves the sociolinguistics classification model via Flask.
"""
import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Load model ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sociolinguistics_model.pkl")

# Fallback: check current directory too
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "sociolinguistics_model.pkl")

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
vectorizer = model_data["vectorizer"]
classes = model_data["classes"]
model_name = model_data.get("best_model_name", "Unknown")
model_accuracy = model_data.get("accuracy", 0)

print(f"Model loaded: {model_name} (accuracy: {model_accuracy:.4f})")
print(f"Classes: {classes}")

# ── Load training metrics ───────────────────────────────────
METRICS_PATH = os.path.join(os.path.dirname(__file__), "training_metrics.json")
with open(METRICS_PATH, "r") as f:
    training_metrics = json.load(f)

print("Training metrics loaded.")


# ── Routes ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Vectorize and predict
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    # Get decision function scores for confidence
    try:
        decision = model.decision_function(vec)[0]
        # Convert decision scores to pseudo-probabilities via softmax
        exp_scores = np.exp(decision - np.max(decision))
        probabilities = exp_scores / exp_scores.sum()
        confidence = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    except Exception:
        confidence = {cls: (1.0 if cls == prediction else 0.0) for cls in classes}

    return jsonify({
        "label": prediction,
        "confidence": confidence,
        "model": model_name,
    })


@app.route("/metrics")
def metrics():
    return jsonify(training_metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
