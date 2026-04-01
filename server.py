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

import torch
import tiktoken
import sys

sys.path.insert(0, os.path.dirname(__file__))  # ensure previous_chapters.py is importable
from previous_chapters import GPTModel

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Load primary GPT-2 model ───────────────────────────────────
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}
MAX_LENGTH = 473
PAD_TOKEN_ID = 50256
LABEL_MAP = {0: "Alay", 1: "EYD", 2: "Jaksel"}

GPT2_PATH = os.path.join(os.path.dirname(__file__), "..", "scbi_classifier.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_model = None
tokenizer = None
USE_GPT2 = False

try:
    gpt2_model = GPTModel(BASE_CONFIG)
    gpt2_model.out_head = torch.nn.Linear(768, 3)
    state_dict = torch.load(os.path.abspath(GPT2_PATH), map_location=device, weights_only=True)
    gpt2_model.load_state_dict(state_dict)
    gpt2_model.to(device)
    gpt2_model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    USE_GPT2 = True
    print("GPT-2 model loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load GPT-2 model: {e}")
    print("Falling back to sklearn pickle model.")

# Sklearn fallback has been removed.


# ── Load training metrics ───────────────────────────────────
METRICS_PATH = os.path.join(os.path.dirname(__file__), "training_metrics.json")
try:
    with open(METRICS_PATH, "r") as f:
        training_metrics = json.load(f)
    print("Training metrics loaded.")
except Exception as e:
    print(f"Could not load training metrics: {e}")
    training_metrics = {}


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

    if USE_GPT2 and gpt2_model is not None:
        # --- GPT-2 inference ---
        input_ids = tokenizer.encode(text)
        # Truncate to max_length
        input_ids = input_ids[:MAX_LENGTH]
        # Pad to max_length
        input_ids += [PAD_TOKEN_ID] * (MAX_LENGTH - len(input_ids))
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = gpt2_model(input_tensor)[:, -1, :]  # last token logits: (1, 3)

        # Softmax for confidence
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        predicted_idx = int(torch.argmax(logits, dim=-1).item())
        prediction = LABEL_MAP[predicted_idx].lower()
        classes_order = [LABEL_MAP[i].lower() for i in range(3)]
        confidence = {cls: round(float(p), 4) for cls, p in zip(classes_order, probs)}

        return jsonify({
            "label": prediction,
            "confidence": confidence,
            "model": "GPT-2 (fine-tuned SCBI)",
        })

    else:
        return jsonify({"error": "GPT-2 model failed to load. No fallback available."}), 500

@app.route("/metrics")
def metrics():
    return jsonify(training_metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
