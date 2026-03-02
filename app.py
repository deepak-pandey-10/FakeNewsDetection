import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# ── Session tracking ──────────────────────────────────────────────
session_stats = {
    "total_predictions": 0,
    "fake_count": 0,
    "real_count": 0,
    "confidence_sum": 0.0,
    "history": [],  # last N predictions
}
MAX_HISTORY = 50


# ── Extract model metadata once at startup ────────────────────────
def _extract_model_info():
    """Pull stats from the Logistic Regression + TF-IDF artifacts."""
    feature_names = vectorizer.get_feature_names_out()
    n_features = len(feature_names)
    classes = model.classes_.tolist()

    # Top features for each class from LR coefficients
    coef = model.coef_[0]  # shape (n_features,) for binary
    top_k = 20

    # Top FAKE indicators (most negative coefficients → class 0 = Fake)
    fake_indices = np.argsort(coef)[:top_k]
    fake_features = [
        {"word": feature_names[i], "weight": round(float(coef[i]), 4)}
        for i in fake_indices
    ]

    # Top REAL indicators (most positive coefficients → class 1 = Real)
    real_indices = np.argsort(coef)[-top_k:][::-1]
    real_features = [
        {"word": feature_names[i], "weight": round(float(coef[i]), 4)}
        for i in real_indices
    ]

    # Model parameters
    params = model.get_params()

    return {
        "model_type": type(model).__name__,
        "regularization": params.get("C", "N/A"),
        "solver": params.get("solver", "N/A"),
        "max_iter": params.get("max_iter", "N/A"),
        "n_features": n_features,
        "classes": classes,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "top_fake_features": fake_features,
        "top_real_features": real_features,
    }


MODEL_INFO = _extract_model_info()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Vectorize and predict
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    # Determine label and confidence
    label = "Real" if prediction == 1 else "Fake"
    confidence = float(max(probabilities)) * 100

    # ── Per-word contribution analysis ─────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    nonzero = text_vectorized.nonzero()
    word_contributions = []
    for idx in nonzero[1]:
        word_contributions.append({
            "word": feature_names[idx],
            "contribution": round(float(coef[idx] * text_vectorized[0, idx]), 4),
        })
    word_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top_words = word_contributions[:10]

    # ── Update session stats ──────────────────────────────────────
    session_stats["total_predictions"] += 1
    session_stats["confidence_sum"] += confidence
    if label == "Fake":
        session_stats["fake_count"] += 1
    else:
        session_stats["real_count"] += 1

    snippet = text[:80] + ("…" if len(text) > 80 else "")
    session_stats["history"].append({
        "text": snippet,
        "prediction": label,
        "confidence": round(confidence, 2),
    })
    if len(session_stats["history"]) > MAX_HISTORY:
        session_stats["history"] = session_stats["history"][-MAX_HISTORY:]

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2),
        "top_words": top_words,
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return full model statistics and session analytics."""
    total = session_stats["total_predictions"]
    avg_confidence = (
        round(session_stats["confidence_sum"] / total, 2) if total > 0 else 0
    )

    return jsonify({
        "model": MODEL_INFO,
        "session": {
            "total_predictions": total,
            "fake_count": session_stats["fake_count"],
            "real_count": session_stats["real_count"],
            "avg_confidence": avg_confidence,
            "history": session_stats["history"][-20:],  # last 20
        },
    })


from flask import Flask

app = Flask(__name__)

# your routes here

if __name__ == "__main__":
    app.run()
