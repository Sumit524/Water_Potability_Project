import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

warnings.filterwarnings("ignore")

# ── Always resolve paths relative to this file ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_PATH       = os.path.join(BASE_DIR, "data",   "kaggle_water_quality.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMPUTER_PATH    = os.path.join(BASE_DIR, "models", "imputer.pkl")
COMPARISON_PATH = os.path.join(BASE_DIR, "models", "comparison.json")
CM_PATH         = os.path.join(BASE_DIR, "models", "confusion_matrix.json")
FI_PATH         = os.path.join(BASE_DIR, "models", "feature_importance.json")

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ── Cache ─────────────────────────────────────────────────────────
_cache = {}

def get_model():
    if "model" not in _cache:
        _cache["model"] = joblib.load(MODEL_PATH)
    return _cache["model"]

def get_pipeline():
    if "pipeline" not in _cache:
        _cache["pipeline"] = (joblib.load(IMPUTER_PATH), joblib.load(SCALER_PATH))
    return _cache["pipeline"]

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def file_missing(path, name="file"):
    return jsonify({"error": f"{name} not found. Run train.py first."}), 404


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dataset")
def api_dataset():
    try:
        if not os.path.exists(DATA_PATH):
            return file_missing(DATA_PATH, "Dataset CSV")
        df      = pd.read_csv(DATA_PATH)
        total   = len(df)
        potable = int((df["Potability"] == 1).sum())
        not_pot = int((df["Potability"] == 0).sum())

        feature_stats = {}
        for col in FEATURES:
            feature_stats[col] = {
                "mean":    round(float(df[col].mean()), 2),
                "std":     round(float(df[col].std()),  2),
                "min":     round(float(df[col].min()),  2),
                "max":     round(float(df[col].max()),  2),
                "missing": int(df[col].isnull().sum()),
            }
        return jsonify({
            "total":           total,
            "potable":         potable,
            "not_potable":     not_pot,
            "missing_cells":   int(df.isnull().sum().sum()),
            "potable_pct":     round(potable / total * 100, 1),
            "not_potable_pct": round(not_pot / total * 100, 1),
            "feature_stats":   feature_stats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparison")
def api_comparison():
    try:
        if not os.path.exists(COMPARISON_PATH):
            return file_missing(COMPARISON_PATH, "comparison.json")
        return jsonify(read_json(COMPARISON_PATH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model")
def api_model():
    try:
        if not os.path.exists(COMPARISON_PATH):
            return file_missing(COMPARISON_PATH, "comparison.json")
        records = read_json(COMPARISON_PATH)
        best    = max(records, key=lambda r: r["roc_auc"])
        cm      = read_json(CM_PATH) if os.path.exists(CM_PATH) else [[0,0],[0,0]]
        return jsonify({
            "model_name":       best["model_name"],
            "accuracy":         best["accuracy"],
            "precision":        best["precision"],
            "recall":           best["recall"],
            "f1":               best["f1"],
            "roc_auc":          best["roc_auc"],
            "confusion_matrix": cm,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature_importance")
def api_feature_importance():
    try:
        if not os.path.exists(FI_PATH):
            return file_missing(FI_PATH, "feature_importance.json")
        return jsonify(read_json(FI_PATH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ph_distribution")
def api_ph_distribution():
    try:
        if not os.path.exists(DATA_PATH):
            return file_missing(DATA_PATH, "Dataset CSV")
        df     = pd.read_csv(DATA_PATH).dropna(subset=["ph"])
        bins   = list(range(2, 14))
        labels, potable_counts, not_potable_counts = [], [], []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            sub    = df[(df["ph"] >= lo) & (df["ph"] < hi)]
            labels.append(f"{lo}-{hi}")
            potable_counts.append(int((sub["Potability"] == 1).sum()))
            not_potable_counts.append(int((sub["Potability"] == 0).sum()))
        return jsonify({
            "labels":      labels,
            "potable":     potable_counts,
            "not_potable": not_potable_counts,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if not os.path.exists(MODEL_PATH):
            return file_missing(MODEL_PATH, "best_model.pkl")
        data    = request.get_json(force=True)
        sample  = {f: float(data[f]) for f in FEATURES}
        imputer, scaler = get_pipeline()
        model   = get_model()
        df      = pd.DataFrame([sample])[FEATURES]
        df_imp  = imputer.transform(df)
        df_sc   = scaler.transform(df_imp)
        pred    = int(model.predict(df_sc)[0])
        proba   = float(model.predict_proba(df_sc)[0][1])
        return jsonify({"prediction": pred, "probability": round(proba, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[AquaML] Starting server...")
    print(f"[AquaML] BASE_DIR  : {BASE_DIR}")
    print(f"[AquaML] DATA_PATH : {DATA_PATH}  exists={os.path.exists(DATA_PATH)}")
    print(f"[AquaML] MODEL_PATH: {MODEL_PATH}  exists={os.path.exists(MODEL_PATH)}")
    print(f"[AquaML] Dashboard : http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)