import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from xgboost import data
from feature_engineering import engineer_features

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
THRESHOLD_PATH  = os.path.join(BASE_DIR, "models", "threshold.json")


# The predictor form only collects these 9 raw sensor values.
# The 5 engineered features are derived server-side before prediction.
RAW_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# Full feature list fed into the model (raw + engineered)
ALL_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    "ph_category", "hardness_solids", "chloramine_ph",
    "conductivity_ratio", "turbidity_organic"
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

def get_threshold():
    """Load optimal decision threshold saved by train.py (default 0.5)."""
    if "threshold" not in _cache:
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                _cache["threshold"] = json.load(f).get("threshold", 0.5)
        else:
            _cache["threshold"] = 0.5
    return _cache["threshold"]

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def file_missing(name="file"):
    return jsonify({"error": f"{name} not found. Run train.py first."}), 404



# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── BUG FIX 3: api/dataset only computes stats for RAW columns ────
# Engineered columns don't exist in the raw CSV → computing them
# here caused a KeyError that crashed the entire dashboard.
@app.route("/api/dataset")
def api_dataset():
    try:
        if not os.path.exists(DATA_PATH):
            return file_missing("Dataset CSV")
        df      = pd.read_csv(DATA_PATH)
        total   = len(df)
        potable = int((df["Potability"] == 1).sum())
        not_pot = int((df["Potability"] == 0).sum())

        # Only compute stats for raw columns that exist in the CSV
        feature_stats = {}
        for col in RAW_FEATURES:
            if col in df.columns:
                feature_stats[col] = {
                    "mean":    round(float(df[col].mean(skipna=True)), 2),
                    "std":     round(float(df[col].std(skipna=True)),  2),
                    "min":     round(float(df[col].min(skipna=True)),  2),
                    "max":     round(float(df[col].max(skipna=True)),  2),
                    "missing": int(df[col].isnull().sum()),
                }
        return jsonify({
            "total":           total,
            "potable":         potable,
            "not_potable":     not_pot,
            "missing_cells":   int(df[RAW_FEATURES].isnull().sum().sum()),
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
            return file_missing("comparison.json")
        return jsonify(read_json(COMPARISON_PATH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model")
def api_model():
    try:
        if not os.path.exists(COMPARISON_PATH):
            return file_missing("comparison.json")
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
            "threshold":        get_threshold(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature_importance")
def api_feature_importance():
    try:
        if not os.path.exists(FI_PATH):
            return file_missing("feature_importance.json")
        return jsonify(read_json(FI_PATH))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ph_distribution")
def api_ph_distribution():
    try:
        if not os.path.exists(DATA_PATH):
            return file_missing("Dataset CSV")
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


# ── BUG FIX 4: Predict endpoint now accepts only 9 raw inputs ─────
# Engineers the 5 derived features here before running the pipeline.
# This also fixes the double-transform bug (engineered features were
# previously passed through the imputer+scaler even though they
# were already computed post-imputation during training).
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        # Check model file exists
        if not os.path.exists(MODEL_PATH):
            return file_missing("best_model.pkl")

        data = request.get_json(force=True)

        # ── Step 1: Accept missing values instead of hard rejecting ──
        MAX_ALLOWED_MISSING = 6
        
        missing_fields = [f for f in RAW_FEATURES if f not in data or data[f] is None or data[f] == ""]
        #ph value ,solid and turbidity are required but we can allow some missing values for the rest of the features. so if above three are missing we will reject the request immediately. but if some of the other features are missing we will fill them with the imputer and warn the user that the prediction may be less reliable.
        required_fields = ["ph", "Solids", "Turbidity"]
        # print("Incoming data:", data)
        if any(f not in data or data[f] is None or data[f] == "" for f in required_fields):
            return jsonify({
                "error": "Required fields (ph, Solids, Turbidity) are missing or empty."
            }), 400

        if len(missing_fields) > MAX_ALLOWED_MISSING:
            return jsonify({
                "error": f"Too many missing sensor readings ({len(missing_fields)}). Maximum allowed is {MAX_ALLOWED_MISSING}.",
                "missing_fields": missing_fields
            }), 400

        # ── Step 2: Build raw DataFrame, missing values become NaN ──
        raw = {}
        for f in RAW_FEATURES:
            val = data.get(f)
            if val is None or val == "":
                raw[f] = np.nan          # imputer will fill this
            else:
                try:
                    raw[f] = float(val)
                except (ValueError, TypeError):
                    return jsonify({
                        "error": f"Invalid value for field '{f}': {val}. Expected a number."
                    }), 400

        df_raw = pd.DataFrame([raw])

        # ── Step 3: Engineer 5 derived features → total 14 features ──
        df_full = engineer_features(df_raw)[ALL_FEATURES]

        # ── Step 4: Imputer fills NaN (raw + derived), scaler normalizes ──
        imputer, scaler = get_pipeline()
        df_imp = imputer.transform(df_full)
        df_imp_df = pd.DataFrame(df_imp, columns=ALL_FEATURES)
        df_sc  = scaler.transform(df_imp)
        filled_values = {}

        for f in missing_fields:
          filled_values[f] = round(float(df_imp_df.iloc[0][f]), 4)

        # ── Step 5: Predict ──
        model     = get_model()
        proba     = float(model.predict_proba(df_sc)[0][1]) 
        threshold = get_threshold()
        pred      = int(proba >= threshold)

        # ── Step 6: Warn if prediction may be less reliable ──
        warning = None 
        if len(missing_fields) > 0:
            warning = f"{len(missing_fields)} sensor(s) were missing and filled automatically: {missing_fields}"

        return jsonify({
            "prediction":     pred,
            "probability":    round(proba, 4),
            "missing_fields": missing_fields,
            "filled_values":  filled_values,  
            "warning":        warning
        })

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