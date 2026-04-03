import os
import joblib
import warnings
import numpy as np
import pandas as pd
import json

from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier

from sklearn.model_selection  import StratifiedKFold, cross_validate
from sklearn.inspection       import permutation_importance
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────

MODEL_DIR       = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
CV_FOLDS        = 5
RANDOM_STATE    = 42

# Models that need scaling before training
NEEDS_SCALING = {"SVM", "Logistic Regression"}

# ── Model Registry ────────────────────────────────────────────────

MODELS = {
   "XGBoost": XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,          # adjust if class imbalance is severe
    eval_metric="logloss",
    random_state=RANDOM_STATE
),
"LightGBM": LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,               
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
),
"Random Forest": RandomForestClassifier(
    n_estimators=500,           
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "SVM": SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=RANDOM_STATE
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
}

# ── Cross Validation ──────────────────────────────────────────────

def cross_validate_all(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  CROSS VALIDATION  ({}-Fold Stratified)".format(CV_FOLDS))
    print("=" * 60)

    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in MODELS.items():
        if name in NEEDS_SCALING:
            scaler   = StandardScaler()
            X_input  = scaler.fit_transform(X_train)
        else:
            X_input  = X_train

        scores = cross_validate(
            model, X_input, y_train,
            cv=skf, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1
        )
        results.append({
            "Model"      : name,
            "CV Accuracy": round(scores["test_accuracy"].mean(), 4),
            "CV F1"      : round(scores["test_f1"].mean(),       4),
            "CV ROC-AUC" : round(scores["test_roc_auc"].mean(),  4),
            "Std (AUC)"  : round(scores["test_roc_auc"].std(),   4),
        })
        print(f"  {name:<22}  AUC={results[-1]['CV ROC-AUC']:.4f}  "
              f"Acc={results[-1]['CV Accuracy']:.4f}  F1={results[-1]['CV F1']:.4f}")

    return pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False)

# ── Train All & Evaluate ──────────────────────────────────────────

def train_and_evaluate(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: np.ndarray,   y_test: np.ndarray
):
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    # ── Preserve feature names BEFORE any numpy conversion ────────
    feature_names = list(X_train.columns)

    results        = []
    trained_models = {}

    # One shared scaler for models that need it
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    for name, model in MODELS.items():
        if name in NEEDS_SCALING:
            model.fit(X_train_scaled, y_train)
            y_pred  = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall"   : round(recall_score(y_test, y_pred),    4),
            "F1 Score" : round(f1_score(y_test, y_pred),        4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),  4),
        })
        trained_models[name] = model
        print(f"  {name:<22}  Acc={results[-1]['Accuracy']:.4f}  "
              f"F1={results[-1]['F1 Score']:.4f}  AUC={results[-1]['ROC-AUC']:.4f}")

    comparison_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

    # ── Save comparison.json ──────────────────────────────────────
    records = pd.DataFrame(results).to_dict(orient="records")
    for r in records:
        r["model_name"] = r.pop("Model")
        r["accuracy"]   = r.pop("Accuracy")
        r["precision"]  = r.pop("Precision")
        r["recall"]     = r.pop("Recall")
        r["f1"]         = r.pop("F1 Score")
        r["roc_auc"]    = r.pop("ROC-AUC")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "comparison.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"[Train] Saved → models/comparison.json")

    # ── Save feature names (used by detailed_report) ──────────────
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"[Train] Saved → models/feature_names.json")

    best_name  = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    return comparison_df, best_model, best_name

# ── Detailed Report ───────────────────────────────────────────────

def detailed_report(model, X_test: pd.DataFrame, y_test: np.ndarray, name: str):
    """
    Saves confusion_matrix.json and feature_importance.json.
    X_test is always a DataFrame here — feature names are safe to read.
    """
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print(f"  BEST MODEL : {name}")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"]))

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Save confusion matrix ─────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)
    print(f"[Train] Saved → models/confusion_matrix.json")

    # ── Save feature importance ───────────────────────────────────
    # Load feature names saved earlier (always a list of strings)
    fn_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(fn_path) as f:
        feature_names = json.load(f)

    if hasattr(model, "feature_importances_"):
        # Tree-based models: use built-in importance (fast, reliable)
        # Measures average impurity decrease across all trees per feature
        importances = model.feature_importances_.tolist()
        method = "tree_impurity"
        print(f"[Train] Using native feature_importances_ ({name})")

    else:
        # SVM / Logistic Regression: use permutation importance
        # Measures how much accuracy drops when each feature is randomly shuffled
        # Works for ANY model — model-agnostic
        print(f"[Train] {name} has no feature_importances_ → using permutation importance...")
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=15,          # shuffle 15 times, take mean → stable result
            random_state=RANDOM_STATE,
            scoring="roc_auc"
        )
        # Use absolute values (some may be slightly negative = feature adds noise)
        importances = np.abs(perm.importances_mean).tolist()
        method = "permutation"
        print(f"[Train] Permutation importance done.")

    if len(importances) == len(feature_names):
        fi = {
            "features":    feature_names,
            "importances": importances,
            "method":      method      # stored so dashboard can show the right label
        }
        with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)
        print(f"[Train] Saved → models/feature_importance.json")
    else:
        print(f"[Train] ⚠ Mismatch: {len(importances)} importances vs "
              f"{len(feature_names)} names — skipping")

# ── Save Best Model ───────────────────────────────────────────────

def save_best_model(model, name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"[Train] Best model ({name}) saved → {BEST_MODEL_PATH}")

# ── Main ──────────────────────────────────────────────────────────

def train(X_train, X_test, y_train, y_test):
    print("\nClass Distribution (train):")
    print(pd.Series(y_train).value_counts())

    cv_results    = cross_validate_all(X_train, y_train)
    print("\nCV Ranking:")
    print(cv_results.to_string(index=False))

    comparison_df, best_model, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )
    print("\nTest Set Ranking:")
    print(comparison_df.to_string(index=False))

    # ── detailed_report receives original DataFrame X_test ────────
    detailed_report(best_model, X_test, y_test, best_name)
    save_best_model(best_model, best_name)

    return best_model, best_name, comparison_df, cv_results


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader         import load_data
    from feature_engineering import engineer_features
    from preprocess          import preprocess

    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    train(X_train, X_test, y_train, y_test)