import os
import joblib
import warnings
import numpy as np
import pandas as pd
import json
from typing import Optional, List, Dict, Tuple

from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.inspection      import permutation_importance
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SVMSMOTE

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DIR       = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
CV_FOLDS        = 5
RANDOM_STATE    = 42
TUNE_TOP_N      = 3          # how many top-CV models to carry into GridSearchCV

# ROC-AUC is the right optimisation metric for imbalanced binary classification
TUNING_METRIC   = "roc_auc"

# Models that require feature scaling before training / tuning
NEEDS_SCALING = {"SVM", "Logistic Regression"}
APPLY_SMOTE     = True          # set False to disable SMOTE globally
SMOTE_NEIGHBORS = 5    
# ── Model Registry ────────────────────────────────────────────────────────────

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
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
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
        verbose=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=RANDOM_STATE,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=RANDOM_STATE,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
}

# ── Hyperparameter Search Spaces ──────────────────────────────────────────────
# Focused grids covering the highest-impact parameters.
# Extend any grid to deepen the search at the cost of more wall-clock time.

PARAM_GRIDS: Dict[str, dict] = {
    "XGBoost": {
        "n_estimators"    : [300, 500],
        "max_depth"       : [4, 5, 6],
        "learning_rate"   : [0.03, 0.05, 0.1],
        "subsample"       : [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    },
    "LightGBM": {
        "n_estimators"     : [300, 500],
        "max_depth"        : [5, 7],
        "num_leaves"       : [31, 63],
        "learning_rate"    : [0.03, 0.05, 0.1],
        "min_child_samples": [10, 20],
    },
    "Random Forest": {
        "n_estimators"     : [300, 500],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 4],
        "min_samples_leaf" : [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators" : [100, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth"    : [3, 4, 5],
    },
    "Extra Trees": {
        "n_estimators"     : [200, 400],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 4],
    },
    "SVM": {
        "C"    : [1, 10, 50],
        "gamma": ["scale", "auto"],
    },
    "Logistic Regression": {
        "C"      : [0.01, 0.1, 1, 10],
        "solver" : ["lbfgs", "liblinear"],
        "penalty": ["l2"],
    },
}

def _scale_pos_weight(y: np.ndarray) -> float:
    """
    FIX #1 – scale_pos_weight for XGBoost.
    Computes neg/pos ratio from training labels so XGBoost correctly
    handles class imbalance instead of using the hardcoded value of 1.
    """
    counts = np.bincount(y.astype(int))
    return float(counts[0]) / float(counts[1]) if counts[1] > 0 else 1.0
# ── Scaler Helper ─────────────────────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SVMSMOTE to the training data to balance class distribution.

    Why SVMSMOTE for this dataset:
      - Uses SVM support vectors to identify the decision boundary
      - Generates synthetic minority samples only near the boundary
      - Handles heavy class overlap (water quality dataset LDA score ~0.61)
      - Robust to widely scattered minority samples

    IMPORTANT: Only call this on training data — never on test/validation data.

    Returns
    -------
    X_resampled : pd.DataFrame  — balanced feature matrix
    y_resampled : np.ndarray    — balanced target vector
    """
    if not APPLY_SMOTE:
        return X_train, y_train

    counts_before = pd.Series(y_train).value_counts().to_dict()

    smote = SVMSMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    counts_after = pd.Series(y_resampled).value_counts().to_dict()

    print("\n[SMOTE] SVMSMOTE applied to training data:")
    print(f"  Before → Class 0: {counts_before.get(0, 0)}  "
          f"Class 1: {counts_before.get(1, 0)}")
    print(f"  After  → Class 0: {counts_after.get(0, 0)}  "
          f"Class 1: {counts_after.get(1, 0)}")
    print(f"  Synthetic samples added: "
          f"{counts_after.get(1, 0) - counts_before.get(1, 0)}")

    # Preserve DataFrame column names for downstream feature importance
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    return X_resampled, y_resampled

def _make_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit and return a brand-new StandardScaler on X_train.
    Called separately for the CV, tuning, and evaluation phases so
    there is NO shared mutable state between phases (prevents data leakage).
    """
    return StandardScaler().fit(X_train)


# ── Cross Validation ──────────────────────────────────────────────────────────

def cross_validate_all(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    """
    Run stratified k-fold CV on all base models (no tuning).
    Returns a DataFrame ranked by CV ROC-AUC descending.
    """
    print("\n" + "=" * 60)
    print("  CROSS VALIDATION  ({}-Fold Stratified)".format(CV_FOLDS))
    print("=" * 60)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Fresh scaler for the CV phase only
    cv_scaler   = _make_scaler(X_train)
    X_train_sc  = cv_scaler.transform(X_train)

    results = []
    for name, model in MODELS.items():
        X_input = X_train_sc if name in NEEDS_SCALING else X_train

        scores = cross_validate(
            model, X_input, y_train,
            cv=skf,
            scoring=["accuracy", "f1", "roc_auc"],
            n_jobs=-1,
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


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_names: List[str],
) -> dict:
    """
    Run GridSearchCV for each model in `model_names`.

    Returns
    -------
    dict  →  {
        model_name: {
            "best_params"    : dict,
            "best_cv_score"  : float,
            "best_estimator" : fitted estimator,
            "scaler"         : StandardScaler | None
                               (not-None only for NEEDS_SCALING models so
                                evaluate.py can transform X_test correctly)
        }
    }
    """
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING  (GridSearchCV, metric={})".format(TUNING_METRIC))
    print("=" * 60)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Fresh scaler for the tuning phase — independent of CV and evaluation
    tune_scaler = _make_scaler(X_train)
    X_train_sc  = tune_scaler.transform(X_train)

    tuning_results = {}

    for name in model_names:
        if name not in PARAM_GRIDS:
            print(f"  [SKIP] {name}: no param grid defined.")
            continue

        print(f"\n  Tuning {name} ...")
        base_model = MODELS[name]
        param_grid = PARAM_GRIDS[name]
        X_input    = X_train_sc if name in NEEDS_SCALING else X_train

        grid_search = GridSearchCV(
            estimator  = base_model,
            param_grid = param_grid,
            cv         = skf,
            scoring    = TUNING_METRIC,
            n_jobs     = -1,
            refit      = True,   # best estimator is re-fitted on full X_input
            verbose    = 0,
        )
        grid_search.fit(X_input, y_train)

        tuning_results[name] = {
            "best_params"    : grid_search.best_params_,
            "best_cv_score"  : round(grid_search.best_score_, 4),
            "best_estimator" : grid_search.best_estimator_,
            # Preserve scaler so evaluate.py can transform X_test correctly
            "scaler"         : tune_scaler if name in NEEDS_SCALING else None,
        }

        print(f"  ✓ {name}")
        print(f"    Best CV {TUNING_METRIC} : {tuning_results[name]['best_cv_score']:.4f}")
        print(f"    Best params      : {grid_search.best_params_}")

    return tuning_results


# ── Train All & Evaluate ──────────────────────────────────────────────────────

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    tuned_models: Optional[dict] = None,
) -> Tuple[pd.DataFrame, object, str, StandardScaler]:
    """
    Fit every model on the full training set and evaluate on the held-out test set.
    When a tuned estimator exists for a model it is used; otherwise the base model is used.

    Returns
    -------
    comparison_df : pd.DataFrame    — per-model test metrics, sorted by ROC-AUC
    best_model    : fitted estimator
    best_name     : model name string (plain, no " [tuned]" suffix)
    eval_scaler   : StandardScaler fitted on X_train for this phase
                    (passed on to detailed_report and evaluate.py for
                     permutation importance on SVM / Logistic Regression)
    """
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    feature_names: List[str] = list(X_train.columns)
    results: List[dict]      = []
    trained_models: Dict[str, object] = {}

    # Fresh scaler for the evaluation phase — independent of CV and tuning
    eval_scaler    = _make_scaler(X_train)
    X_train_scaled = eval_scaler.transform(X_train)
    X_test_scaled  = eval_scaler.transform(X_test)

    for name, base_model in MODELS.items():
        model    = (tuned_models or {}).get(name, base_model)
        is_tuned = name in (tuned_models or {})
        tag      = " [tuned]" if is_tuned else ""

        if name in NEEDS_SCALING:
            if not is_tuned:
                model.fit(X_train_scaled, y_train)
            # For a consistent test-set comparison all scaled models use
            # the same eval_scaler — even the tuned ones.
            y_pred  = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            if not is_tuned:
                model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model"    : name + tag,
            "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall"   : round(recall_score(y_test, y_pred),    4),
            "F1 Score" : round(f1_score(y_test, y_pred),        4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),  4),
        })
        trained_models[name] = model
        print(f"  {(name + tag):<30}  Acc={results[-1]['Accuracy']:.4f}  "
              f"F1={results[-1]['F1 Score']:.4f}  AUC={results[-1]['ROC-AUC']:.4f}")

    comparison_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

    # ── Persist comparison metrics ────────────────────────────────────────────
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
    print(f"\n[Train] Saved → models/comparison.json")

    # ── Persist feature names (used by detailed_report) ──────────────────────
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"[Train] Saved → models/feature_names.json")

    # Strip " [tuned]" suffix to resolve the plain name for dict lookup
    best_row   = comparison_df.iloc[0]["Model"]
    best_name  = best_row.replace(" [tuned]", "")
    best_model = trained_models[best_name]

    return comparison_df, best_model, best_name, eval_scaler


# ── Detailed Report ───────────────────────────────────────────────────────────

def detailed_report(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    name: str,
    eval_scaler: Optional[StandardScaler] = None,
) -> None:
    """
    Prints confusion matrix + classification report.
    Saves confusion_matrix.json and feature_importance.json to models/.

    Parameters
    ----------
    eval_scaler : StandardScaler fitted on X_train during evaluation.
                  Required so that SVM / Logistic Regression permutation
                  importance is computed on correctly scaled data.
    """
    # Apply scaling only if the best model needs it
    X_eval = (eval_scaler.transform(X_test)
              if (name in NEEDS_SCALING and eval_scaler is not None)
              else X_test)

    y_pred = model.predict(X_eval)

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

    # ── Save confusion matrix ─────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)
    print(f"[Train] Saved → models/confusion_matrix.json")

    # ── Save feature importance ───────────────────────────────────────────────
    fn_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(fn_path) as f:
        feature_names = json.load(f)

    if hasattr(model, "feature_importances_"):
        # Tree-based models: native impurity-based importance
        importances = model.feature_importances_.tolist()
        method      = "tree_impurity"
        print(f"[Train] Using native feature_importances_ ({name})")
    else:
        # SVM / Logistic Regression: permutation importance on scaled data
        print(f"[Train] {name} has no feature_importances_ → using permutation importance...")
        perm = permutation_importance(
            model, X_eval, y_test,
            n_repeats=15,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
        )
        importances = np.abs(perm.importances_mean).tolist()
        method      = "permutation"
        print(f"[Train] Permutation importance done.")

    if len(importances) == len(feature_names):
        fi = {
            "features"   : feature_names,
            "importances": importances,
            "method"     : method,
        }
        with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)
        print(f"[Train] Saved → models/feature_importance.json")
    else:
        print(f"[Train] ⚠ Mismatch: {len(importances)} importances vs "
              f"{len(feature_names)} names — skipping")


# ── Save Best Hyperparameters ─────────────────────────────────────────────────

def save_best_hyperparameters(tuning_results: dict) -> None:
    """Serialise best hyperparameters from GridSearchCV to JSON for auditing."""
    summary = {
        name: {
            "best_params"  : info["best_params"],
            "best_cv_score": info["best_cv_score"],
            "metric"       : TUNING_METRIC,
        }
        for name, info in tuning_results.items()
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "best_hyperparameters.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Train] Saved → {path}")


# ── Save Best Model ───────────────────────────────────────────────────────────

def save_best_model(model, name: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"[Train] Best model ({name}) saved → {BEST_MODEL_PATH}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[object, str, pd.DataFrame, pd.DataFrame]:
    """
    Full training pipeline:
      1. Cross-validate all base models (no tuning)
      2. Tune the top-N models with GridSearchCV (ROC-AUC)
      3. Evaluate all models on the held-out test set
      4. Generate detailed report for the overall best model
      5. Persist the best model and all artefacts to disk

    Returns  (exactly 4 values — backward-compatible with evaluate.py)
    -------
    best_model    : fitted estimator
    best_name     : model name string
    comparison_df : test-set metrics DataFrame
    cv_results    : cross-validation metrics DataFrame

    Tuning results are also saved automatically to
    models/best_hyperparameters.json.
    """
    print("\nClass Distribution (train):")
    print(pd.Series(y_train).value_counts())
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── Step 1 : Cross-validate all base models ───────────────────────────────
    cv_results = cross_validate_all(X_train_bal, y_train_bal)
    print("\nCV Ranking:")
    print(cv_results.to_string(index=False))

    # ── Step 2 : Tune top-N models ────────────────────────────────────────────
    top_model_names = cv_results["Model"].head(TUNE_TOP_N).tolist()
    print(f"\n[Tuning] Top-{TUNE_TOP_N} models selected: {top_model_names}")

    tuning_results = tune_hyperparameters(X_train, y_train, top_model_names)
    save_best_hyperparameters(tuning_results)

    # Extract tuned estimators { model_name → best_estimator }
    tuned_estimators = {
        name: info["best_estimator"]
        for name, info in tuning_results.items()
    }

    # ── Step 3 : Evaluate on held-out test set ────────────────────────────────
    comparison_df, best_model, best_name, eval_scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        tuned_models=tuned_estimators,
    )
    print("\nTest Set Ranking:")
    print(comparison_df.to_string(index=False))

    # ── Step 4 : Detailed report + persist ───────────────────────────────────
    detailed_report(best_model, X_test, y_test, best_name, eval_scaler=eval_scaler)
    save_best_model(best_model, best_name)

    # ── Return exactly 4 values (matches evaluate.py caller) ─────────────────
    # best_model, best_name, comparison_df, cv_results = train(...)
    return best_model, best_name, comparison_df, cv_results


# ── Script Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader         import load_data
    from feature_engineering import engineer_features
    from preprocess          import preprocess

    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    train(X_train, X_test, y_train, y_test)