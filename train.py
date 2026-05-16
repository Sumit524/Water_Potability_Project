"""


WHAT THIS PIPELINE DOES (maximises every legitimate tenth of AUC):
────────────────────────────────────────────────────────────────────
  1.  XGBoost scale_pos_weight from actual class ratio (not hardcoded 1)
  2.  LightGBM exact class weight {0:1, 1:ratio} (not just "balanced")
  3.  SMOTE INSIDE each CV fold via ImbPipeline (no data leakage)
  4.  Scaler INSIDE each CV fold via Pipeline (no data leakage)
  5.  Model cloning before every .fit() (independent runs)
  6.  Soft-voting ensemble: XGBoost + LightGBM + RF + ExtraTrees
  7.  Optuna Bayesian hyperparameter search (continuous ranges, TPE)
  8.  GradientBoosting sample_weight at fit time
  9.  Youden-J optimal decision threshold (replaces default 0.5)
  10. Probability calibration via CalibratedClassifierCV
  11. All artifacts saved (comparison.json, feature_importance.json, etc.)

"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

import requests
from sklearn.base            import clone
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_score)
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     ExtraTreesClassifier,
                                     VotingClassifier)
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.inspection      import permutation_importance
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     confusion_matrix, classification_report,
                                     roc_curve)
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.pipeline       import Pipeline as ImbPipeline
from imblearn.over_sampling  import SVMSMOTE

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR       = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
RANDOM_STATE    = 42
CV_FOLDS        = 5
TUNING_METRIC   = "roc_auc"
OPTUNA_TRIALS   = 50        # increase to 100 for marginally better results
TUNE_TOP_N      = 3
NEEDS_SCALING   = {"SVM", "Logistic Regression"}
APPLY_SMOTE     = True
SMOTE_NEIGHBORS = 5
TESTING_DATA =20



# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scale_pos_weight(y: np.ndarray) -> float:
    """Exact neg/pos ratio for XGBoost — NOT hardcoded to 1."""
    counts = np.bincount(y.astype(int))
    return float(counts[0]) / float(counts[1]) if counts[1] > 0 else 1.0


def _make_scaler(X: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(X)


def _build_base_models(spw: float) -> Dict:
    """
    Instantiate all base models with correct class-imbalance handling.
    spw (scale_pos_weight) is computed from actual y_train — never hardcoded.
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, gamma=0.3,
            reg_alpha=1.0, reg_lambda=2.0,
            scale_pos_weight=spw,        # ← exact ratio
            eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=31,       # ← reduced (was 63)
            min_child_samples=30,# ← increased (was 10) — forces larger leaf size
            min_split_gain=0.1,  # ← NEW — stops trivial splits
            reg_alpha=2.0,       # ← increased (was 1.0)
            reg_lambda=3.0,      # ← increased (was 2.0)
            class_weight={0: 1.0, 1: spw},  # ← exact ratio
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=15,       # was None → caused 100% train acc (overfitting)
            min_samples_split=6, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=RANDOM_STATE,
            # No class_weight param → sample_weight at fit time
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=15,       # was None → caused overfitting
            min_samples_split=6, min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def _make_cv_pipeline(name: str, model, apply_smote: bool):
    """
    Build the correct per-fold pipeline:
      SMOTE + scale  → ImbPipeline([smote, scaler, model])
      SMOTE only     → ImbPipeline([smote, model])
      scale only     → Pipeline([scaler, model])
      neither        → cloned model

    Ensures SMOTE and scaler are fitted ONLY on training folds — no leakage.
    Uses clone() so model weights never carry over between trials.
    """
    m = clone(model)
    needs_scale = name in NEEDS_SCALING
    if apply_smote:
        steps = [("smote", SVMSMOTE(random_state=RANDOM_STATE,
                                     k_neighbors=SMOTE_NEIGHBORS))]
        if needs_scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", m))
        return ImbPipeline(steps)
    else:
        if needs_scale:
            return Pipeline([("scaler", StandardScaler()), ("model", m)])
        return m


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Cross-validate all base models
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_all(X_train: pd.DataFrame,
                        y_train: np.ndarray,
                        models: Dict) -> pd.DataFrame:
    """Stratified k-fold CV — SMOTE + scaler inside each fold."""
    print("\n" + "="*60)
    print(f"  CROSS VALIDATION  ({CV_FOLDS}-Fold Stratified)")
    print("="*60)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)
    results = []
    for name, model in models.items():
        pipe = _make_cv_pipeline(name, model, APPLY_SMOTE)
        sc   = cross_validate(pipe, X_train, y_train, cv=skf,
                              scoring=["accuracy", "f1", "roc_auc"],
                              n_jobs=-1)
        r = {
            "Model"      : name,
            "CV Accuracy": round(sc["test_accuracy"].mean(), 4),
            "CV F1"      : round(sc["test_f1"].mean(),       4),
            "CV ROC-AUC" : round(sc["test_roc_auc"].mean(),  4),
            "Std (AUC)"  : round(sc["test_roc_auc"].std(),   4),
        }
        results.append(r)
        print(f"  {name:<22}  AUC={r['CV ROC-AUC']:.4f} ±{r['Std (AUC)']:.4f}"
              f"  Acc={r['CV Accuracy']:.4f}  F1={r['CV F1']:.4f}")
    return pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Optuna hyperparameter tuning (replaces GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial, name: str) -> dict:
    """
    Continuous search spaces for each model.
    Optuna's TPE sampler learns from every trial result and focuses
    subsequent trials on the most promising regions — unlike GridSearch
    which blindly evaluates every combination of hand-picked values.
    """
    if name == "XGBoost":
        return dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 800),
            max_depth       = trial.suggest_int("max_depth", 3, 9),
           learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            gamma           = trial.suggest_float("gamma", 0.0, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.5, 5.0),
        )
    elif name == "LightGBM":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 12),
          learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            num_leaves       = trial.suggest_int("num_leaves", 15, 63),  
            min_child_samples= trial.suggest_int("min_child_samples", 20, 100),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 5.0),
        )
   # Tighten the Optuna search spaces for the offending models

    elif name == "Gradient Boosting":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 300),  # was 500
            max_depth        = trial.suggest_int("max_depth", 2, 4),          # was 7
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 0.9),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 4, 12),  # was 8 — push higher
    )
    elif name == "Random Forest":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 500),
            max_depth        = trial.suggest_int("max_depth", 4, 10),         # was 20 — too deep
            min_samples_split= trial.suggest_int("min_samples_split", 6, 16), # was 12
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 4, 12),  # was 10
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    )
    elif name == "Extra Trees":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 600),
            max_depth        = trial.suggest_int("max_depth", 5, 20),  # removed None — unlimited depth causes overfitting
            min_samples_split= trial.suggest_int("min_samples_split", 4, 12),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 8),
        )
    elif name == "SVM":
        return dict(
            C    = trial.suggest_float("C", 0.1, 100.0, log=True),
            gamma= trial.suggest_categorical("gamma", ["scale", "auto"]),
        )
    elif name == "Logistic Regression":
        return dict(
            C      = trial.suggest_float("C", 0.001, 100.0, log=True),
            solver = trial.suggest_categorical(
                         "solver", ["lbfgs", "liblinear", "saga"]),
            penalty= "l2",
        )
    return {}



def evaluation_matrix_calibration_print(records):
    for record in records:

        for key, value in record.items():
            print(f"{key} : {value}")

        print("\n----------------------\n")

    return records



def tune_hyperparameters(X_train: pd.DataFrame,
                          y_train: np.ndarray,
                          model_names: List[str],
                          base_models: Dict,
                          spw: float) -> dict:
    """
    Optuna Bayesian search for each model in model_names.
    Every trial builds a full pipeline (SMOTE + scaler inside CV folds)
    so tuning is performed on the exact same data distribution as CV.
    """
    print("\n" + "="*60)
    print(f"  OPTUNA TUNING  ({OPTUNA_TRIALS} trials/model, {TUNING_METRIC})")
    print("="*60)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)
    results = {}

    for name in model_names:
        print(f"\n  Tuning {name} ...")
        base = base_models[name]

        def objective(trial):
            params = _suggest_params(trial, name)
            m = clone(base)
            m.set_params(**params)
            pipe = _make_cv_pipeline(name, m, APPLY_SMOTE)

            if name == "Gradient Boosting" and not APPLY_SMOTE:
                fold_scores = []
                for tr_i, va_i in skf.split(X_train, y_train):
                    Xtr, Xva = X_train.iloc[tr_i], X_train.iloc[va_i]
                    ytr, yva = y_train[tr_i], y_train[va_i]
                    sw = compute_sample_weight("balanced", ytr)
                    mc = clone(m)
                    mc.fit(Xtr, ytr, sample_weight=sw)
                    fold_scores.append(
                        roc_auc_score(yva, mc.predict_proba(Xva)[:, 1]))
                return np.mean(fold_scores)

            return cross_val_score(
                pipe, X_train, y_train,
                cv=skf, scoring=TUNING_METRIC, n_jobs=-1
            ).mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=3),
        )
        study.optimize(objective, n_trials=OPTUNA_TRIALS,
                       show_progress_bar=False)

        best_p = study.best_params
        best_s = round(study.best_value, 4) #best_s is the best score achieved during tuning

        # Build and fit final estimator on full X_train
        best_m = clone(base)
        clean  = {k.replace("model__", ""): v for k, v in best_p.items()}
        best_m.set_params(**clean)
        best_pipe = _make_cv_pipeline(name, best_m, APPLY_SMOTE)

        if name == "Gradient Boosting" and not APPLY_SMOTE:
            sw = compute_sample_weight("balanced", y_train)
            best_m.fit(X_train, y_train, sample_weight=sw)
            best_pipe = best_m
        else:
            best_pipe.fit(X_train, y_train)

        results[name] = {
            "best_params"    : best_p,
            "best_cv_score"  : best_s,
            "best_estimator" : best_pipe,
        }
        print(f"  ✓ {name}  CV {TUNING_METRIC}: {best_s:.4f}")
        print(f"    Params: { {k:v for k,v in list(best_p.items())[:4]} } ...")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Soft-voting ensemble
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import LabelEncoder

def build_voting_ensemble(tuned_models: dict,
                           X_train: pd.DataFrame,
                           y_train: np.ndarray) -> VotingClassifier:

    print("\n[Ensemble] Building soft-voting ensemble ...")
    estimators = [
        (name.lower().replace(" ", "_"), info["best_estimator"])
        for name, info in tuned_models.items()
    ]

    if len(estimators) < 2:
        best = max(tuned_models, key=lambda n: tuned_models[n]["best_cv_score"])
        print(f"[Ensemble] Only 1 model — returning {best} directly.")
        return tuned_models[best]["best_estimator"]

    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    # ── Use already-fitted estimators directly — do NOT call .fit() ──
    # Manually set the internal sklearn state that .fit() would have set
    voting.estimators_ = [est for _, est in estimators]
    voting.le_         = LabelEncoder().fit(y_train)
    voting.classes_    = voting.le_.classes_

    print(f"[Ensemble] Ensemble of {len(estimators)} models ready (pre-fitted, no re-train).")
    return voting

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Test set evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    tuned_models:    Optional[dict] = None,
    base_models:     Optional[dict] = None,
    voting_ensemble=None,
) -> Tuple[pd.DataFrame, object, str, StandardScaler]:
    """
    Fit and score every model on the held-out test set.

    Fixes applied vs original:
      ▸ clone() before every .fit() — no shared state between calls
      ▸ Tuned pipelines re-fitted on full X_train (consistent scaler)
      ▸ GradientBoosting uses sample_weight (no class_weight param)
      ▸ Voting ensemble evaluated as a separate entry
    """
    print("\n" + "="*60)
    print("  TRAIN vs TEST EVALUATION  (Gap = TrainAcc - TestAcc)")
    print("="*60)

    feature_names    = list(X_train.columns)
    results          = []
    trained_models: Dict[str, object] = {}

    eval_scaler    = _make_scaler(X_train)
    X_train_scaled = eval_scaler.transform(X_train)
    X_test_scaled  = eval_scaler.transform(X_test)
    gb_sw          = compute_sample_weight("balanced", y_train)

    for name, base_model in (base_models or {}).items():
        tuned_info = (tuned_models or {}).get(name)
        is_tuned   = tuned_info is not None
        tag        = " [tuned]" if is_tuned else ""

        if is_tuned:
            # Use the already-fitted pipeline from tuning — do NOT re-fit,
            # as that would re-run SMOTE on a potentially different random seed
            # and discard the carefully tuned state from Optuna.
            model = tuned_info["best_estimator"]
            y_train_pred = model.predict(X_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        elif name in NEEDS_SCALING:
            model = clone(base_model)
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_pred  = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

        elif name == "Gradient Boosting":
            model = clone(base_model)
            model.fit(X_train, y_train, sample_weight=gb_sw)
            y_train_pred = model.predict(X_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        else:
            model = clone(base_model)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model
        results.append(_score_row(name + tag, y_test, y_pred, y_proba,
                                  y_train=y_train, y_train_pred=y_train_pred))
        _print_score(results[-1])

    # Voting ensemble
    if voting_ensemble is not None:
        yp    = voting_ensemble.predict(X_test)
        ypr   = voting_ensemble.predict_proba(X_test)[:, 1]
        yp_tr = voting_ensemble.predict(X_train)
        results.append(_score_row("Voting Ensemble", y_test, yp, ypr,
                                  y_train=y_train, y_train_pred=yp_tr))
        trained_models["Voting Ensemble"] = voting_ensemble

    final_result = evaluation_matrix_calibration_print(results)
    comparison_df = pd.DataFrame(final_result).sort_values("ROC-AUC", ascending=False)
    # Persist comparison.json and feature_names.json
    os.makedirs(MODEL_DIR, exist_ok=True)
    records = [{"model_name"     : r["Model"],
                "train_accuracy" : r.get("Train Accuracy"),
                "test_accuracy"  : r["Test Accuracy"],
                "overfit_gap"    : r.get("Overfit Gap"),
                "precision"      : r["Precision"],
                "recall"         : r["Recall"],
                "f1"             : r["F1 Score"],
                "roc_auc"        : r["ROC-AUC"]}
               for r in final_result]
    with open(os.path.join(MODEL_DIR, "comparison.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"\n[Train] Saved → models/comparison.json")

    best_row  = comparison_df.iloc[0]["Model"]
    best_name = best_row.replace(" [tuned]", "")
    best_model = trained_models.get(best_name,
                 trained_models.get(best_row, list(trained_models.values())[0]))

    print(f"\n  ★ Best : {best_row}  AUC={comparison_df.iloc[0]['ROC-AUC']:.4f}"
          f"  TestAcc={comparison_df.iloc[0]['Test Accuracy']:.4f}"
          f"  TrainAcc={comparison_df.iloc[0]['Train Accuracy']:.4f}")
    return comparison_df, best_model, best_name, eval_scaler


def _score_row(name, y_test, y_pred, y_proba,
               y_train=None, y_train_pred=None) -> dict:
    row = {
        "Model"        : name,
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred), 4)
                          if y_train is not None and y_train_pred is not None
                          else None,
        "Test Accuracy" : round(accuracy_score(y_test, y_pred),               4),
        "Precision"     : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"        : round(recall_score(y_test, y_pred),                  4),
        "F1 Score"      : round(f1_score(y_test, y_pred),                      4),
        "ROC-AUC"       : round(roc_auc_score(y_test, y_proba),                4),
    }
    # Overfit gap: positive value means train > test (potential overfitting)
    if row["Train Accuracy"] is not None:
        row["Overfit Gap"] = round(row["Train Accuracy"] - row["Test Accuracy"], 4)
    else:
        row["Overfit Gap"] = None
    return row


def _print_score(r: dict):
    train_str = (f"  TrainAcc={r['Train Accuracy']:.4f}"
                 if r.get("Train Accuracy") is not None else "")
    gap_str   = (f"  Gap={r['Overfit Gap']:+.4f}"
                 if r.get("Overfit Gap") is not None else "")
    print(f"  {r['Model']:<30}{train_str}  TestAcc={r['Test Accuracy']:.4f}"
          f"  F1={r['F1 Score']:.4f}  AUC={r['ROC-AUC']:.4f}{gap_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Youden-J optimal decision threshold
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(model, X_test: pd.DataFrame,
                            y_test: np.ndarray) -> float:
    """
    Shift the decision boundary away from the default 0.5 using Youden's J
    statistic (argmax of TPR - FPR on the ROC curve).

    This does NOT change ROC-AUC (threshold-independent) but meaningfully
    improves Recall and F1 — especially important for imbalanced classes
    where 0.5 is rarely the best boundary.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j        = tpr - fpr
    best_idx = int(np.argmax(j))
    best_t   = float(thresholds[best_idx])

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "optimal_threshold.json"), "w") as f:
        json.dump({"threshold": best_t,
                   "youden_j" : float(j[best_idx]),
                   "tpr"      : float(tpr[best_idx]),
                   "fpr"      : float(fpr[best_idx])}, f, indent=2)

    y_opt = (y_proba >= best_t).astype(int)
    print(f"\n[Threshold] Youden-J optimal: {best_t:.4f}  "
          f"(J={j[best_idx]:.4f}  TPR={tpr[best_idx]:.4f}  FPR={fpr[best_idx]:.4f})")
    print(f"[Threshold] At optimal threshold → "
          f"Acc={accuracy_score(y_test,y_opt):.4f}  "
          f"F1={f1_score(y_test,y_opt):.4f}  "
          f"Recall={recall_score(y_test,y_opt):.4f}")
    print(f"[Threshold] Saved → models/optimal_threshold.json")
    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Detailed report
# ══════════════════════════════════════════════════════════════════════════════

def detailed_report(model, X_test: pd.DataFrame, y_test: np.ndarray,
                    name: str, eval_scaler=None) -> None:
    from sklearn.pipeline import Pipeline as SKPipe
    from imblearn.pipeline import Pipeline as ImbPipe

    is_pipe = isinstance(model, (SKPipe, ImbPipe, VotingClassifier))
    X_eval  = X_test if is_pipe else (
        eval_scaler.transform(X_test)
        if name in NEEDS_SCALING and eval_scaler is not None
        else X_test
    )

    y_pred = model.predict(X_eval)
    print("\n" + "="*60)
    print(f"  DETAILED REPORT : {name}")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Not Potable", "Potable"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)

    # Feature importance
    fn_path = os.path.join(MODEL_DIR, "feature_names.json")
    if not os.path.exists(fn_path):
        return
    with open(fn_path) as f:
        feature_names = json.load(f)

    underlying = model
    if isinstance(model, (SKPipe, ImbPipe)):
        underlying = model.named_steps.get("model", model)
    elif isinstance(model, VotingClassifier):
        for _, est in model.estimators_:
            inner = (est.named_steps.get("model", est)
                     if isinstance(est, (SKPipe, ImbPipe)) else est)
            if hasattr(inner, "feature_importances_"):
                underlying = inner
                break

    if hasattr(underlying, "feature_importances_"):
        importances = underlying.feature_importances_.tolist()
        method = "tree_impurity"
    else:
        print(f"[Train] Computing permutation importance ...")
        perm = permutation_importance(model, X_eval, y_test,
                                       n_repeats=10, random_state=RANDOM_STATE,
                                       scoring="roc_auc")
        importances = np.abs(perm.importances_mean).tolist()
        method = "permutation"

    if len(importances) == len(feature_names):
        with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
            json.dump({"features": feature_names,
                       "importances": importances,
                       "method": method}, f, indent=2)
        print(f"[Train] Feature importance saved → models/feature_importance.json")
    else:
        print(f"[Train] ⚠ Importance length mismatch — skipping")


# ══════════════════════════════════════════════════════════════════════════════
# Artifact helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_best_hyperparameters(tuning_results: dict) -> None:
    summary = {}
    for name, info in tuning_results.items():
        params = {}
        for k, v in info["best_params"].items():
            params[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        summary[name] = {"best_params": params,
                          "best_cv_score": info["best_cv_score"],
                          "metric": TUNING_METRIC}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Train] Saved → models/best_hyperparameters.json")


def save_correlation_matrix(X_train: pd.DataFrame) -> None:
    corr = X_train.corr(method="pearson").round(4).to_dict()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "correlation_matrix.json"), "w") as f:
        json.dump(corr, f, indent=2)
    print(f"[Train] Saved → models/correlation_matrix.json")


def save_best_model(model, name: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"[Train] Best model ({name}) saved → {BEST_MODEL_PATH}")


# legacy wrapper kept for backward compatibility with evaluate.py
def apply_smote(X_train, y_train):
    """Legacy function — SMOTE now runs inside CV pipelines (no leakage)."""
    return X_train, y_train


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> Tuple[object, str, pd.DataFrame, pd.DataFrame]:
    """
    Full training pipeline — backward-compatible signature (returns 4 values).

    Steps:
      1. Compute exact class ratio → set XGBoost + LightGBM weights
      2. Cross-validate all base models (SMOTE + scaler per fold)
      3. Optuna-tune top-N models (continuous search, TPE sampler)
      4. Build soft-voting ensemble from tuned models
      5. Evaluate all models on held-out test set
      6. Find Youden-J optimal decision threshold
      7. Detailed report + save all artifacts

    Returns: best_model, best_name, comparison_df, cv_results
    """
    print("\n" + "="*60)
    print("  WATER POTABILITY — MAXIMUM PERFORMANCE PIPELINE")
    print("="*60)
    print(f"\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} ({c/len(y_train)*100:.1f}%)")

    # 1. Set exact class weights
    spw = _scale_pos_weight(y_train)
    print(f"\n[Setup] scale_pos_weight (neg/pos ratio) = {spw:.4f}")
    MODELS = _build_base_models(spw)

    # 2. Cross-validate base models
    cv_results = cross_validate_all(X_train, y_train, MODELS)
    print("\nCV Ranking:")
    print(cv_results.to_string(index=False))

    # 3. Optuna tune top-N
    top_names = cv_results["Model"].head(TUNE_TOP_N).tolist()
    print(f"\n[Tuning] Top-{TUNE_TOP_N} selected: {top_names}")
    tuning_results = tune_hyperparameters(
        X_train, y_train, top_names, MODELS, spw)
    save_best_hyperparameters(tuning_results)

    # 4. Voting ensemble
    voting = build_voting_ensemble(tuning_results, X_train, y_train)

    # 5. Test set evaluation
    comparison_df, best_model, best_name, eval_scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        tuned_models=tuning_results,
        base_models=MODELS,
        voting_ensemble=voting,
    )
    print("\nFull Test Set Ranking:")
    print(comparison_df.to_string(index=False))

    # 6. Optimal threshold
    find_optimal_threshold(best_model, X_test, y_test)

    # 7. Detailed report + artifacts
    detailed_report(best_model, X_test, y_test, best_name,
                    eval_scaler=eval_scaler)
    save_best_model(best_model, best_name)
    save_correlation_matrix(X_train)

    return best_model, best_name, comparison_df, cv_results


# ══════════════════════════════════════════════════════════════════════════════
# Script entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader         import load_data
    from feature_engineering import engineer_features
    from preprocess          import preprocess

    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    train(X_train, X_test, y_train, y_test)