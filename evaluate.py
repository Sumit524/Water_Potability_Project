import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from sklearn.inspection  import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

REPORT_DIR = "reports"
MODEL_DIR  = "models"
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Models that were trained on scaled data — must scale X_test before predicting
NEEDS_SCALING = {"SVM", "Logistic Regression"}

os.makedirs(REPORT_DIR, exist_ok=True)
plt.style.use(PLOT_STYLE)

__all__ = ["evaluate", "evaluate_selected"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filename(base: str, selected: bool) -> str:
    """Return report path, appending '_selected' suffix when appropriate."""
    name = f"{base}_selected.png" if selected else f"{base}.png"
    return os.path.join(REPORT_DIR, name)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Saved → {path}")


# ── 1. Metric Summary Card ────────────────────────────────────────────────────

def _plot_metric_summary(
    y_test, y_pred, y_proba, model_name: str, selected: bool
) -> None:
    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1 Score" : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, ((metric, value), color) in enumerate(zip(metrics.items(), colors)):
        x = 0.1 + i * 0.19
        ax.add_patch(plt.Rectangle(
            (x - 0.08, 0.15), 0.16, 0.7,
            color=color, alpha=0.15, transform=ax.transAxes,
        ))
        ax.text(x, 0.72, metric, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, 0.38, f"{value:.4f}", ha="center", va="center",
                fontsize=16, fontweight="bold", color=color, transform=ax.transAxes)

    ax.set_title(f"Performance Summary — {model_name}", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    _save(fig, _filename("metric_summary", selected))


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, model_name: str, selected: bool) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Potable", "Potable"],
        yticklabels=["Not Potable", "Potable"],
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.tight_layout()
    _save(fig, _filename("confusion_matrix", selected))


# ── 3. ROC Curve ──────────────────────────────────────────────────────────────

def _plot_roc_curve(
    y_test, y_proba, model_name: str, selected: bool
) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # Youden's J statistic: maximise TPR - FPR
    ix             = np.argmax(tpr - fpr)
    best_threshold = float(thresholds[ix])
    print(f"[Evaluate] Best Threshold (Youden's J): {best_threshold:.4f}")

    # Use variant-specific filename so the selected run does not overwrite
    # the base run's threshold file.
    os.makedirs(MODEL_DIR, exist_ok=True)
    threshold_file = "threshold_selected.json" if selected else "threshold.json"
    threshold_data = {"model": model_name, "threshold": best_threshold, "auc": float(auc)}
    with open(os.path.join(MODEL_DIR, threshold_file), "w") as f:
        json.dump(threshold_data, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    ax.scatter(fpr[ix], tpr[ix], color="red", zorder=5,
               label=f"Best threshold = {best_threshold:.3f}")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, _filename("roc_curve", selected))


# ── 4. Precision-Recall Curve ─────────────────────────────────────────────────

def _plot_precision_recall(y_test, y_proba, model_name: str, selected: bool) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2ca02c", lw=2.5)
    ax.fill_between(recall, precision, alpha=0.08, color="#2ca02c")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    plt.tight_layout()
    _save(fig, _filename("precision_recall_curve", selected))


# ── 5. Feature Importance ─────────────────────────────────────────────────────

def _plot_feature_importance(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    selected: bool,
    eval_scaler: StandardScaler = None,
) -> None:
    """
    Plot feature importance.

    For tree-based models  → native feature_importances_ (fast, no data needed).
    For SVM / LR           → permutation importance on SCALED X_test.
                             eval_scaler must be supplied; without it the model
                             receives unscaled data and predictions are invalid.
    """
    feature_names = X_test.columns.tolist()

    if hasattr(model, "feature_importances_"):
        # Tree-based: use built-in importance — no scaling needed
        importances = model.feature_importances_
    else:
        # SVM / LR were trained on scaled data → transform X_test first
        if eval_scaler is not None:
            X_for_perm = eval_scaler.transform(X_test)
        else:
            # Fallback: fit a fresh scaler on X_test (last resort — log a warning)
            print("[Evaluate] ⚠  eval_scaler not supplied for permutation importance. "
                  "Fitting a scaler on X_test as fallback (may not match training scale).")
            X_for_perm = StandardScaler().fit_transform(X_test)

        result      = permutation_importance(
            model, X_for_perm, y_test,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
        )
        importances = result.importances_mean

    # Build a Series so feature names stay aligned after sorting
    importance_series  = pd.Series(importances, index=feature_names).sort_values()
    sorted_features    = importance_series.index.tolist()
    sorted_importances = importance_series.values

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))
    bars   = ax.barh(sorted_features, sorted_importances, color=colors)

    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
        )

    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    _save(fig, _filename("feature_importance", selected))


# ── Master Evaluate Function ──────────────────────────────────────────────────

def _run_evaluation(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    selected: bool,
    label: str,
    eval_scaler: StandardScaler = None,
) -> None:
    """
    Core evaluation logic shared by `evaluate` and `evaluate_selected`.

    Parameters
    ----------
    model        : fitted estimator returned by train().
    model_name   : string name of the best model.
    X_test       : raw (unscaled) test features as a DataFrame.
    y_test       : true labels.
    selected     : True when evaluating a feature-selected variant
                   (controls file-name suffixes).
    label        : human-readable label used in the printed report header.
    eval_scaler  : StandardScaler fitted on X_train during training.
                   Passed through to _plot_feature_importance so that
                   SVM / LR permutation importance uses correctly scaled data.
    """
    # Scale X_test before predicting if the model requires it
    X_input = (eval_scaler.transform(X_test)
               if (model_name in NEEDS_SCALING and eval_scaler is not None)
               else X_test)

    y_pred  = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]

    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT{f' ({label})' if label else ''}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"]))
    print("[Evaluate] Generating visualizations ...\n")

    _plot_metric_summary    (y_test, y_pred, y_proba, model_name, selected)
    _plot_confusion_matrix  (y_test, y_pred,          model_name, selected)
    _plot_roc_curve         (y_test, y_proba,         model_name, selected)
    _plot_precision_recall  (y_test, y_proba,         model_name, selected)
    # Pass eval_scaler so SVM / LR permutation importance is computed correctly
    _plot_feature_importance(model, model_name, X_test, y_test, selected,
                             eval_scaler=eval_scaler)

    print(f"\n[Evaluate] All charts saved to /{REPORT_DIR}/")


def evaluate(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    eval_scaler: StandardScaler = None,
) -> None:
    """Evaluate a model trained on the full feature set."""
    _run_evaluation(model, model_name, X_test, y_test,
                    selected=False, label="", eval_scaler=eval_scaler)


def evaluate_selected(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    eval_scaler: StandardScaler = None,
) -> None:
    """Evaluate a model trained on the selected feature subset."""
    _run_evaluation(model, model_name, X_test, y_test,
                    selected=True, label="selected", eval_scaler=eval_scaler)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader         import load_data, get_selected_features
    from feature_engineering import engineer_features, engineer_features_selected
    from preprocess          import preprocess, preprocess_selected
    from train               import train
    from train_selected      import train_selected

    # ── Load & engineer features ──────────────────────────────────────────────
    df = load_data()
    df = engineer_features(df)

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = preprocess(df)

    # ── Train ─────────────────────────────────────────────────────────────────
    # train() returns exactly 4 values: best_model, best_name, comparison_df, cv_results
    best_model, best_name, _, _ = train(X_train, X_test, y_train, y_test)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # NOTE: eval_scaler is optional here. If best_model is SVM or Logistic
    # Regression, pass the scaler from train() for correct permutation importance.
    # For tree-based models it is not needed and can be omitted.
    evaluate(best_model, best_name, X_test, y_test)