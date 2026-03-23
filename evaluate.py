import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

REPORT_DIR = "reports"
PLOT_STYLE = "seaborn-v0_8-darkgrid"

os.makedirs(REPORT_DIR, exist_ok=True)
plt.style.use(PLOT_STYLE)


# ── 1. Metric Summary Card ────────────────────────────────────────────────────

def plot_metric_summary(y_test, y_pred, y_proba, model_name: str):
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

    for i, (metric, value) in enumerate(metrics.items()):
        x = 0.1 + i * 0.19
        color = colors[i]
        ax.add_patch(plt.Rectangle(
            (x - 0.08, 0.15), 0.16, 0.7,
            color=color, alpha=0.15, transform=ax.transAxes
        ))
        ax.text(x, 0.72, metric, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, 0.38, f"{value:.4f}", ha="center", va="center",
                fontsize=16, fontweight="bold", color=color, transform=ax.transAxes)

    ax.set_title(f"Performance Summary — {model_name}", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "metric_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Saved → {path}")


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Potable", "Potable"],
        yticklabels=["Not Potable", "Potable"],
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Saved → {path}")


# ── 3. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_proba, model_name: str):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Saved → {path}")


# ── 4. Precision-Recall Curve ─────────────────────────────────────────────────

def plot_precision_recall(y_test, y_proba, model_name: str):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2ca02c", lw=2.5)
    ax.fill_between(recall, precision, alpha=0.08, color="#2ca02c")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Saved → {path}")


# ── 5. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str):
    feature_names = X_test.columns.tolist()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        result      = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean

    indices            = np.argsort(importances)[::-1]
    sorted_features    = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))[::-1]
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1], color=colors[::-1])

    for bar, val in zip(bars, sorted_importances[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Saved → {path}")


# ── Master Evaluate Function ──────────────────────────────────────────────────

def evaluate(
    model, model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"]))

    print("[Evaluate] Generating visualizations ...\n")

    plot_metric_summary   (y_test, y_pred, y_proba, model_name)
    plot_confusion_matrix (y_test, y_pred,           model_name)
    plot_roc_curve        (y_test, y_proba,          model_name)
    plot_precision_recall (y_test, y_proba,          model_name)
    plot_feature_importance(model, X_test, y_test,   model_name)

    print(f"\n[Evaluate] All charts saved to /{REPORT_DIR}/")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader         import load_data
    from feature_engineering import engineer_features
    from preprocess          import preprocess
    from train               import train

    df                               = load_data()
    df                               = engineer_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    best_model, best_name, _, _      = train(X_train, X_test, y_train, y_test)

    evaluate(best_model, best_name, X_test, y_test)