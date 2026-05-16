"""
impute_features.py
══════════════════
Fills the 6 missing features in your custom water-quality dataset
(Hardness, Chloramines, Sulfate, Conductivity, Organic_carbon,
Trihalomethanes) using the Kaggle dataset as a reference library.

WHY KNN IMPUTATION (not regression)?
─────────────────────────────────────
  The correlation between your 3 available features (ph, Solids,
  Turbidity) and the 6 missing ones is near-zero in the Kaggle
  dataset.  A regression model would just predict the global mean
  for every row — useless.

  KNN imputation is better: it finds the K most similar rows in
  Kaggle (by ph + scaled_Solids + Turbidity) and borrows their
  values.  The result is statistically representative, even when
  direct linear correlation is weak.

SOLIDS SCALE MISMATCH FIX
──────────────────────────
  Your instrument:  32 – 596  (TDS sensor, likely in mg/L × 0.01
                               or a normalised reading)
  Kaggle dataset:   320 – 61 227  (TDS in mg/L)

  Fix: Quantile mapping — each custom Solids value is mapped to the
  same percentile rank inside the Kaggle Solids distribution.  This
  preserves the relative ordering of your readings without needing
  to know the exact unit conversion factor.

USAGE
─────
  python impute_features.py \
      --kaggle  kaggle_water_quality.csv \
      --custom  custom_data.csv \
      --output  custom_data_completed.csv \
      --k       10
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ── Column names ───────────────────────────────────────────────────────────────
AVAILABLE_FEATURES = ["ph", "Solids", "Turbidity"]
MISSING_FEATURES   = [
    "Hardness", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
]
ALL_FEATURES = AVAILABLE_FEATURES + MISSING_FEATURES
TARGET_COL   = "Potability"


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load datasets
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = "data/kaggle_water_quality.csv",
              custom_path: str = "data/custom_data.csv"):
    kaggle = pd.read_csv(filepath)
    custom = pd.read_csv(custom_path)

    print("=" * 60)
    print("  DATA LOADED")
    print("=" * 60)
    print(f"  Kaggle  : {kaggle.shape[0]:,} rows × {kaggle.shape[1]} columns")
    print(f"  Custom  : {custom.shape[0]:,} rows × {custom.shape[1]} columns")
    print(f"\n  Custom columns  : {custom.columns.tolist()}")
    print(f"  Missing to fill : {MISSING_FEATURES}")

    # Validate custom has the 3 available features
    for col in AVAILABLE_FEATURES:
        if col not in custom.columns:
            raise ValueError(f"Custom dataset is missing required column: '{col}'")

    return kaggle, custom


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Scale Solids to match Kaggle distribution (quantile mapping)
# ══════════════════════════════════════════════════════════════════════════════

def scale_solids(kaggle_solids: pd.Series, custom_solids: pd.Series) -> np.ndarray:
    """
    Map custom Solids values onto the Kaggle Solids distribution using
    percentile (quantile) matching.

    Example:
      Your value 280 is at the 50th percentile of your data.
      The 50th percentile of Kaggle Solids is ~20 928.
      So your 280 becomes 20 928 in the transformed space.

    This is the most robust approach when the units/instrument range differ
    and the exact conversion factor is unknown.
    """
    # qt = QuantileTransformer(
    #     output_distribution="normal",
    #     n_quantiles=min(1000, len(kaggle_solids)),
    #     random_state=42,
    # )
    # qt.fit(kaggle_solids.values.reshape(-1, 1))

    # Map custom → uniform quantile → same kaggle quantile space
    custom_uniform = np.interp(
        custom_solids.values,
        np.percentile(custom_solids.values, np.linspace(0, 100, 1000)),
        np.percentile(kaggle_solids.values,  np.linspace(0, 100, 1000)),
    )

    os.makedirs("data", exist_ok=True)
    np.save("data/solids_src_percentiles.npy",
            np.percentile(custom_solids.values, np.linspace(0, 100, 1000)))
    np.save("data/solids_dst_percentiles.npy",
            np.percentile(kaggle_solids.values,  np.linspace(0, 100, 1000)))
    
    print("\n" + "=" * 60)
    print("  SOLIDS SCALE MAPPING")
    print("=" * 60)
    print(f"  Custom  Solids range : {custom_solids.min():.1f} – {custom_solids.max():.1f}")
    print(f"  Kaggle  Solids range : {kaggle_solids.min():.1f} – {kaggle_solids.max():.1f}")
    print(f"  Mapped  Solids range : {custom_uniform.min():.1f} – {custom_uniform.max():.1f}")
    print(f"  Kaggle  Solids mean  : {kaggle_solids.mean():.1f}")
    print(f"  Mapped  Solids mean  : {custom_uniform.mean():.1f}")

    return custom_uniform


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Build KNN reference from Kaggle
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_reference(kaggle: pd.DataFrame, k: int):
    """
    Prepares the Kaggle dataset as a KNN lookup library.

    Only rows with all 3 available features present are used as
    reference points (Kaggle has 491 missing ph values — these are
    dropped here).

    Returns: fitted NearestNeighbors, reference feature matrix,
             reference target matrix (the 6 missing features).
    """
    # Drop Kaggle rows missing any of the 3 lookup features
    kaggle_clean = kaggle.dropna(subset=AVAILABLE_FEATURES + MISSING_FEATURES).copy()

    print("\n" + "=" * 60)
    print("  KNN REFERENCE LIBRARY")
    print("=" * 60)
    print(f"  Kaggle rows with all features present : {len(kaggle_clean):,}")
    print(f"  K neighbours used per custom row      : {k}")

    X_ref = kaggle_clean[AVAILABLE_FEATURES].values   # lookup features
    Y_ref = kaggle_clean[MISSING_FEATURES].values     # features to borrow

    # Standardise X_ref so no single feature dominates by scale
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(X_ref_scaled)

    return knn, scaler, X_ref_scaled, Y_ref, kaggle_clean


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Impute missing features for each custom row
# ══════════════════════════════════════════════════════════════════════════════

def impute(custom: pd.DataFrame,
           scaled_solids: np.ndarray,
           knn: NearestNeighbors,
           scaler: StandardScaler,
           Y_ref: np.ndarray,
           k: int) -> pd.DataFrame:
    """
    For every row in custom:
      1. Build lookup vector [ph, scaled_Solids, Turbidity]
      2. Find K nearest rows in Kaggle (by Euclidean distance after scaling)
      3. Average their 6 missing feature values → imputed values for this row

    Distance-weighted averaging is used: closer neighbours contribute more.
    """
    print("\n" + "=" * 60)
    print("  IMPUTING MISSING FEATURES")
    print("=" * 60)

    # Build the custom lookup matrix using scaled Solids
    custom_lookup = custom[AVAILABLE_FEATURES].copy()
    custom_lookup["Solids"] = scaled_solids

    X_custom = custom_lookup.values
    X_custom_scaled = scaler.transform(X_custom)   # same scaler as Kaggle

    # Find K nearest Kaggle neighbours for every custom row
    distances, indices = knn.kneighbors(X_custom_scaled)

    # Distance-weighted average of neighbour feature values
    # weight = 1 / (distance + 1e-6) to avoid divide-by-zero
    weights = 1.0 / (distances + 1e-6)                    # (N, k)
    weights /= weights.sum(axis=1, keepdims=True)          # normalise

    # Weighted average: (N, k) @ (k, 6) for each row  → (N, 6)
    imputed_values = np.einsum("nk,nkf->nf",
                               weights,
                               Y_ref[indices])             # (N, k, 6)

    imputed_df = pd.DataFrame(imputed_values,
                              columns=MISSING_FEATURES,
                              index=custom.index)

    # Assemble completed dataset
    completed = custom.copy()
    for col in MISSING_FEATURES:
        completed[col] = imputed_df[col].round(6)

    # Reorder columns to match Kaggle
    final_cols = [c for c in kaggle_column_order() if c in completed.columns]
    completed = completed[final_cols]

    print(f"  ✓ Imputed {len(MISSING_FEATURES)} features for {len(custom)} rows")
    print("\n  Sample imputed values (first 3 rows):")
    print(completed[MISSING_FEATURES].head(3).to_string(index=False))

    return completed


def kaggle_column_order():
    return ["ph", "Solids", "Turbidity", "Hardness","Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes",
             "Potability"]


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Validation summary
# ══════════════════════════════════════════════════════════════════════════════

def validation_summary(kaggle: pd.DataFrame, completed: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  VALIDATION — IMPUTED vs KAGGLE DISTRIBUTIONS")
    print("=" * 60)
    print(f"  {'Feature':<20} {'Kaggle Mean':>12} {'Imputed Mean':>12} "
          f"{'Kaggle Std':>11} {'Imputed Std':>11}")
    print("  " + "-" * 68)
    for col in MISSING_FEATURES:
        km = kaggle[col].mean()
        im = completed[col].mean()
        ks = kaggle[col].std()
        is_ = completed[col].std()
        flag = "  ✓" if abs(km - im) / (km + 1e-9) < 0.15 else "  ⚠"
        print(f"  {col:<20} {km:>12.2f} {im:>12.2f} {ks:>11.2f} {is_:>11.2f}{flag}")

    print("\n  ✓ = imputed mean within 15% of Kaggle mean (good)")
    print("  ⚠ = imputed mean deviates >15% (check Solids scaling)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(filepath: str = "data/kaggle_water_quality.csv",
        custom_path: str = "data/custom_data.csv",
        k: int = 10):

    # 1. Load
    kaggle, custom = load_data(filepath=filepath, custom_path=custom_path)

    # 2. Scale Solids
    scaled_solids = scale_solids(kaggle["Solids"], custom["Solids"])

    # 3. Build KNN reference
    knn, scaler, X_ref_scaled, Y_ref, kaggle_clean = build_knn_reference(kaggle, k)

    # 4. Impute
    completed = impute(custom, scaled_solids, knn, scaler, Y_ref, k)

    # 5. Validate
    validation_summary(kaggle, completed)

    # 6. Save — always write to data/custom_dataset_withAllF.csv
    output_path = os.path.join("data", "custom_dataset_withAllF.csv")
    os.makedirs("data", exist_ok=True)          # create data/ folder if it doesn't exist
    completed.to_csv(output_path, index=False)
    print(f"\n  ✓ Completed dataset saved → {output_path}")
    print(f"    Shape: {completed.shape[0]} rows × {completed.shape[1]} columns")

    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Impute missing water quality features")
    parser.add_argument("--filepath",   default="data/kaggle_water_quality.csv")
    parser.add_argument("--custom_path", default="data/custom_data.csv")
    parser.add_argument("--k",           type=int, default=10,
                        help="Number of KNN neighbours (default: 10)")
    args = parser.parse_args()

    run(filepath=args.filepath, custom_path=args.custom_path, k=args.k)
