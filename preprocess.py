import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    "ph_category", "hardness_solids", "chloramine_ph",
    "conductivity_ratio", "turbidity_organic"
]
TARGET = "Potability"
SCALER_PATH  = "models/scaler.pkl"
IMPUTER_PATH = "models/imputer.pkl"

# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Steps:
      1. Remove duplicate rows
      2. Remove outliers using IQR
      3. Impute missing values with KNN (better than mean for correlated features)
      4. Scale features with RobustScaler (handles remaining outliers well)
      5. Stratified train/test split
      6. Save scaler + imputer for inference use
    """

    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Preprocess] Duplicates removed : {before - len(df)}")

    # 2. Remove outliers (IQR method per feature)
    df = _remove_outliers(df)

    # 3. Split X / y
    X = df[FEATURES]
    y = df[TARGET]
#4 Split FIRST on raw data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y.values, test_size=test_size, random_state=random_state, stratify=y
)

#5 Fit imputer only on train, transform both
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=FEATURES)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw),      columns=FEATURES)

#6 Fit scaler only on train, transform both
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=FEATURES)
    X_test  = pd.DataFrame(scaler.transform(X_test_imp),      columns=FEATURES)

    # 7. Save imputer + scaler for use during inference / IoT prediction
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    print(f"[Preprocess] Scaler  saved → {SCALER_PATH}")
    print(f"[Preprocess] Imputer saved → {IMPUTER_PATH}")

    print(f"[Preprocess] Train : {X_train.shape[0]} samples")
    print(f"[Preprocess] Test  : {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# ── Outlier Removal ───────────────────────────────────────────────────────────

def _remove_outliers(df: pd.DataFrame, factor: float = 5.0) -> pd.DataFrame:
    """
    Remove rows where any feature value is beyond factor * IQR.
    factor=5.0 is conservative — removes only extreme outliers,
    keeping enough data for high accuracy.
    """
    before = len(df)
    for col in FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]
    print(f"[Preprocess] Outliers removed   : {before - len(df)}")
    return df.reset_index(drop=True)


# # ── Inference Helper ──────────────────────────────────────────────────────────

# def preprocess_single(sample: dict) -> np.ndarray:
#     """
#     Prepare a single IoT sensor reading for prediction.
#     Uses the saved imputer + scaler (must run preprocess() first).
#     """
#     imputer = joblib.load(IMPUTER_PATH)
#     scaler  = joblib.load(SCALER_PATH)

#     df = pd.DataFrame([sample])[FEATURES]
#     df_imputed = imputer.transform(df)
#     df_scaled  = scaler.transform(df_imputed)

#     return df_scaled


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    print(f"\nX_train sample:\n{X_train.head(3)}")