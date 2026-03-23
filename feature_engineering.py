import pandas as pd

# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single responsibility: add domain-relevant features to boost model accuracy.
    Called BEFORE preprocess.py (on raw loaded data).

    New features added:
      - ph_category        : acid / neutral / alkaline classification
      - hardness_solids    : interaction between hardness and dissolved solids
      - chloramine_ph      : chloramine effectiveness depends on pH
      - conductivity_ratio : conductivity normalized by solids (ion concentration proxy)
      - turbidity_organic  : combined organic pollution indicator
    """

    df = df.copy()

    # 1. pH Category (WHO safe range: 6.5 - 8.5)
    df["ph_category"] = pd.cut(
        df["ph"],
        bins=[0, 6.5, 8.5, 14],
        labels=[0, 1, 2],        # 0=acidic, 1=safe, 2=alkaline
        include_lowest=True
    ).astype(float)              # float keeps NaN compatible with imputer

    # 2. Hardness × Solids interaction (high both = very hard water)
    df["hardness_solids"] = df["Hardness"] * df["Solids"] / 1e6

    # 3. Chloramine × pH interaction (chloramine efficacy drops at extreme pH)
    df["chloramine_ph"] = df["Chloramines"] * df["ph"]

    # 4. Conductivity per unit Solids (measures ion concentration)
    df["conductivity_ratio"] = df["Conductivity"] / (df["Solids"] + 1)

    # 5. Turbidity × Organic Carbon (combined pollution signal)
    df["turbidity_organic"] = df["Turbidity"] * df["Organic_carbon"]

    new_features = [
        "ph_category", "hardness_solids",
        "chloramine_ph", "conductivity_ratio", "turbidity_organic"
    ]
    print(f"[FeatureEng] Added {len(new_features)} features : {new_features}")
    print(f"[FeatureEng] Total columns now : {df.shape[1]}")

    return df


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df = engineer_features(df)
    print(df.head(3))