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


# ── Feature Engineering (Selected Features: ph, Solids, Turbidity) ────────────

def engineer_features_selected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering scoped strictly to the selected-feature DataFrame
    (ph, Solids, Turbidity) returned by get_selected_features().

    Only interactions derivable from these three columns are created:
      - ph_category      : WHO-based acid / neutral / alkaline label (mirrors full pipeline)
      - ph_deviation     : absolute distance from the ideal neutral pH of 7.0
      - solids_turbidity : interaction — high solids + high turbidity = strong contamination signal
      - ph_turbidity     : turbidity impact modulated by pH (acidic water carries more particles)
      - solids_log       : log-transform of Solids to reduce its heavy right skew

    Called BEFORE preprocess_selected() (on the raw selected-feature DataFrame).

    Args:
        df : DataFrame with columns ph, Solids, Turbidity (+ optionally Potability).

    Returns:
        DataFrame with 5 additional engineered columns.
    """
    df = df.copy()

    # 1. pH Category — identical to full pipeline for consistency
    df["ph_category"] = pd.cut(
        df["ph"],
        bins=[0, 6.5, 8.5, 14],
        labels=[0, 1, 2],       # 0=acidic, 1=safe/neutral, 2=alkaline
        include_lowest=True
    ).astype(float)

    # 2. pH Deviation from neutral (7.0) — captures how far water strays from safe range
    df["ph_deviation"] = (df["ph"] - 7.0).abs()

    # 3. Solids × Turbidity interaction — both high together = strong contamination signal
    df["solids_turbidity"] = df["Solids"] * df["Turbidity"] / 1e5

    # 4. pH × Turbidity — acidic/alkaline conditions intensify turbidity effects
    df["ph_turbidity"] = df["ph"] * df["Turbidity"]

    # 5. Log-transform Solids — its distribution is heavily right-skewed; log brings it closer to normal
    df["solids_log"] = df["Solids"].apply(lambda x: pd.NA if pd.isna(x) else max(x, 1)).pipe(
        lambda s: s.astype(float)
    )
    df["solids_log"] = np.log1p(df["Solids"].clip(lower=0))

    new_features = [
        "ph_category", "ph_deviation",
        "solids_turbidity", "ph_turbidity", "solids_log"
    ]
    print(f"[FeatureEng-Selected] Added {len(new_features)} features : {new_features}")
    print(f"[FeatureEng-Selected] Total columns now : {df.shape[1]}")

    return df

# ── Quick Test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from data_loader import load_data, get_selected_features

    df = load_data()

    # Full pipeline
    print("=" * 55)
    print("FULL PIPELINE FEATURE ENGINEERING")
    print("=" * 55)
    df_full = engineer_features(df)
    print(df_full.head(3))

    # Selected-feature pipeline
    print("\n" + "=" * 55)
    print("SELECTED-FEATURE ENGINEERING  (ph, Solids, Turbidity)")
    print("=" * 55)
    df_sel = get_selected_features(df, include_target=True)
    df_sel = engineer_features_selected(df_sel)
    print(df_sel.head(3))