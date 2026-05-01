"""
Data loading module for water potability dataset.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath="data/kaggle_water_quality.csv"):
    """
    Load water potability dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with water quality features and potability target
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def get_data_info(df):
    """Get comprehensive data information."""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': df.describe().to_dict()
    }
    return info


def get_selected_features(df, features=None, include_target=True):
    """
    Return a DataFrame with only selected features.

    Args:
        df: Input DataFrame (from load_data)
        features: List of feature column names to select.
                  Defaults to ['ph', 'Solids', 'Turbidity'].
        include_target: Whether to include the 'Potability' target column (default True)

    Returns:
        DataFrame containing only the selected feature columns (+ Potability if requested)
    """
    if features is None:
        features = ['ph', 'Solids', 'Turbidity']

    # Validate that requested columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}. "
                         f"Available columns: {list(df.columns)}")

    selected_cols = features.copy()
    if include_target:
        if 'Potability' not in df.columns:
            logger.warning("'Potability' column not found; skipping target inclusion.")
        else:
            selected_cols.append('Potability')

    subset_df = df[selected_cols].copy()
    logger.info(f"Selected features: {features} | include_target={include_target}")
    logger.info(f"Resulting shape: {subset_df.shape}")
    return subset_df


if __name__ == "__main__":
    df = load_data()
    
    print("="*60)
    print("DATA LOADING REPORT")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nMissing Percentage:")
    print((df.isnull().sum() / len(df) * 100).round(2))
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"\nClass Distribution:")
    print(df['Potability'].value_counts())