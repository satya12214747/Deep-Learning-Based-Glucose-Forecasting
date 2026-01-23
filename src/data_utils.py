import pandas as pd
import numpy as np


def load_kaggle_glucose_data(csv_path):
    """
    Load and preprocess CGM glucose data from Kaggle dataset.

    Parameters
    ----------
    csv_path : str
        Path to the Kaggle CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with glucose values indexed by time.
    """
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # Try to identify glucose column
    glucose_cols = [c for c in df.columns if 'glucose' in c]
    if not glucose_cols:
        raise ValueError("No glucose column found in dataset")

    glucose_col = glucose_cols[0]

    # Drop missing values
    df = df.dropna(subset=[glucose_col])

    # Convert glucose values to numeric
    df[glucose_col] = pd.to_numeric(df[glucose_col], errors='coerce')
    df = df.dropna(subset=[glucose_col])

    return df[[glucose_col]]
