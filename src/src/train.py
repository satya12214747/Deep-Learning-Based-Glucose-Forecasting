"""
Training script for LSTM-based glucose forecasting.

This script defines the end-to-end training pipeline used in the project:
- Data loading (placeholder)
- Feature engineering (placeholder)
- Temporal train/validation split
- LSTM model definition
- Model training and evaluation

NOTE:
Actual dataset loading and feature engineering logic will be added later.
This file is intentionally structured to reflect a real-world ML training pipeline.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
LOOKBACK_WINDOW = 24  # number of past timesteps used for prediction

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Data Loading (Placeholder)
# -----------------------------
def load_data():
    """
    Placeholder function for loading CGM time-series data.

    Returns
    -------
    X : np.ndarray
        Input sequences of shape (num_samples, timesteps, num_features)
    y : np.ndarray
        Target glucose values
    """
    print("[INFO] Loading data (placeholder)...")

    # Dummy data to keep the pipeline runnable
    X = np.random.rand(100, LOOKBACK_WINDOW, 5)
    y = np.random.rand(100)

    return X, y


# -----------------------------
# Temporal Train/Validation Split
# -----------------------------
def temporal_split(X, y, train_ratio=0.8):
    """
    Perform chronological (time-aware) train-validation split.
    """
    split_idx = int(len(X) * train_ratio)

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, X_val, y_train, y_val


# -----------------------------
# Model Definition
# -----------------------------
def build_lstm_model(input_shape):
    """
    Build LSTM model for glucose forecasting.
    """
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mae"
    )

    return model


# -----------------------------
# Training Pipeline
# -----------------------------
def train():
    print("[INFO] Starting training pipeline...")

    # Load data
    X, y = load_data()

    # Temporal split
    X_train, X_val, y_train, y_val = temporal_split(X, y)

    # Build model
    model = build_lstm_model(input_shape=X_train.shape[1:])
    model.summary()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save model
    os.makedirs("results", exist_ok=True)
    model.save("results/lstm_glucose_model")

    print("[INFO] Training complete. Model saved to results/ directory.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
