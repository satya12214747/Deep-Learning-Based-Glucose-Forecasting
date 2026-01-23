import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_utils import load_kaggle_glucose_data
from model import LSTMGlucoseModel


def create_sequences(data, seq_length=24):
    """
    Convert time series into input-output sequences.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train(csv_path, epochs=10, seq_length=24, lr=0.001):
    # Load data
    df = load_kaggle_glucose_data(csv_path)
    glucose = df.values.astype(float)

    # Normalize
    mean = glucose.mean()
    std = glucose.std()
    glucose = (glucose - mean) / std

    # Create sequences
    X, y = create_sequences(glucose, seq_length)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Model
    model = LSTMGlucoseModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training completed.")


if __name__ == "__main__":
    # Example usage (update path when running locally)
    train("path_to_kaggle_glucose.csv")
