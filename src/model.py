import torch
import torch.nn as nn


class LSTMGlucoseModel(nn.Module):
    """
    LSTM-based model for glucose forecasting.
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMGlucoseModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Predicted glucose values.
        """
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
