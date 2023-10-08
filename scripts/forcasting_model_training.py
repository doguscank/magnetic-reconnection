import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_utils import *

# We'll continue using these selected columns but will adapt the data preparation for forecasting
selected_columns = [
    "Hour",
    "BX, nT (GSE, GSM)",
    "BY, nT (GSM)",
    "BZ, nT (GSM)",
    "SW Plasma Speed, kms",
    "SW Plasma flow long. angle",
    "SW Plasma flow lat. angle",
    "E elecrtric field",
    "Plasma Beta",
    "Alfen mach number",
    "Dst-index, nT"
]

df = read_multi_data(range(2020, 2024), replace_fillers=True, keep_columns=selected_columns, drop_na_values=True)
df = df[selected_columns]

# Function to find sequential data based on the 'Hour' column
def find_sequential_data(df):
    sequences = []
    current_sequence = [df.iloc[0]]

    for i in range(1, len(df)):
        hour_diff = df.iloc[i]['Hour'] - df.iloc[i - 1]['Hour']
        is_sequential = hour_diff == 1 or (df.iloc[i - 1]['Hour'] == 23 and df.iloc[i]['Hour'] == 0)

        if is_sequential:
            current_sequence.append(df.iloc[i])
        else:
            sequences.append(pd.DataFrame(current_sequence))
            current_sequence = [df.iloc[i]]

    sequences.append(pd.DataFrame(current_sequence))  # append the last sequence
    return sequences

# Finding sequential data
sequences = find_sequential_data(df)

# Function to create sequences for forecasting from the sequential data
def create_sequences(sequential_data, seq_length):
    sequences = []
    targets = []

    for seq_data in sequential_data:
        data = seq_data.drop(columns=['Hour']).values
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            label = data[i+seq_length]
            sequences.append(seq)
            targets.append(label)

    return np.array(sequences), np.array(targets)

# Creating sequences of length 24 for forecasting
seq_length = 24
X, y = create_sequences(sequences, seq_length)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# Modify the LSTM model for regression
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define the loss function, and the optimizer
model = LSTMRegressor(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=y_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "weights/20_epoch_forecast.pth")
