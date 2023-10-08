import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_utils import *

# Function to filter data based on conditions
def filter_indices(df):
    neg_bz = df["BZ, nT (GSM)"] < 0
    
    # Conditions
    class0 = neg_bz
    class1 = ~neg_bz
    
    # Getting indices
    class0_ind = np.where(class0)[0]
    class1_ind = np.where(class1)[0]
    
    return class0_ind, class1_ind

def find_sequential_data(df: pd.DataFrame):
    sequences = []
    hour_dropped = df.drop(["Hour"], axis=1)

    current_sequence = [hour_dropped.iloc[0]]
    
    for i in range(1, len(df)):
        hour_diff = df.iloc[i]['Hour'] - df.iloc[i - 1]['Hour']
        is_sequential = hour_diff == 1 or (df.iloc[i - 1]['Hour'] == 23 and df.iloc[i]['Hour'] == 0)
        
        if is_sequential:
            current_sequence.append(hour_dropped.iloc[i])
        else:
            sequences.append(pd.DataFrame(current_sequence))
            current_sequence = [hour_dropped.iloc[i]]
    
    sequences.append(pd.DataFrame(current_sequence))  # append the last sequence
    return sequences

def create_subsequences(sequence, subseq_length):
    if subseq_length > len(sequence):
        return []
    
    subsequences = [sequence[i:i + subseq_length] for i in range(len(sequence) - subseq_length + 1)]
    return subsequences

selected_columns = [
    "Hour",
    # "BX, nT (GSE, GSM)",
    # "BY, nT (GSM)",
    "BZ, nT (GSM)",
    "SW Plasma Speed, kms",
    "SW Plasma flow long. angle",
    "SW Plasma flow lat. angle",
    "E elecrtric field",
    "Plasma Beta",
    "Alfen mach number",
    "Dst-index, nT"
]

df = read_multi_data(range(2022, 2024), replace_fillers=True, keep_columns=selected_columns, drop_na_values=True)

df = df[selected_columns]

ind_cls_0, ind_cls_1 = filter_indices(df)

labels = np.full((len(df), ), 0)
labels[ind_cls_1] = 1
labels = labels.astype(int)

df["Label"] = labels
df.drop(["BZ, nT (GSM)"], inplace=True, axis=1)

sequences = find_sequential_data(df)

subsequences = []
labels = []

for idx, seq in enumerate(sequences):
    for sub_idx, subseq in enumerate(create_subsequences(seq, 31)):
        if len(subseq) != 0:
            subsequences.append(subseq[:-1])
            labels.append(subseq[-1:]["Label"])

# Convert subsequences and labels to numpy arrays
X = np.array([subseq.drop(columns=["Label"]).values for subseq in subsequences])
y = np.array([int(label.iloc[0]) for label in labels])  # Taking the mode as the label for each subsequence

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define the loss function and the optimizer
model = LSTMClassifier(input_size=X_train.shape[2], hidden_size=50, num_layers=2, num_classes=len(np.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent step
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Test the trained model (optional)
with torch.no_grad():
    test_data = X_test
    test_labels = y_test
    scores = model(test_data)
    predictions = torch.argmax(scores, dim=1)
    num_correct = (predictions == test_labels).sum()
    accuracy = float(num_correct) / float(test_labels.shape[0])
    print(f'Accuracy: {accuracy*100:.2f}%')

    torch.save(model.state_dict(), f"weights/single_cls_{num_epochs}_epoch_{f'{accuracy*100:.2f}'.replace('.', '_')}.pth")