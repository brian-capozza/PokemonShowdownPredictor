import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from main import main

# ---- SETTINGS ----
FRACTION = 0.50
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. LOAD AND PREPROCESS DATA ----
df = main()

weather_cols = [col for col in df.columns if col.startswith('weather_')]
feature_cols = [
    'move_p1_0', 'move_p1_1', 'move_p2_0', 'move_p2_1',
    'active_p1_0', 'active_p1_1', 'active_p2_0', 'active_p2_1',
    'switch_p1', 'switch_p2'
] + weather_cols

groups = df.groupby('game')
seqs, lengths, labels = [], [], []

for game_id, group in groups:
    group_sorted = group.sort_values('turn')
    n_turns = len(group_sorted)
    use_n = max(1, int(n_turns * FRACTION))   # at least 1 turn
    X_seq = group_sorted.iloc[:use_n][feature_cols].values      # (seq_len, num_features)
    label = group_sorted.iloc[0]['p1_win']
    seqs.append(X_seq)
    lengths.append(use_n)
    labels.append(label)

max_seq_len = max(lengths)
num_features = len(feature_cols)

# Pad all sequences to max_seq_len
X_all = np.zeros((len(seqs), max_seq_len, num_features), dtype=np.float32)
for idx, (seq, l) in enumerate(zip(seqs, lengths)):
    X_all[idx, :l, :] = seq
y_all = np.array(labels).astype(np.float32)
lengths = np.array(lengths)

# Train/test split
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    X_all, y_all, lengths, test_size=0.2, random_state=42
)

# ---- 2. DATASET & DATALOADER ----
class GameDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

train_loader = DataLoader(GameDataset(X_train, y_train, lengths_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(GameDataset(X_test, y_test, lengths_test), batch_size=BATCH_SIZE)

# ---- 3. MODEL ----
class WinnerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        # lengths: (batch,)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, _) = self.lstm(packed_x)
        out = hn[-1, :, :]   # last layer's hidden state
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

model = WinnerLSTM(input_dim=num_features, hidden_dim=64).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- 4. TRAINING LOOP ----
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch, lengths_batch in train_loader:
        X_batch, y_batch, lengths_batch = X_batch.to(DEVICE), y_batch.to(DEVICE), lengths_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch, lengths_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

# ---- 5. EVALUATION ----
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch, lengths_batch in test_loader:
        X_batch, lengths_batch = X_batch.to(DEVICE), lengths_batch.to(DEVICE)
        outputs = model(X_batch, lengths_batch)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.2f}")

# ---- 6. PREDICT ON NEW GAME ----
def predict_winner(turns_array):
    """
    turns_array: numpy array of shape (seq_len, num_features)
    Returns predicted label: 1 (p1 wins) or 0
    """
    seq_len = turns_array.shape[0]
    pad_array = np.zeros((max_seq_len, num_features), dtype=np.float32)
    pad_array[:seq_len] = turns_array
    tensor_in = torch.tensor(pad_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    length_in = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(tensor_in, length_in)
        pred = int((output > 0.5).item())
    return pred

# --- Example usage:
# test_seq = X_test[0][:lengths_test[0]] # One game's sequence
# print("Winner prediction for test game:", predict_winner(test_seq))