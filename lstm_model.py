"""
Leak-safe LSTM win predictor for Pokémon Showdown (Gen 5 OU).

Improvements vs previous version:
- Adds causal, high-value features:
    * Team HP totals + damage per turn + damage_diff.
    * Switch counts + switch_diff (momentum).
    * Team strength from base stats.
    * Rolling (past-only) windows over hp/remaining/damage/switch.
    * Last-move power, last-move type, last-move category for each side.
- Uses dynamic prefix sampling: early epochs see shorter prefixes,
  later epochs see near-full games (curriculum), without leakage.
- Normalizes train loss per batch for sensible logs.
- Stronger regularization + capacity:
    * Larger projection + LSTM hidden dim.
    * Turn-level dropout (randomly zero whole turns).
    * Stochastic depth on self-attention residual.
    * Label smoothing and base-rate bias init.
- Architecture: Projection → Self-Attention → BiLSTM → Attention Pool.
- Produces:
    * Training vs validation loss and accuracy curves.
    * Win probability curves across turns for sample games.
    * Attention vs time for one game.
    * Gradient-based feature importance.
    * Calibration curve + prediction distribution.
"""

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
CSV_PATH = "datasets/gen5ou_games.csv"

BATCH_SIZE = 32
EPOCHS = 60
LR = 8e-4
WEIGHT_DECAY = 1e-4

TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PAD_VALUE = -1.0

PATIENCE = 12        # early stopping on val loss

# Prefix curriculum
MIN_PREFIX_LEN = 3
PREFIX_MIN_FRAC_START = 0.45   # early epochs: shorter prefixes
PREFIX_MIN_FRAC_END = 0.75     # later epochs: near-full games
current_prefix_min_frac = PREFIX_MIN_FRAC_START  # updated each epoch

# Regularization
GRAD_NOISE_STD = 0.003
TURN_DROP_PROB = 0.2          # probability of dropping a whole turn during training
LABEL_SMOOTHING = 0.1         # small label smoothing for training loss

#torch.manual_seed(SEED)
#np.random.seed(SEED)
#random.seed(SEED)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df = df.sort_values(["game_id", "turn"]).reset_index(drop=True)

print(f"Loaded {len(df)} turns across {df['game_id'].nunique()} games.")


# =========================
# HIGH-LEVEL FEATURE ENGINEERING (CAUSAL ONLY)
# =========================

def team_avg_hp_ratio(df, side):
    curr, cap = [], []
    for i in range(1, 7):
        c = f"{side}_slot{i}_current_health"
        m = f"{side}_slot{i}_hp"
        if c in df:
            curr.append(df[c].fillna(0))
        if m in df:
            cap.append(df[m].fillna(1))
    if not curr or not cap:
        return pd.Series(0.0, index=df.index)
    curr_sum = pd.concat(curr, axis=1).sum(axis=1)
    cap_sum = pd.concat(cap, axis=1).sum(axis=1).replace(0, 1.0)
    return curr_sum / cap_sum


def active_stat(df, side, stat):
    vals = []
    for i in range(1, 7):
        v = f"{side}_slot{i}_{stat}"
        a = f"{side}_slot{i}_active"
        if v in df:
            if a in df:
                vals.append(df[v].fillna(0) * df[a].fillna(0))
            else:
                vals.append(df[v].fillna(0))
    if not vals:
        return pd.Series(0.0, index=df.index)
    return pd.concat(vals, axis=1).sum(axis=1)


def team_hp_total(df, side):
    cols = []
    for i in range(1, 7):
        c = f"{side}_slot{i}_current_health"
        if c in df:
            cols.append(df[c].fillna(0))
    if not cols:
        return pd.Series(0.0, index=df.index)
    return pd.concat(cols, axis=1).sum(axis=1)


def team_strength(df, side):
    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    cols = []
    for i in range(1, 7):
        for s in stats:
            c = f"{side}_slot{i}_{s}"
            if c in df:
                cols.append(df[c].fillna(0))
    if not cols:
        return pd.Series(0.0, index=df.index)
    return pd.concat(cols, axis=1).sum(axis=1)


print("Adding high-level momentum / advantage features (causal only)...")

# --- Team HP ratios (instantaneous) ---
df["p1_team_hp_ratio"] = team_avg_hp_ratio(df, "p1")
df["p2_team_hp_ratio"] = team_avg_hp_ratio(df, "p2")
df["team_hp_ratio_diff"] = df["p1_team_hp_ratio"] - df["p2_team_hp_ratio"]

# --- Team HP totals and damage (instantaneous + diff to previous turn) ---
df["p1_team_hp_total"] = team_hp_total(df, "p1")
df["p2_team_hp_total"] = team_hp_total(df, "p2")
df["team_hp_total_diff"] = df["p1_team_hp_total"] - df["p2_team_hp_total"]

df["p1_team_hp_total_prev"] = df.groupby("game_id")["p1_team_hp_total"].shift(1)
df["p2_team_hp_total_prev"] = df.groupby("game_id")["p2_team_hp_total"].shift(1)

df["p1_damage_taken"] = (
    (df["p1_team_hp_total_prev"] - df["p1_team_hp_total"])
    .clip(lower=0)
    .fillna(0)
)
df["p2_damage_taken"] = (
    (df["p2_team_hp_total_prev"] - df["p2_team_hp_total"])
    .clip(lower=0)
    .fillna(0)
)
df["damage_diff"] = df["p2_damage_taken"] - df["p1_damage_taken"]  # >0 means P1 dealt more

# --- Remaining Pokémon (instantaneous) ---
df["p1_remaining"] = 0
df["p2_remaining"] = 0
for i in range(1, 7):
    c1 = f"p1_slot{i}_current_health"
    c2 = f"p2_slot{i}_current_health"
    if c1 in df:
        df["p1_remaining"] += (df[c1].fillna(0) > 0).astype(int)
    if c2 in df:
        df["p2_remaining"] += (df[c2].fillna(0) > 0).astype(int)
df["remaining_diff"] = df["p1_remaining"] - df["p2_remaining"]

# --- Active HP and speed (instantaneous) ---
df["p1_active_hp"] = active_stat(df, "p1", "current_health")
df["p2_active_hp"] = active_stat(df, "p2", "current_health")
df["active_hp_diff"] = df["p1_active_hp"] - df["p2_active_hp"]

df["p1_active_speed"] = active_stat(df, "p1", "spe")
df["p2_active_speed"] = active_stat(df, "p2", "spe")
df["active_speed_diff"] = df["p1_active_speed"] - df["p2_active_speed"]

# --- Boosts (instantaneous) ---
p1_boost_cols = [c for c in df.columns if c.startswith("p1_") and c.endswith("_boost")]
p2_boost_cols = [c for c in df.columns if c.startswith("p2_") and c.endswith("_boost")]
df["p1_boost_sum"] = df[p1_boost_cols].fillna(0).sum(axis=1) if p1_boost_cols else 0
df["p2_boost_sum"] = df[p2_boost_cols].fillna(0).sum(axis=1) if p2_boost_cols else 0
df["boost_diff"] = df["p1_boost_sum"] - df["p2_boost_sum"]

# --- Status counts (instantaneous) ---
status_suffixes = [
    "status_tox", "status_brn", "status_par",
    "status_psn", "status_frz", "status_slp",
]


def status_count(df, side):
    s = 0
    for i in range(1, 7):
        for suf in status_suffixes:
            col = f"{side}_slot{i}_{suf}"
            if col in df:
                s += df[col].fillna(0)
    return s


df["p1_status_count"] = status_count(df, "p1")
df["p2_status_count"] = status_count(df, "p2")
df["status_count_diff"] = df["p1_status_count"] - df["p2_status_count"]

# --- Momentum & change features (only past turns) ---
df["hp_change"] = df.groupby("game_id")["team_hp_ratio_diff"].diff().fillna(0)

df["hp_momentum_3turn"] = (
    df.groupby("game_id")["team_hp_ratio_diff"]
      .rolling(3, min_periods=1)
      .mean()
      .reset_index(0, drop=True)
)

df["pokemon_lost_p1"] = (
    df.groupby("game_id")["p1_remaining"]
      .diff().fillna(0).clip(upper=0).abs()
)
df["pokemon_lost_p2"] = (
    df.groupby("game_id")["p2_remaining"]
      .diff().fillna(0).clip(upper=0).abs()
)
df["recent_ko_advantage"] = df["pokemon_lost_p2"] - df["pokemon_lost_p1"]

df["cumulative_hp_advantage"] = df.groupby("game_id")["team_hp_ratio_diff"].cumsum()
df["cumulative_remaining_advantage"] = df.groupby("game_id")["remaining_diff"].cumsum()

# --- Turn number (safe), but NO game_progress ---
df["turn_number"] = df.groupby("game_id").cumcount() + 1

# --- Active HP percentage and fainted counts (instantaneous) ---
den_p1_hp = active_stat(df, "p1", "hp") + 1e-6
den_p2_hp = active_stat(df, "p2", "hp") + 1e-6
df["p1_active_hp_pct"] = df["p1_active_hp"] / den_p1_hp
df["p2_active_hp_pct"] = df["p2_active_hp"] / den_p2_hp
df["active_hp_pct_diff"] = df["p1_active_hp_pct"] - df["p2_active_hp_pct"]

df["p1_fainted"] = 6 - df["p1_remaining"]
df["p2_fainted"] = 6 - df["p2_remaining"]

# --- Team strength (static per game but repeated per turn, safe) ---
df["p1_team_strength"] = team_strength(df, "p1")
df["p2_team_strength"] = team_strength(df, "p2")
df["team_strength_diff"] = df["p1_team_strength"] - df["p2_team_strength"]

# --- Switch momentum (cumulative, causal) ---
for side in ["p1", "p2"]:
    col = f"{side}_switch"
    if col not in df:
        df[col] = 0
df["p1_switch_cumsum"] = df.groupby("game_id")["p1_switch"].cumsum()
df["p2_switch_cumsum"] = df.groupby("game_id")["p2_switch"].cumsum()
df["switch_diff"] = df["p1_switch_cumsum"] - df["p2_switch_cumsum"]
df["switch_diff_change"] = df.groupby("game_id")["switch_diff"].diff().fillna(0)

# --- Rolling windows over key diffs (past only) ---
roll_specs = [
    ("team_hp_ratio_diff", 3),
    ("remaining_diff", 3),
    ("damage_diff", 3),
    ("switch_diff", 5),
]
for col, w in roll_specs:
    if col in df:
        df[f"{col}_roll{w}"] = (
            df.groupby("game_id")[col]
              .rolling(w, min_periods=1)
              .mean()
              .reset_index(0, drop=True)
        )

# --- Last move features (power, type, category) ---
move_types = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic",
    "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel",
]
move_categories = ["Physical", "Special", "Status"]

for side in ["p1", "p2"]:
    # Base power of last used move this turn
    power_terms = []
    for i in range(1, 7):
        for m in range(1, 4 + 1):
            used_col = f"{side}_slot{i}_move{m}_used"
            power_col = f"{side}_slot{i}_move{m}_base_power"
            if used_col in df and power_col in df:
                power_terms.append(df[used_col].fillna(0) * df[power_col].fillna(0))
    df[f"{side}_last_move_power"] = sum(power_terms) if power_terms else 0

    # Type one-hot for last move
    for t in move_types:
        terms = []
        for i in range(1, 7):
            for m in range(1, 4 + 1):
                used_col = f"{side}_slot{i}_move{m}_used"
                type_col = f"{side}_slot{i}_move{m}_type_{t}"
                if used_col in df and type_col in df:
                    terms.append(df[used_col].fillna(0) * df[type_col].fillna(0))
        df[f"{side}_last_move_type_{t}"] = sum(terms) if terms else 0

    # Category one-hot for last move
    for cat in move_categories:
        terms = []
        for i in range(1, 7):
            for m in range(1, 4 + 1):
                used_col = f"{side}_slot{i}_move{m}_used"
                cat_col = f"{side}_slot{i}_move{m}_category_{cat}"
                if used_col in df and cat_col in df:
                    terms.append(df[used_col].fillna(0) * df[cat_col].fillna(0))
        df[f"{side}_last_move_category_{cat}"] = sum(terms) if terms else 0

# Difference in last-move power
df["last_move_power_diff"] = df["p1_last_move_power"] - df["p2_last_move_power"]

print("✅ High-level features added (no future / final-state dependence).")


# =========================
# FEATURE SELECTION (NO LEAKY COLUMNS)
# =========================

EXCLUDE_COLS = [
    "game_id",
    "p1_win",
    "turn",
    # just in case old runs left these in:
    "game_progress",
    "game_progress_clean",
    "early_game",
    "mid_game",
    "late_game",
    "fainted_ratio",
    "fainted_ratio_diff",
    "active_hp_diff_norm",
    "p1_team_hp_total_prev",
    "p2_team_hp_total_prev",
]

FEATURE_COLS = [
    c for c in df.columns
    if c not in EXCLUDE_COLS and c not in ["Unnamed: 0"]
]

print(f"Using {len(FEATURE_COLS)} features (engineered + raw, leak-safe).")

X_all = df[FEATURE_COLS]
y_all = df["p1_win"].astype(int)
game_ids = df["game_id"]


# =========================
# TRAIN / VAL / TEST SPLIT BY GAME
# =========================

unique_games = game_ids.unique()
np.random.shuffle(unique_games)

n_test = int(len(unique_games) * TEST_SPLIT)
n_val = int(len(unique_games) * VAL_SPLIT)

test_games = set(unique_games[:n_test])
val_games = set(unique_games[n_test:n_test + n_val])
train_games = set(unique_games[n_test + n_val:])

train_mask = game_ids.isin(train_games)
val_mask = game_ids.isin(val_games)
test_mask = game_ids.isin(test_games)

X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]
y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
game_train, game_val, game_test = game_ids[train_mask], game_ids[val_mask], game_ids[test_mask]

print(f"✅ Train games: {len(train_games)}, Val games: {len(val_games)}, Test games: {len(test_games)}")


# =========================
# SCALE FEATURES (TRAIN ONLY)
# =========================

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=FEATURE_COLS,
    index=X_train.index,
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=FEATURE_COLS,
    index=X_val.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURE_COLS,
    index=X_test.index,
)


# =========================
# BUILD PER-GAME SEQUENCES
# =========================

def build_sequences(X_df, y_series, game_series):
    sequences, labels, lengths = [], [], []
    for gid in game_series.unique():
        mask = (game_series == gid)
        seq_arr = X_df[mask].values
        lbl = float(y_series[mask].iloc[-1])  # final label (game outcome)

        seq = torch.tensor(seq_arr, dtype=torch.float32)
        sequences.append(seq)
        labels.append(torch.tensor(lbl, dtype=torch.float32))
        lengths.append(len(seq))

    return sequences, labels, lengths


train_seqs, train_labels, train_lengths = build_sequences(X_train_scaled, y_train, game_train)
val_seqs, val_labels, val_lengths = build_sequences(X_val_scaled, y_val, game_val)
test_seqs, test_labels, test_lengths = build_sequences(X_test_scaled, y_test, game_test)

print(f"✅ Train sequences: {len(train_seqs)}")
print(f"✅ Val sequences:   {len(val_seqs)}")
print(f"✅ Test sequences:  {len(test_seqs)}")


# =========================
# DATASETS & DATALOADERS
# =========================

class BattleDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn_train(batch):
    """
    Collate with dynamic prefix sampling for turn-level supervision.
    Uses module-level current_prefix_min_frac; updated each epoch.
    """
    seqs, labels = zip(*batch)
    new_seqs, new_lengths = [], []

    for seq in seqs:
        full_len = len(seq)
        min_len = max(MIN_PREFIX_LEN, int(current_prefix_min_frac * full_len))
        if min_len >= full_len:
            prefix_len = full_len
        else:
            prefix_len = np.random.randint(min_len, full_len + 1)
        new_seqs.append(seq[:prefix_len])
        new_lengths.append(prefix_len)

    lengths_tensor = torch.tensor(new_lengths, dtype=torch.long)
    padded = pad_sequence(new_seqs, batch_first=True, padding_value=PAD_VALUE)
    labels_tensor = torch.stack(labels)

    return padded, lengths_tensor, labels_tensor


def collate_fn_eval(batch):
    """
    Collate without prefix sampling: full games.
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_VALUE)
    labels_tensor = torch.stack(labels)
    return padded, lengths, labels_tensor


train_loader = DataLoader(
    BattleDataset(train_seqs, train_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn_train,
)
val_loader = DataLoader(
    BattleDataset(val_seqs, val_labels),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_eval,
)
test_loader = DataLoader(
    BattleDataset(test_seqs, test_labels),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_eval,
)


# =========================
# MODEL ARCHITECTURE
# =========================

class StochasticDepth(nn.Module):
    """
    Stochastic depth / DropPath on residual branches.
    Applies per-sample (row-wise) dropping.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # Broadcast mask over all non-batch dims
        shape = [x.size(0)] + [1] * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep_prob).float()
        return x * mask / keep_prob


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.15, drop_path_prob=0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path_prob)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, E]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        attn_out = self.drop_path(attn_out)
        x = self.norm(x + self.dropout(attn_out))
        return x


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        """
        x:        [B, T, H]
        lengths:  [B]
        """
        B, T, H = x.size()
        device = x.device
        lengths = lengths.to(device)

        mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

        scores = self.attn(x).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (weights * x).sum(dim=1)

    def get_weights(self, x, lengths):
        B, T, H = x.size()
        device = x.device
        lengths = lengths.to(device)

        mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)  # [B, T]
        return weights


class BattleLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim=256, hidden_dim=224,
                 turn_drop_prob=TURN_DROP_PROB, attn_drop_path_prob=0.10):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.turn_drop_prob = float(turn_drop_prob)

        # Projection from wide feature space -> manageable embedding
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

        # Pre-LSTM self-attention with stochastic depth
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.15,
            drop_path_prob=attn_drop_path_prob,
        )

        # BiLSTM over turns
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.30,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling
        self.pool = AttentionPool(hidden_dim=2 * hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        """
        x:       [B, T, F]
        lengths: [B]
        """
        B, T, F = x.size()
        device = x.device
        lengths = lengths.to(device)

        # Turn-level dropout: randomly drop entire turns during training
        if self.training and self.turn_drop_prob > 0.0:
            keep_prob = 1.0 - self.turn_drop_prob
            mask = (torch.rand(B, T, 1, device=device) < keep_prob).float()
            x = x * mask / keep_prob

        # Projection
        x = self.feature_proj(x)  # [B, T, embed_dim]

        # Pre-LSTM multi-head attention
        key_padding_mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]
        x = self.self_attn(x, key_padding_mask=key_padding_mask)

        # Pack for LSTM
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Clamp lengths to padded length
        T_out = lstm_out.size(1)
        lengths_clamped = lengths.clamp(max=T_out)

        # Attention pooling over turns
        context = self.pool(lstm_out, lengths_clamped)

        # Final classification
        logits = self.classifier(context).squeeze(1)
        return logits

    def get_attention_over_time(self, x, lengths):
        """
        x:       [1, T, F]
        lengths: [1]
        returns: numpy array of attention weights over time
        """
        with torch.no_grad():
            device = x.device
            B, T, F = x.size()
            lengths = lengths.to(device)

            # Turn-level dropout OFF in eval mode by default
            x_proj = self.feature_proj(x)

            key_padding_mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]
            x_attn = self.self_attn(x_proj, key_padding_mask=key_padding_mask)

            packed = pack_padded_sequence(
                x_attn,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

            T_out = lstm_out.size(1)
            lengths_clamped = lengths.clamp(max=T_out)

            weights = self.pool.get_weights(lstm_out, lengths_clamped)  # [1, T_out]
            return weights[0].cpu().numpy()


# =========================
# TRAINING SETUP
# =========================

model = BattleLSTM(len(FEATURE_COLS)).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
)

# Initialize final bias to match base positive rate (helps calibration & accuracy)
base_rate = float(y_train.mean())
base_rate = min(max(base_rate, 1e-4), 1 - 1e-4)
initial_logit = float(np.log(base_rate / (1.0 - base_rate)))
with torch.no_grad():
    model.classifier[-1].bias.fill_(initial_logit)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, lengths)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            correct += (preds == y.int()).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def evaluate_loss(loader):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, lengths)
            # Use true labels for validation loss (no smoothing)
            sample_losses = criterion(logits, y)
            total_loss += sample_losses.mean().item()
            count += 1
    return total_loss / max(1, count)


best_val_loss = float("inf")
best_val_acc_at_best_loss = 0.0
patience_counter = 0
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_loss": [],
}

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    batch_count = 0

    # update prefix curriculum (module-level variable)
    progress = (epoch - 1) / max(1, EPOCHS - 1)
    current_prefix_min_frac = (
        PREFIX_MIN_FRAC_START
        + progress * (PREFIX_MIN_FRAC_END - PREFIX_MIN_FRAC_START)
    )
    current_prefix_min_frac = float(np.clip(current_prefix_min_frac, 0.0, 1.0))

    for x, lengths, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, lengths)

        # Label smoothing for training loss
        if LABEL_SMOOTHING > 0.0:
            y_smooth = y * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        else:
            y_smooth = y

        sample_losses = criterion(logits, y_smooth)

        # Prefix-length-based weights (longer prefixes closer to full game)
        w = (lengths.float() / lengths.max().float()).to(DEVICE)
        loss = (w * sample_losses).mean()

        loss.backward()

        if GRAD_NOISE_STD > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(GRAD_NOISE_STD * torch.randn_like(p.grad))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_count += 1

    train_loss_epoch = total_loss / max(1, batch_count)
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    val_loss = evaluate_loss(val_loader)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss_epoch)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch:2d} | Loss {train_loss_epoch:7.4f} | "
        f"Train {train_acc:.4f} | Val {val_acc:.4f} | ValLoss {val_loss:.4f} "
        f"| prefix_min_frac={current_prefix_min_frac:.2f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_val_acc_at_best_loss = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"⚠ Early stop at epoch {epoch} (no val loss improvement)")
            break


state_dict = torch.load("best_model.pt", map_location=DEVICE)
model.load_state_dict(state_dict)

val_acc_final = evaluate(val_loader)
test_acc = evaluate(test_loader)
print(
    f"\n✅ Test Accuracy: {test_acc:.4f} | "
    f"Best Val Acc (at best loss): {best_val_acc_at_best_loss:.4f} | "
    f"Best Val Loss: {best_val_loss:.4f}"
)


# =========================
# TRAINING HISTORY PLOTS
# =========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["epoch"], history["train_loss"], label="Train Loss",
         linewidth=2.0, marker="o")
ax1.plot(history["epoch"], history["val_loss"], label="Val Loss",
         linewidth=2.0, marker="s")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Validation Loss")
ax1.grid(alpha=0.3)
ax1.legend()

ax2.plot(history["epoch"], history["train_acc"], label="Train Acc",
         linewidth=2.0, marker="o")
ax2.plot(history["epoch"], history["val_acc"], label="Val Acc",
         linewidth=2.0, marker="s")
ax2.axhline(best_val_acc_at_best_loss, linestyle="--", color="gray", alpha=0.6,
            label=f"Best Val Acc @ Best Loss ({best_val_acc_at_best_loss:.3f})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Train vs Val Accuracy")
ax2.set_ylim(0.5, 1.0)
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=200, bbox_inches="tight")
plt.show()
print("✅ Saved: training_history.png")


# =========================
# WIN PROBABILITY CURVES FOR SAMPLE TEST GAMES
# =========================

print("\nGenerating win probability curves for sample test games...")

model.eval()
test_game_ids_unique = game_test.unique()

n_games = min(12, len(test_seqs))
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
axes = axes.flatten()

for idx in range(n_games):
    seq = test_seqs[idx].unsqueeze(0).to(DEVICE)
    true_label = test_labels[idx].item()
    game_id = test_game_ids_unique[idx]

    T = seq.shape[1]
    probs = []
    for t in range(1, T + 1):
        with torch.no_grad():
            length_t = torch.tensor([t], dtype=torch.long).to(DEVICE)
            prob = torch.sigmoid(model(seq[:, :t, :], length_t)).item()
        probs.append(prob)

    ax = axes[idx]
    final_prob = probs[-1]
    final_pred = final_prob > 0.5
    correct = ((final_pred and true_label == 1) or ((not final_pred) and true_label == 0))
    color = "#27AE60" if correct else "#E74C3C"

    ax.plot(probs, linewidth=2.0, color=color)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(probs)), 0, probs, alpha=0.2, color=color)

    winner = "P1" if true_label == 1 else "P2"
    status = "✓" if correct else "✗"
    ax.set_title(
        f"Game {game_id} • Winner: {winner} {status} (Pred: {final_prob:.2f})",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Turn")
    ax.set_ylabel("P1 Win Prob")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle(
    "Win Probability Predictions (Green = Correct, Red = Incorrect)",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("win_curves_grid.png", dpi=200, bbox_inches="tight")
plt.show()
print("✅ Saved: win_curves_grid.png")


# =========================
# ATTENTION VISUALIZATION FOR ONE GAME
# =========================

print("\nGenerating attention visualization for a sample test game...")

if len(test_seqs) > 0:
    sample_idx = 0
    seq = test_seqs[sample_idx].unsqueeze(0).to(DEVICE)
    T = seq.shape[1]
    length = torch.tensor([T], dtype=torch.long).to(DEVICE)
    true_label = test_labels[sample_idx].item()
    game_id = test_game_ids_unique[sample_idx]

    attn_weights = model.get_attention_over_time(seq, length)

    probs = []
    for t in range(1, T + 1):
        with torch.no_grad():
            prob = torch.sigmoid(
                model(seq[:, :t, :],
                      torch.tensor([t], dtype=torch.long).to(DEVICE))
            ).item()
        probs.append(prob)

    turns = np.arange(1, len(probs) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(turns, probs, label="P1 Win Prob", linewidth=2.5)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("P1 Win Probability")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    attn_len = min(len(attn_weights), len(turns))
    attn_norm = attn_weights[:attn_len] / (attn_weights[:attn_len].max() + 1e-8)
    ax2.bar(turns[:attn_len], attn_norm, alpha=0.25, label="Attention (scaled)")
    ax2.set_ylabel("Attention (normalized)")
    ax2.set_ylim(0, 1.05)

    winner = "P1" if true_label == 1 else "P2"
    fig.suptitle(
        f"Game {game_id} • Winner: {winner} • Win Probability vs Attention",
        fontsize=14,
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("attention_win_curve_example.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("✅ Saved: attention_win_curve_example.png")


# =========================
# FEATURE IMPORTANCE (GRADIENT-BASED)
# =========================

print("\nComputing gradient-based feature importance...")

model.train()
feature_grads = torch.zeros(len(FEATURE_COLS))

num_games_for_importance = min(50, len(test_seqs))
for idx in range(num_games_for_importance):
    seq_base = test_seqs[idx].unsqueeze(0)
    seq = seq_base.clone().detach().to(DEVICE)
    seq.requires_grad_(True)

    T = seq.shape[1]
    midpoint = max(1, T // 2)
    length_mid = torch.tensor([midpoint], dtype=torch.long).to(DEVICE)

    output = model(seq[:, :midpoint, :], length_mid)
    output.backward()

    if seq.grad is not None:
        feature_grads += seq.grad[0, :midpoint, :].abs().mean(dim=0).cpu()

    model.zero_grad()

feature_grads /= max(1, num_games_for_importance)
model.eval()

top_k_grad = min(15, len(FEATURE_COLS))
top_indices_grad = torch.topk(feature_grads, k=top_k_grad).indices
top_features_grad = [(FEATURE_COLS[i], feature_grads[i].item()) for i in top_indices_grad]

print("\n🔥 Top 15 Most Influential Features (by gradient magnitude):")
for rank, (feat, importance) in enumerate(top_features_grad, 1):
    print(f"{rank:2d}. {feat:40s} → {importance:.4f}")

feat_names_grad = [f for f, _ in top_features_grad][::-1]
feat_vals_grad = [v for _, v in top_features_grad][::-1]

plt.figure(figsize=(10, 6))
plt.barh(feat_names_grad, feat_vals_grad)
plt.xlabel("Gradient-based Importance")
plt.title("Top Feature Importances (Mid-game Gradients)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()
print("✅ Saved: feature_importance.png")


# =========================
# CALIBRATION PLOT
# =========================

print("\nGenerating calibration and prediction distribution plots...")

all_probs = []
all_labels = []
with torch.no_grad():
    for x, lengths, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_probs = []
bin_accs = []

for i in range(n_bins):
    m = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
    if m.sum() > 0:
        bin_probs.append(bin_centers[i])
        bin_accs.append(all_labels[m].mean())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
ax.plot(bin_probs, bin_accs, "o-", label="Model")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Actual Win Rate")
ax.set_title("Calibration Curve")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

ax = axes[1]
ax.hist(all_probs[all_labels == 1], bins=20, alpha=0.6, label="P1 Wins")
ax.hist(all_probs[all_labels == 0], bins=20, alpha=0.6, label="P2 Wins")
ax.axvline(0.5, color="black", linestyle="--", alpha=0.7)
ax.set_xlabel("Predicted P1 Win Probability")
ax.set_ylabel("Count")
ax.set_title("Prediction Distribution")
ax.legend()
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("calibration_analysis.png", dpi=200, bbox_inches="tight")
plt.show()
print("✅ Saved: calibration_analysis.png")


# =========================
# SUMMARY STATS
# =========================

print("\n" + "=" * 60)
print("📊 MODEL PERFORMANCE SUMMARY")
print("=" * 60)

high_conf_mask = (all_probs > 0.7) | (all_probs < 0.3)
if high_conf_mask.sum() > 0:
    high_conf_acc = (
        (all_probs[high_conf_mask] > 0.5) ==
        all_labels[high_conf_mask]
    ).mean()
else:
    high_conf_acc = float("nan")

medium_conf_mask = (all_probs >= 0.4) & (all_probs <= 0.6)
if medium_conf_mask.sum() > 0:
    medium_conf_acc = (
        (all_probs[medium_conf_mask] > 0.5) ==
        all_labels[medium_conf_mask]
    ).mean()
else:
    medium_conf_acc = float("nan")

print(f"Overall Test Accuracy:          {test_acc:.2%}")
print(f"High Confidence (>70% or <30%): {high_conf_acc:.2%} "
      f"({high_conf_mask.sum()}/{len(all_probs)} games)")
print(f"Medium Confidence (40–60%):     {medium_conf_acc:.2%} "
      f"({medium_conf_mask.sum()}/{len(all_probs)} games)")
print(f"\nMean Predicted Probability:     {all_probs.mean():.3f}")
print(f"Actual P1 Win Rate:             {all_labels.mean():.3f}")
print("=" * 60)
