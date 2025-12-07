import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)

status_suffixes = [
    "status_tox", "status_brn", "status_par",
    "status_psn", "status_frz", "status_slp",
]

move_types = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic",
    "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel",
]

move_categories = ["Physical", "Special", "Status"]

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    

    def _active_stat(self, df, side, stat):
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
    

    def _team_hp_total(self, df, side):
        cols = []
        for i in range(1, 7):
            c = f"{side}_slot{i}_current_health"
            if c in df:
                cols.append(df[c].fillna(0))
        if not cols:
            return pd.Series(0.0, index=df.index)
        return pd.concat(cols, axis=1).sum(axis=1)


    def _team_strength(self, df, side):
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
    
    def _status_count(self, df, side):
        s = 0
        for i in range(1, 7):
            for suf in status_suffixes:
                col = f"{side}_slot{i}_{suf}"
                if col in df:
                    s += df[col].fillna(0)
        return s


    def run(self):
        df = self.df.sort_values(["game_id", "turn"]).reset_index(drop=True)
        
        print(f"Loaded {len(df)} turns across {df['game_id'].nunique()} games.")

        print("Adding high-level momentum / advantage features (causal only)...")

        # --- Team HP totals and damage (instantaneous + diff to previous turn) ---
        df["p1_team_hp_total"] = self._team_hp_total(df, "p1")
        df["p2_team_hp_total"] = self._team_hp_total(df, "p2")

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
        #df["remaining_diff"] = df["p1_remaining"] - df["p2_remaining"]

        # --- Active HP and speed (instantaneous) ---
        df["p1_active_hp"] = self._active_stat(df, "p1", "current_health")
        df["p2_active_hp"] = self._active_stat(df, "p2", "current_health")
        df["active_hp_diff"] = df["p1_active_hp"] - df["p2_active_hp"]

        df["p1_active_speed"] = self._active_stat(df, "p1", "spe")
        df["p2_active_speed"] = self._active_stat(df, "p2", "spe")
        df["active_speed_diff"] = df["p1_active_speed"] - df["p2_active_speed"]

        # --- Boosts (instantaneous) ---
        p1_boost_cols = [c for c in df.columns if c.startswith("p1_") and c.endswith("_boost")]
        p2_boost_cols = [c for c in df.columns if c.startswith("p2_") and c.endswith("_boost")]
        df["p1_boost_sum"] = df[p1_boost_cols].fillna(0).sum(axis=1) if p1_boost_cols else 0
        df["p2_boost_sum"] = df[p2_boost_cols].fillna(0).sum(axis=1) if p2_boost_cols else 0
        df["boost_diff"] = df["p1_boost_sum"] - df["p2_boost_sum"]


        df["p1_status_count"] = self._status_count(df, "p1")
        df["p2_status_count"] = self._status_count(df, "p2")
        df["status_count_diff"] = df["p1_status_count"] - df["p2_status_count"]

        df["pokemon_lost_p1"] = (
            df.groupby("game_id")["p1_remaining"]
            .diff().fillna(0).clip(upper=0).abs()
        )
        df["pokemon_lost_p2"] = (
            df.groupby("game_id")["p2_remaining"]
            .diff().fillna(0).clip(upper=0).abs()
        )
        df["recent_ko_advantage"] = df["pokemon_lost_p2"] - df["pokemon_lost_p1"]

        #df["cumulative_remaining_advantage"] = df.groupby("game_id")["remaining_diff"].cumsum()

        # --- Turn number (safe), but NO game_progress ---
        df["turn_number"] = df.groupby("game_id").cumcount() + 1

        # --- Active HP percentage and fainted counts (instantaneous) ---
        den_p1_hp = self._active_stat(df, "p1", "hp") + 1e-6
        den_p2_hp = self._active_stat(df, "p2", "hp") + 1e-6
        df["p1_active_hp_pct"] = df["p1_active_hp"] / den_p1_hp
        df["p2_active_hp_pct"] = df["p2_active_hp"] / den_p2_hp
        df["active_hp_pct_diff"] = df["p1_active_hp_pct"] - df["p2_active_hp_pct"]

        df["p1_fainted"] = 6 - df["p1_remaining"]
        df["p2_fainted"] = 6 - df["p2_remaining"]

        # --- Team strength (static per game but repeated per turn, safe) ---
        df["p1_team_strength"] = self._team_strength(df, "p1")
        df["p2_team_strength"] = self._team_strength(df, "p2")
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


        # --- Last move features (power, type, category) ---
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

        print("Features Added")
        

        EXCLUDE_COLS = [
            "game_id",
            "p1_win",
            "turn",
        ]

        feature_cols= [
            c for c in df.columns
            if c not in EXCLUDE_COLS and c not in ["Unnamed: 0"]
        ]

        print(f"Using {len(feature_cols)} features")

        X_all = df[feature_cols]
        y_all = df["p1_win"].astype(int)
        game_ids = df["game_id"]

        return X_all, y_all, game_ids, feature_cols
    

class BattleDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SequenceBuilder:
    def __init__(self, X, y, game_ids, feature_cols, current_prefix_min_frac):
        self.X = X
        self.y = y
        self.game_ids = game_ids
        self.feature_cols = feature_cols
        self.current_prefix_min_frac = current_prefix_min_frac

    
    def _build_sequences(self, X_df, y_series, game_series):
        sequences, labels, lengths = [], [], []
        for gid in game_series.unique():
            mask = (game_series == gid)
            seq_arr = X_df[mask].values
            lbl = float(y_series[mask].iloc[-1])

            seq = torch.tensor(seq_arr, dtype=torch.float32)
            sequences.append(seq)
            labels.append(torch.tensor(lbl, dtype=torch.float32))
            lengths.append(len(seq))

        return sequences, labels, lengths
    
    def _collate_fn_train(self, batch):
        seqs, labels = zip(*batch)
        new_seqs, new_lengths = [], []

        for seq in seqs:
            full_len = len(seq)
            min_len = max(self.min_prefix_length,
                        int(self.current_prefix_min_frac * full_len))

            if min_len >= full_len:
                prefix_len = full_len
            else:
                prefix_len = np.random.randint(min_len, full_len + 1)

            new_seqs.append(seq[:prefix_len])
            new_lengths.append(prefix_len)

        lengths_tensor = torch.tensor(new_lengths, dtype=torch.long)
        padded = pad_sequence(new_seqs, batch_first=True, padding_value=self.pad_value)
        labels_tensor = torch.stack(labels)

        return padded, lengths_tensor, labels_tensor



    def _collate_fn_eval(self, batch):
        """
        Collate without prefix sampling: full games.
        """
        seqs, labels = zip(*batch)

        seqs = list(seqs)
        labels = list(labels)

        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=self.pad_value)
        labels_tensor = torch.stack(labels)
        return padded, lengths, labels_tensor
    

    def train_test_val_split(self, batch_size, test_split, val_split, min_prefix_length, pad_value):
        self.min_prefix_length = min_prefix_length
        self.pad_value = pad_value

        unique_games = self.game_ids.unique()
        np.random.shuffle(unique_games)

        n_test = int(len(unique_games) * test_split)
        n_val = int(len(unique_games) * val_split)

        test_games = set(unique_games[:n_test])
        val_games = set(unique_games[n_test:n_test + n_val])
        train_games = set(unique_games[n_test + n_val:])

        train_mask = self.game_ids.isin(train_games)
        val_mask = self.game_ids.isin(val_games)
        test_mask = self.game_ids.isin(test_games)

        X_train, X_val, X_test = self.X[train_mask], self.X[val_mask], self.X[test_mask]
        y_train, y_val, y_test = self.y[train_mask], self.y[val_mask], self.y[test_mask]
        game_train, game_val, game_test = self.game_ids[train_mask], self.game_ids[val_mask], self.game_ids[test_mask]

        print(f"Train games: {len(train_games)}, Val games: {len(val_games)}, Test games: {len(test_games)}")


        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=self.feature_cols,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=self.feature_cols,
            index=X_val.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=self.feature_cols,
            index=X_test.index,
        )

        train_seqs, train_labels, train_lengths = self._build_sequences(X_train_scaled, y_train, game_train)
        val_seqs, val_labels, val_lengths = self._build_sequences(X_val_scaled, y_val, game_val)
        test_seqs, test_labels, test_lengths = self._build_sequences(X_test_scaled, y_test, game_test)

        train_loader = DataLoader(
            BattleDataset(train_seqs, train_labels),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn_train,
        )
        val_loader = DataLoader(
            BattleDataset(val_seqs, val_labels),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn_eval,
        )
        test_loader = DataLoader(
            BattleDataset(test_seqs, test_labels),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn_eval,
        )

        return (train_loader, val_loader, test_loader), \
            (game_test, test_seqs, test_labels), \
            y_train, \
            self

        

    def set_prefix_frac(self, frac):
        self.current_prefix_min_frac = frac
