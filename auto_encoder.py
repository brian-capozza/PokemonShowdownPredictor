import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import plotly.express as px


def embed_column(df, column, emb_dict, prefix=None):
    """Replace string column with embedding vectors."""
    emb_dim = len(next(iter(emb_dict.values())))
    prefix = prefix or column

    emb_cols = [f"{prefix}_{i}" for i in range(emb_dim)]
    df[emb_cols] = df[column].apply(lambda x: pd.Series(emb_dict.get(x, [0]*emb_dim)))
    df.drop(columns=[column], inplace=True)
    
    return df


class PokemonAE(nn.Module):
    def __init__(self, num_types, embed_dim):
        super().__init__()

        self.type_emb = nn.Embedding(num_types, embed_dim, padding_idx=0)

        self.encoder = nn.Sequential(
            nn.Linear(6 + embed_dim*2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 6 + embed_dim*2)
        )

    def forward(self, stats, t1, t2):
        t1_emb = self.type_emb(t1)
        t2_emb = self.type_emb(t2)
        x = torch.cat([stats, t1_emb, t2_emb], dim=1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    
    
class MoveAE(nn.Module):
    def __init__(self, num_types, num_cats, embed_dim):
        super().__init__()

        self.type_emb = nn.Embedding(num_types, embed_dim, padding_idx=0)
        self.cat_emb  = nn.Embedding(num_cats, embed_dim, padding_idx=0)

        # input = accuracy + base_power + pp + type_emb + cat_emb
        input_dim = 3 + embed_dim*2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, stats, type_id, cat_id):
        t_emb = self.type_emb(type_id)
        c_emb = self.cat_emb(cat_id)
        x = torch.cat([stats, t_emb, c_emb], dim=1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class PokemonAutoEncoder():
    def __init__(self, pokemon_dict):
        self.EMBED_DIM = 3
        self.df = pd.DataFrame.from_dict(pokemon_dict, orient="index")
        self.df["pokemon_name"] = self.df.index
        self._preprocessing()
        self._train()

    def _preprocessing(self):
        all_types = sorted({t for row in self.df[["type1", "type2"]].values for t in row if t})
        type_to_id = {t: i+1 for i, t in enumerate(all_types)}  # shift by 1 → 0 = NONE
        self.num_types = len(all_types) + 1  # padding index included

        self.df["type1_id"] = self.df["type1"].map(lambda t: type_to_id.get(t, 0))
        self.df["type2_id"] = self.df["type2"].map(lambda t: type_to_id.get(t, 0))

        scaler = StandardScaler()
        stats_scaled = scaler.fit_transform(self.df[["hp","atk","def","spa","spd","spe"]])
        self.stats_tensor = torch.tensor(stats_scaled, dtype=torch.float32)
        self.t1_tensor = torch.tensor(self.df["type1_id"].values, dtype=torch.long)
        self.t2_tensor = torch.tensor(self.df["type2_id"].values, dtype=torch.long)

    def _train(self):
        ae = PokemonAE(self.num_types, self.EMBED_DIM)
        optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(1200):
            optimizer.zero_grad()
            out, _ = ae(self.stats_tensor, self.t1_tensor, self.t2_tensor)
            loss = loss_fn(out[:, :6], self.stats_tensor)  # reconstruct ONLY stats
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, Z = ae(self.stats_tensor, self.t1_tensor, self.t2_tensor)

        self.emb = Z.numpy()

    def plot_embeddings(self):
        df_plot = pd.DataFrame({
            "x": self.emb[:,0],
            "y": self.emb[:,1],
            "pokemon": self.df["pokemon_name"]
        })

        fig = px.scatter(df_plot, x="x", y="y", text="pokemon")
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Autoencoder Embedding With Pokémon Type Embeddings")

        fig.show()


    def get_embeddings(self):
        return {
            name: self.emb[i].tolist()
            for i, name in enumerate(self.df["pokemon_name"])
        }



class MoveAutoEncoder:
    def __init__(self, moves_dict):
        self.EMBED_DIM = 3
        self.df = pd.DataFrame.from_dict(moves_dict, orient="index")
        self.df["move_name"] = self.df.index
        self._preprocess()
        self._train()

    def _preprocess(self):
        self.df["accuracy"] = self.df["accuracy"].map(lambda x: 100 if x is True else x)
        self.df["base_power"] = self.df["base_power"].fillna(0)

        all_types = sorted({t for t in self.df["type"].unique() if t})
        self.type_to_id = {t: i+1 for i, t in enumerate(all_types)}
        self.num_types = len(all_types) + 1
        self.df["type_id"] = self.df["type"].map(lambda x: self.type_to_id.get(x, 0))

        all_cats = sorted(self.df["category"].unique())
        self.cat_to_id = {c: i+1 for i, c in enumerate(all_cats)}
        self.num_cats = len(all_cats) + 1
        self.df["cat_id"] = self.df["category"].map(lambda x: self.cat_to_id.get(x, 0))

        scaler = StandardScaler()
        numeric = self.df[["accuracy", "base_power", "pp"]]
        scaled = scaler.fit_transform(numeric)

        self.stats_tensor = torch.tensor(scaled, dtype=torch.float32)
        self.type_tensor  = torch.tensor(self.df["type_id"].values, dtype=torch.long)
        self.cat_tensor   = torch.tensor(self.df["cat_id"].values, dtype=torch.long)

    def _train(self):
        ae = MoveAE(self.num_types, self.num_cats, self.EMBED_DIM)

        optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(1200):
            optimizer.zero_grad()
            out, _ = ae(self.stats_tensor, self.type_tensor, self.cat_tensor)
            loss = loss_fn(out[:, :3], self.stats_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, Z = ae(self.stats_tensor, self.type_tensor, self.cat_tensor)

        self.emb = Z.numpy()

    def get_embeddings(self):
        return {
            name: self.emb[i].tolist()
            for i, name in enumerate(self.df["move_name"])
        }

    def plot_embeddings(self):
        dim = self.emb.shape[1]   # number of latent dimensions

        df_plot = pd.DataFrame(self.emb, columns=[f"z{i+1}" for i in range(dim)])
        df_plot["move"] = self.df["move_name"]
        df_plot["type"] = self.df["type"]
        df_plot["category"] = self.df["category"]
        df_plot["power"] = self.df["base_power"]
        df_plot["accuracy"] = self.df["accuracy"]

        # ---- 2D ----
        if dim == 2:
            fig = px.scatter(
                df_plot,
                x="z1", y="z2",
                color="type",
                hover_name="move",
                hover_data={"power": True, "accuracy": True, "category": True},
                title="Move AutoEncoder — 2D Embeddings",
            )
            fig.update_traces(marker=dict(size=6))
            fig.show()
            return

        # ---- 3D ----
        if dim == 3:
            fig = px.scatter_3d(
                df_plot,
                x="z1", y="z2", z="z3",
                color="type",
                hover_name="move",
                hover_data={"power": True, "accuracy": True, "category": True},
                title="Move AutoEncoder — 3D Embeddings",
            )
            fig.update_traces(marker=dict(size=4))
            fig.show()
            return

        # ---- Higher Dimensions ----
        raise ValueError(f"Cannot plot embeddings with dimension {dim}. Expected 2 or 3.")

