# Create Dataset and Dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PokemonDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx], dtype=torch.float), torch.from_numpy(
            self.y[idx], dtype=torch.float
        )


