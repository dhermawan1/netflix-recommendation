import torch
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df.user_id.values, dtype=torch.long)
        self.movies = torch.tensor(df.movie_id.values, dtype=torch.long)
        self.ratings = torch.tensor(df.rating.values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
