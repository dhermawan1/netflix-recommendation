import torch
import torch.nn as nn

class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users+1, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies+1, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim*2, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, user, movie):
        user_vec = self.user_embedding(user)
        movie_vec = self.movie_embedding(movie)

        x = torch.cat([user_vec, movie_vec], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()
