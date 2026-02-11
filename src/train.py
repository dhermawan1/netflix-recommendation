import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import RecommenderModel
from dataset import MovieDataset

df = pd.read_csv("data/dummy_ratings.csv")

dataset = MovieDataset(df)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_users = df.user_id.nunique()
num_movies = df.movie_id.nunique()

model = RecommenderModel(num_users, num_movies)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 30

for epoch in range(epochs):
    total_loss = 0

    for user, movie, rating in loader:
        pred = model(user, movie)
        loss = loss_fn(pred, rating)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss {total_loss:.4f}")

torch.save(model.state_dict(), "models/recommender.pt")
print("Model saved")
