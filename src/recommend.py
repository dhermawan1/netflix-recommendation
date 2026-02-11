import torch
from model import RecommenderModel
import pandas as pd

movies = {
    1:"Avengers",
    2:"Batman",
    3:"Spiderman",
    4:"Ironman",
    5:"Harry Potter",
    6:"LOTR"
}

def load_model():
    model = RecommenderModel(50,50)
    model.load_state_dict(torch.load("models/recommender.pt"))
    model.eval()
    return model

def recommend(user_id):
    model = load_model()

    movie_ids = list(movies.keys())
    user_tensor = torch.tensor([user_id]*len(movie_ids))
    movie_tensor = torch.tensor(movie_ids)

    preds = model(user_tensor, movie_tensor).detach().numpy()

    result = list(zip(movie_ids, preds))
    result.sort(key=lambda x: x[1], reverse=True)

    return [movies[m] for m,_ in result[:3]]
