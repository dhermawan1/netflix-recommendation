import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_size=16):
        super(RecommenderNet, self).__init__()
        # Embeddings: A lookup table that stores embeddings of a fixed dictionary and size.
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.genre_embedding = nn.Embedding(num_genres, embedding_size)
        
    def forward(self, user_indices, movie_indices, genre_indices):
        # Retrieve embeddings for the specific users and movies in the batch
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        genre_embed = self.genre_embedding(genre_indices)
        
        # Dot product: Multiply vectors and sum across the embedding dimension
        # We add the genre embedding to the movie embedding to enrich the item representation
        # Shape: (Batch_Size, Embedding_Size) -> (Batch_Size)

        predicted_rating = (user_embed * (movie_embed + genre_embed)).sum(dim=1)

        # score = torch.sum(user_embed * (movie_embed + genre_embed), dim=1)
        # predicted_rating = torch.sigmoid(score) * 4 + 1

        return predicted_rating
