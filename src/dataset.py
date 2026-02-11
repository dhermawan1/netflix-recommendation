import torch
import pandas as pd
import numpy as np
from src.config import NUM_USERS, NUM_RATINGS

def generate_dummy_data():
    """
    Simulates a real-world scenario by creating a small dataset with users, movies, and ratings.
    """
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define dummy movies with IDs
    movies_data = [
        (0, "The Matrix", "Action"),
        (1, "Inception", "Sci-Fi"),
        (2, "The Godfather", "Crime"),
        (3, "Toy Story", "Animation"),
        (4, "Pulp Fiction", "Crime"),
        (5, "The Dark Knight", "Action"),
        (6, "Interstellar", "Sci-Fi"),
        (7, "Forrest Gump", "Drama"),
        (8, "Frozen", "Animation"),
        (9, "Avengers: Endgame", "Action"),
        (10, "The Shawshank Redemption", "Drama"),
        (11, "Gladiator", "Action"),
        (12, "Spirited Away", "Animation"),
        (13, "Parasite", "Thriller"),
        (14, "The Lion King", "Animation"),
        (15, "Fight Club", "Drama"),
        (16, "Blade Runner 2049", "Sci-Fi"),
        (17, "Joker", "Crime"),
        (18, "Spider-Man: Into the Spider-Verse", "Animation"),
        (19, "The Silence of the Lambs", "Thriller"),
        (20, "Star Wars: A New Hope", "Sci-Fi"),
        (21, "Saving Private Ryan", "War"),
        (22, "Jurassic Park", "Adventure"),
        (23, "The Lord of the Rings: The Fellowship of the Ring", "Adventure"),
        (24, "Coco", "Animation"),
        (25, "The Departed", "Crime"),
        (26, "The Shining", "Horror"),
        (27, "Goodfellas", "Crime"),
        (28, "Alien", "Sci-Fi"),
        (29, "The Prestige", "Drama"),
        (30, "Memento", "Mystery"),
        (31, "Seven", "Thriller"),
        (32, "Mad Max: Fury Road", "Action"),
        (33, "Logan", "Action"),
        (34, "Your Name", "Animation"),
        (35, "Arrival", "Sci-Fi"),
        (36, "The Wolf of Wall Street", "Biography"),
        (37, "Whiplash", "Drama"),
        (38, "Django Unchained", "Western"),
        (39, "Inglourious Basterds", "War"),
        (40, "La La Land", "Romance"),
        (41, "The Grand Budapest Hotel", "Comedy"),
        (42, "The Truman Show", "Comedy"),
        (43, "A Clockwork Orange", "Sci-Fi"),
        (44, "Psycho", "Horror"),
        (45, "No Country for Old Men", "Crime"),
        (46, "Up", "Animation"),
        (47, "Terminator 2: Judgment Day", "Action"),
        (48, "Back to the Future", "Sci-Fi"),
        (49, "Heat", "Crime")
    ]
    movies_df = pd.DataFrame(movies_data, columns=['movie_id', 'title', 'genre'])

    # 2. Generate random ratings
    num_users = NUM_USERS
    num_ratings = NUM_RATINGS

    # Randomly assign users to movies with random ratings
    user_ids = np.random.randint(0, num_users, num_ratings)
    movie_ids = np.random.randint(0, len(movies_df), num_ratings)
    ratings = np.random.randint(1, 6, num_ratings)

    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings
    })


    # Remove duplicates (simulate keeping the latest rating if a user rated twice)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

    # Prepare Genre Mapping
    unique_genres = movies_df['genre'].unique()
    genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
    movies_df['genre_id'] = movies_df['genre'].map(genre_to_idx)

    # Map genre IDs to ratings dataframe
    # We map from movie_id to ensure every rating row gets the correct genre
    ratings_df['genre_id'] = ratings_df['movie_id'].map(movies_df.set_index('movie_id')['genre_id'])
    
    return movies_df, ratings_df, genre_to_idx

def get_tensors(ratings_df):
    """
    Converts ratings dataframe to PyTorch tensors.
    """
    # Convert to tensors
    # LongTensor is used for indices (discrete values)
    # FloatTensor is used for ratings (continuous values)
    user_tensor = torch.LongTensor(ratings_df['user_id'].values)
    movie_tensor = torch.LongTensor(ratings_df['movie_id'].values)
    genre_tensor = torch.LongTensor(ratings_df['genre_id'].values)
    rating_tensor = torch.FloatTensor(ratings_df['rating'].values)
    
    return user_tensor, movie_tensor, genre_tensor, rating_tensor
