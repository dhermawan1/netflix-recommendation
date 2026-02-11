import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 2. Generating a Dummy Dataset ---
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
    (9, "Avengers: Endgame", "Action")
]
movies_df = pd.DataFrame(movies_data, columns=['movie_id', 'title', 'genre'])

# 2. Generate random ratings
num_users = 20
num_ratings = 100

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

print("Movies DataFrame:")
print(movies_df.head())
print("\nRatings DataFrame:")
print(ratings_df.head())

# --- 3. Data Preparation (Added Genre Tensor) ---
# Prepare Genre Mapping
unique_genres = movies_df['genre'].unique()
genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
movies_df['genre_id'] = movies_df['genre'].map(genre_to_idx)

# Map genre IDs to ratings dataframe
# We map from movie_id to ensure every rating row gets the correct genre
ratings_df['genre_id'] = ratings_df['movie_id'].map(movies_df.set_index('movie_id')['genre_id'])

# Convert to tensors
# LongTensor is used for indices (discrete values)
# FloatTensor is used for ratings (continuous values)
user_tensor = torch.LongTensor(ratings_df['user_id'].values)
movie_tensor = torch.LongTensor(ratings_df['movie_id'].values)
genre_tensor = torch.LongTensor(ratings_df['genre_id'].values)
rating_tensor = torch.FloatTensor(ratings_df['rating'].values)

print(f"Total unique ratings available for training: {len(rating_tensor)}")
print(f"Genre Tensor shape: {genre_tensor.shape}")
print(f"Genre Map: {genre_to_idx}")


# --- 4. Building the Matrix Factorization Model ---
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
        return predicted_rating

# Instantiate the model
# We strictly define num_users and num_movies based on our dataset limits
model = RecommenderNet(num_users=num_users, num_movies=len(movies_df), num_genres=len(unique_genres))
print(model)

# --- 5. Training the Model ---
# 1. Define Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 2. Training Loop
epochs = 200
print("Starting training...")

for epoch in range(epochs):
    optimizer.zero_grad() # Clear previous gradients
    
    # Forward pass: Compute predicted ratings
    predictions = model(user_tensor, movie_tensor, genre_tensor)
    
    # Compute loss: Compare predictions with actual ratings
    loss = loss_fn(predictions, rating_tensor)
    
    # Backward pass: Compute gradients and update weights
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

print("Training complete.")

# --- 6. Making Recommendations ---
def recommend_movies(user_id, model, movies_df, ratings_df, num_recommendations=3):
    # Get all movie IDs available
    all_movie_ids = torch.arange(len(movies_df))
    
    # Get movies already rated by the user so we don't recommend them again
    rated_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
    
    # Filter out watched movies
    movies_to_predict = [m_id for m_id in all_movie_ids.tolist() if m_id not in rated_movie_ids]
    
    if not movies_to_predict:
        return []
    
    # Prepare input tensors for the model
    movies_tensor_pred = torch.LongTensor(movies_to_predict)
    user_tensor_pred = torch.LongTensor([user_id] * len(movies_to_predict))
    
    # Get genre indices for the movies
    # Map movie_id to genre_id using the movies_df
    movie_genre_map = movies_df.set_index('movie_id')['genre_id']
    genres_list = [movie_genre_map[m_id] for m_id in movies_to_predict]
    genre_tensor_pred = torch.LongTensor(genres_list)
    
    # Predict ratings for unwatched movies
    with torch.no_grad():
        predictions = model(user_tensor_pred, movies_tensor_pred, genre_tensor_pred)
    
    # Sort predictions (descending order) to get top rated movies
    # torch.argsort returns the INDICES of the sorted values
    sorted_indices = torch.argsort(predictions, descending=True)
    
    top_indices = sorted_indices[:num_recommendations]
    
    recommended_movie_ids = movies_tensor_pred[top_indices].numpy()
    predicted_ratings = predictions[top_indices].numpy()
    
    # Fetch movie details for the results
    results = []
    for m_id, pred_rating in zip(recommended_movie_ids, predicted_ratings):
        movie_info = movies_df[movies_df['movie_id'] == m_id].iloc[0]
        results.append((movie_info['title'], movie_info['genre'], pred_rating))
        
    return results

# --- TEST THE SYSTEM ---
user_id_to_test = 0
print(f"generating recommendations for User ID {user_id_to_test}...")

recommendations = recommend_movies(user_id_to_test, model, movies_df, ratings_df)

if recommendations:
    for title, genre, rating in recommendations:
        print(f"Recommend: '{title}' ({genre}) - Predicted Rating: {rating:.2f}")
else:
    print("No movies left to recommend!")
