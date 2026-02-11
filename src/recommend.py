import torch
import pandas as pd
import json
import os
from src.model import RecommenderNet
from src.config import EMBEDDING_SIZE, NUM_USERS, MODEL_PATH, MOVIES_CSV, RATINGS_CSV, GENRE_MAP

def load_artifacts():
    # Load Model
    if not os.path.exists(MOVIES_CSV):
        raise FileNotFoundError("Data not generated yet. Please run src/train.py first.")
        
    movies_df = pd.read_csv(MOVIES_CSV)
    ratings_df = pd.read_csv(RATINGS_CSV)
    with open(GENRE_MAP, "r") as f:
        genre_to_idx = json.load(f)

    num_movies = len(movies_df)
    num_genres = len(genre_to_idx)

    model = RecommenderNet(num_users=NUM_USERS, num_movies=num_movies, num_genres=num_genres, embedding_size=EMBEDDING_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"Model not found at {MODEL_PATH}, using untrained model.")
    
    model.eval()
    
    return model, movies_df, ratings_df, genre_to_idx

def recommend_movies(user_id, num_recommendations=3):
    model, movies_df, ratings_df, genre_to_idx = load_artifacts()

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
    # In dataset.py, genre_id is added to movies_df upon generation. 
    # Since we loaded movies_df from CSV, check if genre_id exists.
    if 'genre_id' not in movies_df.columns:
        # Re-create mapping if needed (but train.py saves movies_df generally *after* adding genre_id?)
        # Let's check train.py. Yes, train.py saves movies_df *after* generate_dummy_data which adds 'genre_id'.
        # However, generate_dummy_data adds 'genre_id' to `movies_df`?
        # Let's look at `generate_dummy_data` in `src/dataset.py`.
        # Yes: movies_df['genre_id'] = movies_df['genre'].map(genre_to_idx)
        pass

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
        results.append({
            "title": str(movie_info['title']),
            "genre": str(movie_info['genre']),
            "rating": float(pred_rating)
        })
        
    return results

if __name__ == "__main__":
    user_id_to_test = 0
    print(f"Generating recommendations for User ID {user_id_to_test}...")
    
    try:
        recommendations = recommend_movies(user_id_to_test)
        
        if recommendations:
            for rec in recommendations:
                print(f"Recommend: '{rec['title']}' ({rec['genre']}) - Predicted Rating: {rec['rating']:.2f}")
        else:
            print("No movies left to recommend!")
    except Exception as e:
        print(f"Error: {e}")
