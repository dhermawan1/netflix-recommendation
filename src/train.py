import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import json
from src.dataset import generate_dummy_data, get_tensors
from src.model import RecommenderNet
from src.config import EMBEDDING_SIZE, NUM_USERS, EPOCHS, LEARNING_RATE, MODEL_PATH, MOVIES_CSV, RATINGS_CSV, GENRE_MAP

def train():
    # 1. Generate Data
    print("Generating dummy data...")
    movies_df, ratings_df, genre_to_idx = generate_dummy_data()
    
    # Save data for consistency
    os.makedirs("data", exist_ok=True)
    movies_df.to_csv(MOVIES_CSV, index=False)
    ratings_df.to_csv(RATINGS_CSV, index=False)
    with open(GENRE_MAP, "w") as f:
        json.dump(genre_to_idx, f)
        
    print(f"Data saved to data/ directory. Genre Map: {genre_to_idx}")

    # 2. Prepare Tensors
    user_tensor, movie_tensor, genre_tensor, rating_tensor = get_tensors(ratings_df)
    
    print(f"User Tensor Shape: {user_tensor.shape}")
    print(f"Movie Tensor Shape: {movie_tensor.shape}")
    print(f"Genre Tensor Shape: {genre_tensor.shape}")
    print(f"Rating Tensor Shape: {rating_tensor.shape}")

    # 3. Model Initialization
    num_movies = len(movies_df)
    num_genres = len(genre_to_idx)
    
    # Instantiate the model
    # We strictly define num_users and num_movies based on our dataset limits
    model = RecommenderNet(num_users=NUM_USERS, num_movies=num_movies, num_genres=num_genres, embedding_size=EMBEDDING_SIZE)
    print(model)

    # 4. Define Loss and Optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad() # Clear previous gradients
        
        # Forward pass: Compute predicted ratings
        predictions = model(user_tensor, movie_tensor, genre_tensor)
        
        # Compute loss: Compare predictions with actual ratings
        loss = loss_fn(predictions, rating_tensor)
        
        # Backward pass: Compute gradients and update weights
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')
            
    print("Training complete.")

    # 6. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
