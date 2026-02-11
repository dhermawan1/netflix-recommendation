# Netflix-Style Movie Recommendation System

This is a movie recommendation system built using **PyTorch**, **FastAPI**, and **Embedding-based Collaborative Filtering**. This project demonstrates an end-to-end Machine Learning Engineering workflow, from data generation and model training to serving recommendations via a REST API.

## Features

- **Matrix Factorization Model**: Utilizes User, Movie, and Genre embeddings to predict user ratings.
- **PyTorch Implementation**: Efficient and scalable model training.
- **Data Generation**: Custom script to generate dummy user-movie interaction data simulation.
- **Training Pipeline**: Automated script for data generation, model training, and artifact saving.
- **REST API**: FastAPI service to provide real-time recommendations.
- **Production-Ready Structure**: organized modular code structure (`src/`, `api/`, `data/`, `models/`).

## Project Structure

```
netflix-recommendation/
├── api/
│   └── main.py             # FastAPI application entry point
├── data/                   # Generated data (movies, ratings, genre mappings)
├── models/                 # Saved PyTorch models
├── notebooks/
│   └── MovieRecommender.ipynb # Jupyter Notebook for exploration and theory
├── src/
│   ├── config.py           # Configuration parameters
│   ├── dataset.py          # Data generation and PyTorch Dataset class
│   ├── model.py            # Neural Network Architecture (RecommenderNet)
│   ├── recommend.py        # Inference logic
│   └── train.py            # Training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the Model and Generate Data

Before running the recommendation service, you need to generate the dummy data and train the model. This script saves the data to `data/` and the trained model to `models/recommender.pt`.

```bash
python -m src.train
```

### 2. Get Recommendations via CLI

You can test the recommendation logic directly from the command line for a specific user.

```bash
python -m src.recommend
```

### 3. Run the API Service

Start the FastAPI server to serve recommendations over HTTP.

```bash
# Using Python module execution
python -m api.main

# OR using Uvicorn directly
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 4. Test the API

**Health Check:**
```bash
curl http://127.0.0.1:8000/
```

**Get Recommendations for User 0:**
```bash
curl http://127.0.0.1:8000/recommend/0
```

**Interactive Documentation:**
Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the endpoints using the Swagger UI.

## Model Architecture

The core model `RecommenderNet` learns low-dimensional representations (embeddings) for Users, Movies, and Genres.

- **Input**: User ID, Movie ID, Genre ID
- **Layers**:
    - `user_embedding`: Maps user IDs to embedding vectors.
    - `movie_embedding`: Maps movie IDs to embedding vectors.
    - `genre_embedding`: Maps genre IDs to embedding vectors.
- **Forward Pass**:
    - `score = (user_embed * (movie_embed + genre_embed)).sum(dim=1)`
- **Loss Function**: Mean Squared Error (MSE) against actual ratings.
- **Optimizer**: Adam.

## Configuration

You can adjust hyperparameters and file paths in `src/config.py`:

- `EMBEDDING_SIZE`
- `NUM_USERS`
- `NUM_RATINGS`
- `EPOCHS`
- `LEARNING_RATE`

## Docker (Optional)

Build and run the containerized application:

```bash
docker build -t ai-recommender .
docker run -p 8000:8000 ai-recommender
```