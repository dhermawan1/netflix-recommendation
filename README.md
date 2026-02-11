# Netflix-Style Recommendation System

Production-style movie recommendation system built using PyTorch,
FastAPI, and embedding-based collaborative filtering.

This project demonstrates end-to-end ML engineering:
- Model training
- Feature engineering
- Real-time recommendation API
- Production deployment design

## Architecture

Embedding model
FastAPI serving
Vector-ready design
MLOps ready

## Setup

```bash
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload
```

## Run:

uvicorn api.main:app --reload

## Open:

http://127.0.0.1:8000/docs

## Docker Build:

docker build -t ai-recommender .
docker run -p 8000:8000 ai-recommender