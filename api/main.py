from fastapi import FastAPI
from src.recommend import recommend_movies

app = FastAPI(title="AI Recommendation System")

@app.get("/")
def root():
    return {"message":"AI Recommendation API running"}

@app.get("/recommend/{user_id}")
def get_recommendation(user_id:int):
    recs = recommend_movies(user_id)
    return {"user_id":user_id,"recommendations":recs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
