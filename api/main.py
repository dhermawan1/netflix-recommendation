from fastapi import FastAPI
from src.recommend import recommend

app = FastAPI(title="AI Recommendation System")

@app.get("/")
def root():
    return {"message":"AI Recommendation API running"}

@app.get("/recommend/{user_id}")
def get_recommendation(user_id:int):
    recs = recommend(user_id)
    return {"user_id":user_id,"recommendations":recs}
