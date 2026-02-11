
from src.recommend import recommend_movies
import numpy as np

try:
    recs = recommend_movies(0)
    print("Recommendations:", recs)
    if recs:
        first_rec = recs[0]
        print("Type of first element:", type(first_rec))
        print("Type of title:", type(first_rec[0]))
        print("Type of genre:", type(first_rec[1]))
        print("Type of rating:", type(first_rec[2]))
        
        # Check for numpy types
        if isinstance(first_rec[2], (np.float32, np.float64)):
            print("Rating is numpy float")
except Exception as e:
    print("Error:", e)
