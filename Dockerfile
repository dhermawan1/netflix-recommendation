FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Train model to generate data and artifacts
RUN python -m src.train

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
