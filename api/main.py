from fastapi import FastAPI

from api.model import charger_modele, predict
from api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="API Sentiment")

@app.on_event("startup")
def startup():
    charger_modele()

@app.get("/health")
def health():
    return {"status": "OK", "model": "tfidf-logreg"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: PredictionRequest):
    label, score = predict(req.text)
    return {"label": label, "score": score}
