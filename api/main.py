from fastapi import FastAPI
from api.schemas import PredictionRequest, PredictionResponse
from api.model import predict, charger_modele

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
