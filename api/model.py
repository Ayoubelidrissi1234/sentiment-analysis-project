import joblib
from pathlib import Path

MODELE = Path(__file__).resolve().parent.parent / "models" / "sentiment.joblib"
_pipeline = None

def charger_modele():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODELE)
    return _pipeline

def predict(texte: str) -> tuple[str, float]:
    pipeline = charger_modele()
    label = pipeline.predict([texte])[0]
    proba = pipeline.predict_proba([texte])[0].max()
    return str(label), float(proba)
