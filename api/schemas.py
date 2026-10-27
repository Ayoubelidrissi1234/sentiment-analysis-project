from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def texte_non_vide(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Le texte ne peut pas être vide")
        return v

class PredictionResponse(BaseModel):
    label: str
    score: float
