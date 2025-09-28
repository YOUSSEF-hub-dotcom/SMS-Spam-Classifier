
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI(title="SMS Spam Classifier API")

class MessageInput(BaseModel):
    message: str

class PredictionOutput(BaseModel):
    label: str
    probability: float

@app.post("/predict", response_model=PredictionOutput)
def predict_message(input_data: MessageInput):
    if not input_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    X = vectorizer.transform([input_data.message])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    label = "Ham" if pred == 0 else "Spam"
    probability = float(max(proba))

    return {"label": label, "probability": probability}
