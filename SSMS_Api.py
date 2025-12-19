from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc


model = mlflow.sklearn.load_model("runs:/e557a16afe5c4887857275510f0153cd/spam_classifier_model")

app = FastAPI(title="SMS Spam Classifier API with MLflow")

class MessageInput(BaseModel):
    message: str

class PredictionOutput(BaseModel):
    label: str
    probability: float

@app.post("/predict", response_model=PredictionOutput)
def predict_message(input_data: MessageInput):
    if not input_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    proba = model.predict_proba([input_data.message])[0]
    pred = model.predict([input_data.message])[0]

    label = "Ham" if pred == 0 else "Spam"
    probability = float(max(proba))

    return {"label": label, "probability": probability}
