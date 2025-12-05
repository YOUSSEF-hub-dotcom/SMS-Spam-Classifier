from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc


model = mlflow.pyfunc.load_model("runs:/9b22cab5ecdb462481f66fd0e0d5e9fd/spam_classifier_model")

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
