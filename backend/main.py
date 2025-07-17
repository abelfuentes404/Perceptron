from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from trainer import train_model
from perceptron import SimplePerceptron
from utils import load_weights
import numpy as np

app = FastAPI(title="Perceptrón Lógico API")

class InputData(BaseModel):
    x1: int
    x2: int

@app.post("/train/")
def train(logic: str = Query("and", enum=["and", "or"]), epochs: int = 20):
    model = train_model(logic, epochs)
    return {
        "message": f"Modelo entrenado con compuerta {logic.upper()}",
        "errors": model.errors,
        "weights": model.get_weights()
    }

@app.post("/predict/")
def predict(data: InputData):
    weights = load_weights()
    if weights is None:
        return {"error": "Modelo no entrenado aún."}
    model = SimplePerceptron(input_size=2)
    model.set_weights(weights)
    prediction = model.predict([data.x1, data.x2])
    return {
        "entrada": [data.x1, data.x2],
        "predicción": round(prediction, 4),
        "resultado": 1 if prediction > 0.5 else 0
    }

@app.get("/weights/")
def get_weights():
    weights = load_weights()
    return {"weights": weights} if weights else {"error": "Modelo no entrenado"}

