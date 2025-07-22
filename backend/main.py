from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from trainer import train_logic_model
from perceptron import SimplePerceptron
from utils import load_weights, ensure_python_types
import numpy as np
from data_store import save_loan_prediction, get_loan_data, update_prediction_confirmation

app = FastAPI(title="Perceptrón para Préstamos")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LogicInput(BaseModel):
    x1: int
    x2: int

class LoanInput(BaseModel):
    loan_amount: float
    monthly_income: float

LOAN_THRESHOLD = 3000
INCOME_THRESHOLD = 2000

@app.post("/train/logic")
async def train_logic(logic: str = Query("and", enum=["and", "or"]), epochs: int = 20):
    try:
        _, results = train_logic_model(logic, epochs)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/confirm-prediction/{prediction_id}")
async def confirm_prediction(prediction_id: int, confirmation: dict = Body(...)):
    try:
        # Validación exhaustiva
        if not isinstance(confirmation.get('is_correct'), bool):
            raise HTTPException(
                status_code=422,
                detail="El campo 'is_correct' debe ser un booleano"
            )

        # Obtener todos los datos
        all_data = get_loan_data()
        
        # Encontrar la predicción por ID
        prediction_index = next(
            (i for i, item in enumerate(all_data) if item.get("id") == prediction_id),
            None
        )
        
        if prediction_index is None:
            raise HTTPException(
                status_code=404,
                detail=f"Predicción con ID {prediction_id} no encontrada"
            )

        # Actualizar confirmación
        success = update_prediction_confirmation(prediction_index, confirmation['is_correct'])
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="No se pudo actualizar la confirmación"
            )

        # Respuesta detallada
        return {
            "status": "success",
            "message": "Confirmación registrada correctamente",
            "prediction_id": prediction_id,
            "is_correct": confirmation['is_correct'],
            "updated_entry": all_data[prediction_index]
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log detallado del error
        print(f"Error en confirm_prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.post("/predict/logic")
async def predict_logic(data: LogicInput):
    try:
        if data.x1 not in [0, 1] or data.x2 not in [0, 1]:
            raise HTTPException(status_code=400, detail="Las entradas deben ser 0 o 1")
        
        weights = load_weights("logic_weights.json")
        if weights is None:
            raise HTTPException(status_code=400, detail="Primero entrene el modelo")
        
        model = SimplePerceptron(input_size=2)
        model.set_weights(weights)
        prediction = model.predict([data.x1, data.x2])
        
        return ensure_python_types({
            "input": [data.x1, data.x2],
            "output": prediction,
            "result": int(prediction > 0.5)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/loan")
async def predict_loan(data: LoanInput):
    try:
        loan_risk = 1 if data.loan_amount > LOAN_THRESHOLD else 0
        income_security = 1 if data.monthly_income > INCOME_THRESHOLD else 0
        will_repay = not (loan_risk and not income_security)

        prediction = {
            "loan_amount": data.loan_amount,
            "monthly_income": data.monthly_income,
            "binary_input": [loan_risk, income_security],
            "probability": float(will_repay),
            "will_repay": will_repay
        }

        save_loan_prediction(data.dict(), prediction)
        # Obtener ID real (último insertado)
        full_data = get_loan_data()
        real_id = full_data[-1]["id"]

        return {
            **prediction,
            "result": "Pagarà" if will_repay else "No pagarà",
            "id": real_id,  # <--- devolver ID real
            "thresholds": {
                "loan": LOAN_THRESHOLD,
                "income": INCOME_THRESHOLD
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thresholds")
async def get_thresholds():
    return ensure_python_types({
        "loan_threshold": LOAN_THRESHOLD,
        "income_threshold": INCOME_THRESHOLD
    })

@app.get("/")
async def root():
    return {"message": "API de Perceptrón Lógico - Usa /train/logic y /predict/loan"}

@app.post("/retrain-loan-model")
async def retrain_loan_model(epochs: int = 10):
    try:
        all_data = get_loan_data()
        confirmed_data = [d for d in all_data if d["confirmed"] is not None]
        
        if len(confirmed_data) < 3:  # Reducido a 3 ejemplos mínimos
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos 3 predicciones confirmadas (tienes {len(confirmed_data)})"
            )
        
        X = []
        y = []
        for entry in confirmed_data:
            X.append(entry["binary_input"])
            y.append(1 if entry["confirmed"] else 0)
        
        model = SimplePerceptron(input_size=2)
        
        # Cargar pesos existentes o usar los predeterminados
        weights = load_weights("logic_weights.json")
        if weights:
            model.set_weights(weights)
        
        # Entrenar con más epochs para mejor aprendizaje
        model.train(np.array(X), np.array(y), epochs=50)
        
        # Guardar nuevos pesos
        save_weights(model.get_weights(), "logic_weights.json")
        
        return {
            "message": f"Modelo reentrenado con {len(confirmed_data)} ejemplos",
            "new_weights": model.get_weights(),
            "error_evolution": model.errors  # Para monitorear el aprendizaje
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))