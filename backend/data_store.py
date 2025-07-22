import json
import os
from datetime import datetime
from typing import List, Dict
import filelock  # Necesitarás instalarlo: pip install filelock

LOAN_DATA_FILE = "loan_predictions.json"
LOCK_FILE = "loan_predictions.lock"

def get_lock():
    return filelock.FileLock(LOCK_FILE, timeout=5)

def initialize_loan_file():
    """Asegura que el archivo existe y es válido"""
    with get_lock():
        if not os.path.exists(LOAN_DATA_FILE):
            with open(LOAN_DATA_FILE, 'w') as f:
                json.dump([], f)
        else:
            # Verificar integridad del archivo
            try:
                with open(LOAN_DATA_FILE, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                # Si está corrupto, lo reiniciamos
                with open(LOAN_DATA_FILE, 'w') as f:
                    json.dump([], f)

def save_loan_prediction(input_data: Dict, prediction: Dict):
    """Guarda una predicción de manera atómica y segura"""
    initialize_loan_file()
    with get_lock():
        try:
            # Leer datos existentes
            with open(LOAN_DATA_FILE, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
            
            # Crear nueva entrada
            new_entry = {
                "loan_amount": float(input_data["loan_amount"]),
                "monthly_income": float(input_data["monthly_income"]),
                "binary_input": prediction["binary_input"],
                "prediction": bool(prediction["will_repay"]),
                "probability": float(prediction["probability"]),
                "confirmed": None,
                "timestamp": datetime.now().isoformat(),
                "id": len(existing_data)  # ID único basado en posición
            }
            
            # Escribir de manera atómica
            temp_file = LOAN_DATA_FILE + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(existing_data + [new_entry], f, indent=2)
            
            # Reemplazar archivo original
            os.replace(temp_file, LOAN_DATA_FILE)
            
        except Exception as e:
            print(f"Error crítico guardando predicción: {str(e)}")
            raise

def get_loan_data() -> List[Dict]:
    """Obtiene los datos de préstamos de manera segura"""
    initialize_loan_file()
    with get_lock():
        try:
            with open(LOAN_DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
        except Exception as e:
            print(f"Error leyendo datos: {str(e)}")
            return []

def update_prediction_confirmation(index: int, is_correct: bool) -> bool:
    """Actualiza una confirmación de manera atómica"""
    initialize_loan_file()
    with get_lock():
        try:
            with open(LOAN_DATA_FILE, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return False
            
            if 0 <= index < len(data):
                # Crear copia actualizada
                updated_data = data.copy()
                updated_data[index]["confirmed"] = bool(is_correct)
                
                # Escribir de manera atómica
                temp_file = LOAN_DATA_FILE + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(updated_data, f, indent=2)
                
                os.replace(temp_file, LOAN_DATA_FILE)
                return True
            return False
        except Exception as e:
            print(f"Error actualizando confirmación: {str(e)}")
            return False