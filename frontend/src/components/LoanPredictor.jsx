import React, { useState } from 'react';
import axios from 'axios';

// Definir la URL de la API (debería ser la misma que en App.jsx)
const API_URL = 'http://localhost:8000'; // Asegúrate que coincida con tu backend

export default function LoanPredictor() {
  const [loanInput, setLoanInput] = useState({
    loan_amount: '',
    monthly_income: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [activePredictionId, setActivePredictionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    if (!loanInput.loan_amount || !loanInput.monthly_income) {
      setError('Por favor ingresa ambos valores');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_URL}/predict/loan`, {
        loan_amount: parseFloat(loanInput.loan_amount),
        monthly_income: parseFloat(loanInput.monthly_income)
      });

      const newPrediction = {
        ...response.data,
        id: Date.now(), // ID único basado en timestamp
        confirmed: null // Inicialmente no confirmado
      };

      setPrediction(newPrediction);
      setPredictionHistory(prev => [newPrediction, ...prev]);
      setActivePredictionId(newPrediction.id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al realizar la predicción');
      console.error("Error en handlePredict:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const confirmPrediction = async (isCorrect) => {
    if (!activePredictionId) {
      setError("No hay una predicción activa para confirmar");
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      // 1. Actualizar el estado local primero
      const updatedHistory = predictionHistory.map(item => 
        item.id === activePredictionId 
          ? { ...item, confirmed: isCorrect } 
          : item
      );
      setPredictionHistory(updatedHistory);

      // 2. Enviar confirmación al backend
      const response = await axios.post(
        `${API_URL}/confirm-prediction/${activePredictionId}`,
        { is_correct: isCorrect },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      console.log("Confirmación exitosa:", response.data);

      // 3. Opcional: Reentrenar el modelo
      try {
        const retrainResponse = await axios.post(`${API_URL}/retrain-loan-model`);
        console.log("Modelo reentrenado:", retrainResponse.data);
      } catch (retrainError) {
        console.warn("Error en reentrenamiento:", retrainError.response?.data);
      }

      alert(`Predicción marcada como ${isCorrect ? 'correcta' : 'incorrecta'}`);
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 
                         error.message || 
                         "Error desconocido al confirmar";
      setError(`Error: ${errorMessage}`);
      console.error("Error en confirmPrediction:", {
        error,
        response: error.response,
        request: error.request
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="loan-container">
      <h2>Simulador de Préstamos</h2>
      
      {error && <div className="error">{error}</div>}

      <div className="input-group">
        <label>Monto del préstamo:</label>
        <input
          type="number"
          value={loanInput.loan_amount}
          onChange={(e) => setLoanInput({...loanInput, loan_amount: e.target.value})}
          placeholder="Ej: 50000"
          disabled={isLoading}
        />
      </div>

      <div className="input-group">
        <label>Ingresos mensuales:</label>
        <input
          type="number"
          value={loanInput.monthly_income}
          onChange={(e) => setLoanInput({...loanInput, monthly_income: e.target.value})}
          placeholder="Ej: 2000"
          disabled={isLoading}
        />
      </div>

      <button onClick={handlePredict} disabled={isLoading}>
        {isLoading ? 'Procesando...' : 'Predecir'}
      </button>

      {prediction && (
        <div className="prediction-result">
          <h3>Resultado</h3>
          <p><strong>Préstamo:</strong> ${prediction.loan_amount}</p>
          <p><strong>Ingresos:</strong> ${prediction.monthly_income}</p>
          <p className={`decision ${prediction.will_repay ? 'approve' : 'reject'}`}>
            Decisión: {prediction.result}
          </p>
          
          <div className="confirmation-buttons">
            <p>¿La predicción fue correcta?</p>
            <button 
              onClick={() => confirmPrediction(true)} 
              disabled={isLoading}
            >
              ✅ Sí
            </button>
            <button 
              onClick={() => confirmPrediction(false)} 
              disabled={isLoading}
            >
              ❌ No
            </button>
          </div>
        </div>
      )}

      {predictionHistory.length > 0 && (
        <div className="history-section">
          <h3>Historial de Predicciones</h3>
          <ul>
            {predictionHistory.map((item) => (
              <li key={item.id}>
                ${item.loan_amount} | ${item.monthly_income} → 
                {item.will_repay ? "Pagarà" : "No pagarà"} | 
                {item.confirmed === true ? "✅" : 
                 item.confirmed === false ? "❌" : "❓"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}