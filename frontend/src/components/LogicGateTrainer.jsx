import { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, Title, CategoryScale);

const API_URL = 'http://localhost:8000';

export default function LogicGateTrainer() {
    const [logicGate, setLogicGate] = useState('and');
    const [epochs, setEpochs] = useState(20);
    const [trainingResult, setTrainingResult] = useState(null);
    const [input1, setInput1] = useState(0);
    const [input2, setInput2] = useState(0);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleTrain = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.post(
                `${API_URL}/train/logic?logic=${logicGate}&epochs=${epochs}`
            );
            setTrainingResult(response.data);
            setPrediction(null);
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = async () => {
        setError('');
        try {
            const response = await axios.post(`${API_URL}/predict/logic`, {
                x1: parseInt(input1),
                x2: parseInt(input2)
            });
            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        }
    };

    const chartData = trainingResult?.errors ? {
        labels: trainingResult.errors.map((_, i) => i + 1),
        datasets: [{
            label: 'Error durante el entrenamiento',
            data: trainingResult.errors,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    } : null;

    return (
        <div className="logic-container">
            <h2>Compuertas Lógicas</h2>
            
            <div className="training-section">
                <div className="form-group">
                    <label>Compuerta:</label>
                    <select 
                        value={logicGate} 
                        onChange={(e) => setLogicGate(e.target.value)}
                        disabled={loading}
                    >
                        <option value="and">AND</option>
                        <option value="or">OR</option>
                    </select>
                </div>
                
                <div className="form-group">
                    <label>Épocas:</label>
                    <input 
                        type="number" 
                        value={epochs} 
                        onChange={(e) => setEpochs(e.target.value)} 
                        min="1" 
                        max="1000"
                        disabled={loading}
                    />
                </div>
                
                <button onClick={handleTrain} disabled={loading}>
                    {loading ? 'Entrenando...' : 'Entrenar'}
                </button>
                
                {trainingResult && (
                    <div className="results">
                        <h3>Resultados</h3>
                        <p>{trainingResult.message}</p>
                        {chartData && (
                            <div className="chart-container">
                                <Line data={chartData} />
                            </div>
                        )}
                    </div>
                )}
            </div>
            
            <div className="prediction-section">
                <h3>Probar Compuerta</h3>
                <div className="input-group">
                    <label>Entrada X1:</label>
                    <select 
                        value={input1} 
                        onChange={(e) => setInput1(e.target.value)}
                        disabled={!trainingResult}
                    >
                        <option value="0">0</option>
                        <option value="1">1</option>
                    </select>
                </div>
                
                <div className="input-group">
                    <label>Entrada X2:</label>
                    <select 
                        value={input2} 
                        onChange={(e) => setInput2(e.target.value)}
                        disabled={!trainingResult}
                    >
                        <option value="0">0</option>
                        <option value="1">1</option>
                    </select>
                </div>
                
                <button onClick={handlePredict} disabled={!trainingResult}>
                    Predecir
                </button>
                
                {prediction && (
                    <div className="prediction-result">
                        <p>Entrada: [{prediction.input.join(', ')}]</p>
                        <p>Salida: {prediction.output.toFixed(4)}</p>
                        <p>Resultado: {prediction.result}</p>
                    </div>
                )}
            </div>
            
            {error && <div className="error">{error}</div>}
        </div>
    );
}