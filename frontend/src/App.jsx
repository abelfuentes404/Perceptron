import { useState } from 'react';
import LogicGateTrainer from './components/LogicGateTrainer';
import LoanPredictor from './components/LoanPredictor';
import './styles.css';

export default function App() {
  const [mode, setMode] = useState('logic'); // 'logic' o 'loan'

  return (
    <div className="app-container">
      <h1>Sistema Perceptrón Dual Mode</h1>
      
      <div className="mode-selector">
        <button 
          onClick={() => setMode('logic')} 
          className={mode === 'logic' ? 'active' : ''}
        >
          Modo Compuertas
        </button>
        <button 
          onClick={() => setMode('loan')} 
          className={mode === 'loan' ? 'active' : ''}
        >
          Modo Préstamos
        </button>
      </div>
      
      {mode === 'logic' ? <LogicGateTrainer /> : <LoanPredictor />}
    </div>
  );
}