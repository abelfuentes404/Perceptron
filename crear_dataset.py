"""
Generador de Dataset para Predicción de Pagos de Préstamos
=========================================================
Este script crea un dataset sintético para entrenar el perceptrón
"""

import numpy as np
import pandas as pd
import random

def generar_dataset(num_registros=1000, nombre_archivo='prestamos_dataset.csv'):
    """
    Genera un dataset sintético de préstamos con patrones predecibles
    """
    print(f"🧠 Generando dataset con {num_registros} registros...")
    
    # Semilla para reproducibilidad
    np.random.seed(42)
    random.seed(42)
    
    # Generar datos sintéticos
    montos = np.random.uniform(500, 10000, num_registros)
    ingresos = np.random.uniform(1000, 5000, num_registros)
    
    # Crear variable objetivo con lógica predecible + ruido
    pagara = []
    for monto, ingreso in zip(montos, ingresos):
        # Lógica básica: si el monto es menor al 30% del ingreso, es más probable que pague
        ratio = monto / ingreso
        
        # Probabilidad basada en el ratio
        if ratio < 0.3:
            probabilidad_pago = 0.8  # 80% de probabilidad de pago
        elif ratio < 0.6:
            probabilidad_pago = 0.5  # 50% de probabilidad de pago
        else:
            probabilidad_pago = 0.2  # 20% de probabilidad de pago
        
        # Añadir ruido aleatorio
        probabilidad_pago += random.uniform(-0.1, 0.1)
        probabilidad_pago = max(0, min(1, probabilidad_pago))  # Asegurar entre 0 y 1
        
        # Decidir si paga o no
        pagara.append(1 if random.random() < probabilidad_pago else 0)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'monto_prestamo': montos,
        'ingreso_mensual': ingresos,
        'pagara': pagara
    })
    
    # Guardar a CSV
    df.to_csv(nombre_archivo, index=False)
    print(f"✅ Dataset guardado como '{nombre_archivo}'")
    print(f"📊 Estadísticas del dataset:")
    print(f"   - Total de registros: {len(df)}")
    print(f"   - Pagarán: {df['pagara'].sum()} ({df['pagara'].mean()*100:.1f}%)")
    print(f"   - No pagarán: {len(df) - df['pagara'].sum()} ({(1-df['pagara'].mean())*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Generar dataset con 500 registros
    dataset = generar_dataset(500, 'prestamos_dataset.csv')
    
    # Mostrar primeras filas
    print("\n📋 Primeras 10 filas del dataset:")
    print(dataset.head(10))