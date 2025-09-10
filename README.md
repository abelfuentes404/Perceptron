# 🧠 Perceptrón Clásico y Cuántico  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-purple)](https://qiskit.org/)  

Un proyecto de investigación que implementa un **perceptrón simple** para el aprendizaje de compuertas lógicas y un caso práctico de predicción de préstamos. Incluye una versión **clásica** y otra **cuántica** desarrolladas en Python, mostrando cómo el aprendizaje automático puede representarse en modelos tradicionales y en un enfoque inspirado en la computación cuántica.  

---

## 📌 Características  

- 🏗️ **Perceptrón Clásico**  
  - Aprende a simular compuertas lógicas (AND, OR, NAND, etc.).  
  - Guarda su aprendizaje en archivos `.pkl`.  

- ⚛️ **Perceptrón Cuántico**  
  - Implementación experimental de un perceptrón utilizando principios de computación cuántica.  
  - Permite comparar desempeño entre el modelo clásico y el cuántico.  

- 📊 **Dataset Personalizado**  
  - Datos de préstamos generados automáticamente (`crear_dataset.py`).  
  - Contiene columnas como: **Monto del préstamo**, **Ingresos mensuales**, y **¿Pagará? (Sí/No)**.  
  - Regla base: *Si los ingresos mensuales son menores al préstamo → el cliente no podrá pagar*.  

---

## 📂 Estructura del Proyecto  

```bash
Perceptron/
├── clasica/                   # Perceptrón clásico (compuertas lógicas, guardado de aprendizaje)
│   ├── perceptron.py
│   ├── modelos.pkl
│   └── ...
├── crear_dataset.py            # Genera dataset de préstamos
├── perceptron-simple-cuantico.py  # Implementación cuántica
├── prestamos_dataset.csv       # Dataset generado de préstamos
├── requirements.txt            # Librerías necesarias
├── *.pkl                       # Archivos con el aprendizaje guardado
└── README.md                   # Documentación del proyecto


🚀 Instalación y Uso
1️⃣ Clonar el repositorio

git clone https://github.com/abelfuentes404/Perceptron.git
cd Perceptron

2️⃣ Crear entorno virtual (opcional, recomendado)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Instalar dependencias
pip install -r requirements.txt

4️⃣ Generar dataset (si no existe)
python crear_dataset.py

5️⃣ Ejecutar perceptrón clásico
python clasica/perceptron.py

6️⃣ Ejecutar perceptrón cuántico
python perceptron-simple-cuantico.py

📊 Dataset de Ejemplo
| Monto Préstamo | Ingresos Mensuales | Pagará |
| -------------- | ------------------ | ------ |
| 5000           | 8000               | 1 (Sí) |
| 7000           | 3000               | 0 (No) |
| 2000           | 2500               | 1 (Sí) |
| 9000           | 4000               | 0 (No) |

🧩 Conceptos Clave
🔹 Perceptrón Clásico

El perceptrón es un modelo matemático inspirado en las neuronas biológicas.

Entrada: valores (ej. ingresos, monto).

Suma ponderada con pesos.

Función de activación (ej. escalón).

Salida: clasificación (Sí/No).

🔹 Perceptrón Cuántico

Usa representaciones en vectores de estado cuántico y compuertas cuánticas.

Explora cómo la superposición y entrelazamiento pueden servir para el aprendizaje.

En este proyecto se usa de forma experimental, comparando contra el modelo clásico.

📦 Requisitos

Las principales dependencias están en requirements.txt, pero incluyen:

numpy

pandas

scikit-learn

qiskit (para la parte cuántica)

📈 Futuras Mejoras

Extender a perceptrones multicapa (MLP).

Comparar métricas de desempeño entre el modelo clásico y cuántico.

Visualizaciones gráficas de fronteras de decisión.

Integración con datasets más complejos.

👨‍💻 Autor

Abel Fuentes Guzmán
https://github.com/abelfuentes404
