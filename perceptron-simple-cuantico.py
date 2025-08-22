"""
Perceptrón Híbrido Clásico-Cuántico con Carga de CSV - VERSIÓN CORREGIDA
=======================================================================
"""

import numpy as np
import pandas as pd
import pickle
import random
import os
from collections import deque

# Verificación de librerías cuánticas
try:
    # Para Qiskit 1.x
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit.primitives import Sampler
    
    # Configurar semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    
    QUANTUM_AVAILABLE = True
    print("✓ Librerías cuánticas detectadas correctamente")
    
except ImportError as e:
    print(f"✗ Error importando librerías cuánticas: {e}")
    QUANTUM_AVAILABLE = False


class PerceptronSimple:
    def __init__(self, tasa_aprendizaje=0.01, n_iteraciones=100):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.n_iteraciones = n_iteraciones
        self.pesos = None
        self.sesgo = None
        self.errores = []
        self.media = None
        self.desviacion = None

    def funcion_activacion(self, x):
        return 1 if x >= 0 else 0

    def predecir(self, x):
        suma_ponderada = np.dot(x, self.pesos) + self.sesgo
        return self.funcion_activacion(suma_ponderada)

    def entrenar(self, X, y):
        n_muestras, n_caracteristicas = X.shape
        
        if self.pesos is None:
            self.pesos = np.zeros(n_caracteristicas)
        if self.sesgo is None:
            self.sesgo = 0

        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        
        X_norm = (X - self.media) / self.desviacion

        for epoca in range(self.n_iteraciones):
            error = 0
            for idx, x_i in enumerate(X_norm):
                prediccion = self.predecir(x_i)
                update = self.tasa_aprendizaje * (y[idx] - prediccion)
                self.pesos += update * x_i
                self.sesgo += update
                error += int(update != 0.0)
            self.errores.append(error)
            
            if (epoca + 1) % 10 == 0:
                print(f"Época {epoca + 1}/{self.n_iteraciones}, Error: {error}")

    def aprender_online(self, X, y):
        """Aprendizaje en línea con un solo dato"""
        if self.media is None or self.desviacion is None:
            return
            
        X_norm = (X - self.media) / self.desviacion
        prediccion = self.predecir(X_norm)
        update = self.tasa_aprendizaje * (y - prediccion)
        self.pesos += update * X_norm
        self.sesgo += update

    def normalizar_dato(self, x):
        if self.media is None or self.desviacion is None:
            return x
        return (x - self.media) / self.desviacion

    def guardar_modelo(self, archivo='perceptron_modelo.pkl'):
        with open(archivo, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos, 
                'sesgo': self.sesgo,
                'media': self.media,
                'desviacion': self.desviacion,
                'tasa_aprendizaje': self.tasa_aprendizaje,
                'n_iteraciones': self.n_iteraciones
            }, f)
        print(f"💾 Modelo clásico guardado en {archivo}")

    def cargar_modelo(self, archivo='perceptron_modelo.pkl'):
        try:
            with open(archivo, 'rb') as f:
                datos = pickle.load(f)
            self.pesos = datos['pesos']
            self.sesgo = datos['sesgo']
            self.media = datos.get('media', None)
            self.desviacion = datos.get('desviacion', None)
            self.tasa_aprendizaje = datos.get('tasa_aprendizaje', 0.01)
            self.n_iteraciones = datos.get('n_iteraciones', 100)
            print(f"📂 Modelo clásico cargado desde {archivo}")
            return True
        except FileNotFoundError:
            print("❌ No se encontró un modelo clásico previo.")
            return False
        except KeyError as e:
            print(f"❌ Error al cargar el modelo: {e}. Iniciando desde cero.")
            return False


if QUANTUM_AVAILABLE:
    class PerceptronCuantico:
        def __init__(self):
            self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
            self.ansatz = RealAmplitudes(num_qubits=2, reps=1)
            self.optimizer = COBYLA(maxiter=50)
            self.sampler = Sampler()
            self.modelo = None
            self.X_entrenamiento = None
            self.y_entrenamiento = None
            self.esta_entrenado = False

        def entrenar(self, X, y):
            self.X_entrenamiento = X
            self.y_entrenamiento = y
            X_normalizado = self._normalizar_datos_cuantico(X)
            
            self.modelo = VQC(
                sampler=self.sampler,
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
            )
            
            print("🔮 Entrenando modelo cuántico (simulación)...")
            self.modelo.fit(X_normalizado, y)
            print("✅ Entrenamiento cuántico completado.")
            self.esta_entrenado = True

        def predecir(self, X):
            if not self.esta_entrenado or self.modelo is None:
                raise ValueError("Modelo no entrenado. Llama a entrenar() primero.")
                
            X_normalizado = self._normalizar_datos_cuantico(X)
            resultado = self.modelo.predict(X_normalizado)
            
            # Asegurarnos de devolver siempre un array
            if isinstance(resultado, (int, float, np.number)):
                return np.array([resultado])
            elif hasattr(resultado, 'shape') and resultado.shape == ():
                # Array 0-dimensional (escalar)
                return np.array([resultado.item()])
            return resultado

        def _normalizar_datos_cuantico(self, X):
            if self.X_entrenamiento is None:
                return X
                
            X_min = np.min(self.X_entrenamiento, axis=0)
            X_max = np.max(self.X_entrenamiento, axis=0)
            # Evitar división por cero
            if np.any(X_max - X_min == 0):
                return np.zeros_like(X)
            return 2 * (X - X_min) / (X_max - X_min) - 1

        def guardar_modelo(self, archivo='perceptron_cuantico.pkl'):
            print("⚠️  Advertencia: El guardado de modelos cuánticos es limitado.")
            with open(archivo, 'wb') as f:
                pickle.dump({
                    'feature_map': self.feature_map,
                    'ansatz': self.ansatz,
                    'optimizer': self.optimizer,
                    'X_entrenamiento': self.X_entrenamiento,
                    'y_entrenamiento': self.y_entrenamiento,
                    'esta_entrenado': self.esta_entrenado
                }, f)
            print(f"💾 Configuración cuántica guardada en {archivo}")

        def cargar_modelo(self, archivo='perceptron_cuantico.pkl'):
            try:
                with open(archivo, 'rb') as f:
                    datos = pickle.load(f)
                self.feature_map = datos['feature_map']
                self.ansatz = datos['ansatz']
                self.optimizer = datos['optimizer']
                self.X_entrenamiento = datos.get('X_entrenamiento', None)
                self.y_entrenamiento = datos.get('y_entrenamiento', None)
                self.esta_entrenado = datos.get('esta_entrenado', False)
                
                if self.esta_entrenado and self.X_entrenamiento is not None:
                    # Recrear el modelo con los datos de entrenamiento
                    self.modelo = VQC(
                        sampler=self.sampler,
                        feature_map=self.feature_map,
                        ansatz=self.ansatz,
                        optimizer=self.optimizer,
                    )
                    # Reentrenar rápidamente para restaurar el estado
                    X_normalizado = self._normalizar_datos_cuantico(self.X_entrenamiento)
                    self.modelo.fit(X_normalizado, self.y_entrenamiento)
                
                print(f"📂 Configuración cuántica cargada desde {archivo}")
                return True
            except FileNotFoundError:
                print("❌ No se encontró un modelo cuántico previo.")
                return False
            except Exception as e:
                print(f"❌ Error al cargar el modelo cuántico: {e}")
                return False


def cargar_dataset(nombre_archivo='prestamos_dataset.csv'):
    """
    Carga el dataset desde un archivo CSV
    """
    try:
        df = pd.read_csv(nombre_archivo)
        print(f"📊 Dataset cargado: {len(df)} registros")
        
        # Separar características y etiquetas
        X = df[['monto_prestamo', 'ingreso_mensual']].values
        y = df['pagara'].values
        
        print(f"   - Características: {X.shape}")
        print(f"   - Etiquetas: {y.shape}")
        print(f"   - Distribución: {y.sum()} pagarán, {len(y)-y.sum()} no pagarán")
        
        return X, y
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {nombre_archivo}")
        print("   Ejecuta primero: python crear_dataset.py")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return None, None


def es_respuesta_afirmativa(respuesta):
    return respuesta.lower() in ['sí', 'si', 's', 'yes', 'y', '1', 'true', 'verdadero', 'afirmativo']


def es_respuesta_negativa(respuesta):
    return respuesta.lower() in ['no', 'n', '0', 'false', 'falso', 'negativo']


def mostrar_menu_principal():
    print("\n" + "="*60)
    print("🤖 PERCEPTRÓN PARA PREDICCIÓN DE PAGOS DE PRÉSTAMOS")
    print("="*60)
    print("¿Qué algoritmo deseas usar?")
    print("1. Perceptrón Clásico")
    if QUANTUM_AVAILABLE:
        print("2. Perceptrón Cuántico (Simulación)")
    else:
        print("2. Perceptrón Cuántico (No disponible)")
    print("3. Salir")
    

def main():
    # Cargar dataset
    X, y = cargar_dataset()
    
    if X is None or y is None:
        return
    
    # Bucle principal
    while True:
        mostrar_menu_principal()
        opcion = input("Selecciona una opción (1-3): ")
        
        if opcion == "3":
            print("👋 ¡Hasta luego!")
            break
            
        elif opcion == "1":
            print("\n--- 🧠 MODO CLÁSICO ---")
            perceptron = PerceptronSimple(tasa_aprendizaje=0.01, n_iteraciones=100)
            modelo_cargado = perceptron.cargar_modelo()
            
            if not modelo_cargado:
                print("📚 Entrenando modelo clásico...")
                perceptron.entrenar(X, y)
                perceptron.guardar_modelo()
                
            ejecutar_modo_clasico(perceptron)
            
        elif opcion == "2" and QUANTUM_AVAILABLE:
            print("\n--- 🔮 MODO CUÁNTICO (SIMULACIÓN) ---")
            perceptron = PerceptronCuantico()
            modelo_cargado = perceptron.cargar_modelo()
            
            if not modelo_cargado or not perceptron.esta_entrenado:
                print("📚 Entrenando modelo cuántico...")
                perceptron.entrenar(X, y)
                perceptron.guardar_modelo()
                
            ejecutar_modo_cuantico(perceptron)
            
        else:
            print("❌ Opción no válida. Por favor, selecciona 1, 2 o 3.")


def ejecutar_modo_clasico(perceptron):
    while True:
        print("\n--- 🔍 Predicción de Pago (Clásico) ---")
        try:
            monto = float(input("Monto del préstamo: "))
            ingreso = float(input("Ingreso mensual del usuario: "))
        except ValueError:
            print("❌ Por favor, ingresa números válidos.")
            continue

        # Normalizar y predecir
        X_nuevo = np.array([monto, ingreso])
        X_nuevo_norm = perceptron.normalizar_dato(X_nuevo)
        prediccion = perceptron.predecir(X_nuevo_norm)
        
        resultado = "✅ PAGARÁ" if prediccion == 1 else "❌ NO PAGARÁ"
        print(f"🎯 Predicción: {resultado}")

        # Feedback y aprendizaje online
        corroboracion = input("¿La predicción fue correcta? (sí/no): ")
        
        if es_respuesta_negativa(corroboracion):
            try:
                verdadera_etiqueta = int(input("Ingresa la etiqueta correcta (1 para PAGÓ, 0 para NO PAGÓ): "))
                
                # Ajustar pesos con aprendizaje online
                perceptron.aprender_online(X_nuevo, verdadera_etiqueta)
                print("🔄 Modelo actualizado con tu feedback.")
                perceptron.guardar_modelo()
            except ValueError:
                print("❌ Etiqueta no válida. No se actualizó el modelo.")
        else:
            print("✅ Predicción correcta. Refuerzo positivo aplicado.")
            perceptron.aprender_online(X_nuevo, prediccion)
            perceptron.guardar_modelo()

        # Preguntar si desea continuar
        continuar = input("¿Predecir otro? (sí/no): ")
        if es_respuesta_negativa(continuar):
            break


def ejecutar_modo_cuantico(perceptron):
    while True:
        print("\n--- 🔍 Predicción de Pago (Cuántico) ---")
        try:
            monto = float(input("Monto del préstamo: "))
            ingreso = float(input("Ingreso mensual del usuario: "))
        except ValueError:
            print("❌ Por favor, ingresa números válidos.")
            continue

        # Predecir
        X_nuevo = np.array([[monto, ingreso]])
        try:
            prediccion_result = perceptron.predecir(X_nuevo)
            
            # Manejar diferentes formatos de retorno
            if hasattr(prediccion_result, 'shape') or hasattr(prediccion_result, '__len__'):
                if hasattr(prediccion_result, 'shape') and prediccion_result.shape == ():
                    prediccion = float(prediccion_result)
                else:
                    prediccion = prediccion_result[0] if len(prediccion_result) > 0 else 0
            else:
                prediccion = float(prediccion_result)
                
            resultado = "✅ PAGARÁ" if prediccion == 1 else "❌ NO PAGARÁ"
            print(f"🎯 Predicción: {resultado}")

            # Feedback
            corroboracion = input("¿La predicción fue correcta? (sí/no): ")
            
            if es_respuesta_negativa(corroboracion):
                try:
                    verdadera_etiqueta = int(input("Ingresa la etiqueta correcta (1 para PAGÓ, 0 para NO PAGÓ): "))
                    print("📝 Para el modo cuántico, el aprendizaje requiere reentrenamiento completo.")
                    print("   Ejecuta el entrenamiento nuevamente para incorporar este dato.")
                except ValueError:
                    print("❌ Etiqueta no válida.")
            else:
                print("✅ Predicción correcta.")

        except Exception as e:
            print(f"❌ Error en predicción cuántica: {e}")
            print("💡 Solución: Ejecuta el entrenamiento nuevamente.")

        # Preguntar si desea continuar
        continuar = input("¿Predecir otro? (sí/no): ")
        if es_respuesta_negativa(continuar):
            break


if __name__ == "__main__":
    # Ejecutar el programa principal
    main()