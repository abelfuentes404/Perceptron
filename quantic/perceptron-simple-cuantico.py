"""
Perceptrón Híbrido Clásico-Cuántico para Predicción de Pagos de Préstamos
=======================================================================

Este script implementa dos versiones de un perceptrón para predecir si un usuario
pagará un préstamo basado en el monto del préstamo y su ingreso mensual:

1. Perceptrón Simple (clásico): Implementación tradicional con aprendizaje en línea
2. Perceptrón Cuántico: Implementación que simula un circuito cuántico variational

Autor: [Tu nombre]
Fecha: [Fecha]
Licencia: MIT
"""

import numpy as np
import pickle
from typing import Union, Tuple, List

# Try to import quantum libraries (optional)
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit.utils import algorithm_globals
    from qiskit import QuantumInstance
    QUANTUM_AVAILABLE = True
except ImportError:
    print("Advertencia: Librerías cuánticas no disponibles. Usando solo modo clásico.")
    QUANTUM_AVAILABLE = False


class PerceptronSimple:
    """
    Implementación clásica de un perceptrón simple para clasificación binaria.
    
    Esta clase implementa un perceptrón con función de activación escalón (Heaviside)
    y capacidad de aprendizaje en línea con persistencia de los pesos aprendidos.
    
    Atributos:
        tasa_aprendizaje (float): Tasa de aprendizaje para el algoritmo.
        n_iteraciones (int): Número de iteraciones de entrenamiento.
        pesos (np.array): Vector de pesos del perceptrón.
        sesgo (float): Término de sesgo del perceptrón.
        errores (list): Historial de errores por época de entrenamiento.
        media (np.array): Media de los datos de entrenamiento para normalización.
        desviacion (np.array): Desviación estándar para normalización.
    """
    
    def __init__(self, tasa_aprendizaje: float = 0.01, n_iteraciones: int = 100):
        """
        Inicializa el perceptrón con los parámetros dados.
        
        Args:
            tasa_aprendizaje: Tasa de aprendizaje para el descenso de gradiente.
            n_iteraciones: Número de épocas de entrenamiento.
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.n_iteraciones = n_iteraciones
        self.pesos = None
        self.sesgo = None
        self.errores = []
        self.media = None
        self.desviacion = None

    def funcion_activacion(self, x: float) -> int:
        """
        Función de activación escalón (Heaviside).
        
        Args:
            x: Valor de entrada a la función de activación.
            
        Returns:
            1 si x >= 0, 0 en caso contrario.
        """
        return 1 if x >= 0 else 0

    def predecir(self, x: np.array) -> int:
        """
        Realiza una predicción para una instancia de entrada.
        
        Args:
            x: Vector de características de entrada (normalizado).
            
        Returns:
            Predicción: 1 (pagará) o 0 (no pagará).
        """
        suma_ponderada = np.dot(x, self.pesos) + self.sesgo
        return self.funcion_activacion(suma_ponderada)

    def entrenar(self, X: np.array, y: np.array):
        """
        Entrena el perceptrón con los datos proporcionados.
        
        Args:
            X: Matriz de características de entrenamiento.
            y: Vector de etiquetas de entrenamiento.
        """
        n_muestras, n_caracteristicas = X.shape
        
        # Inicializar pesos y sesgo si no existen
        if self.pesos is None:
            self.pesos = np.zeros(n_caracteristicas)
        if self.sesgo is None:
            self.sesgo = 0

        # Calcular y guardar parámetros de normalización
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        
        # Normalizar datos
        X_norm = (X - self.media) / self.desviacion

        # Entrenamiento por épocas
        for epoca in range(self.n_iteraciones):
            error = 0
            for idx, x_i in enumerate(X_norm):
                prediccion = self.predecir(x_i)
                update = self.tasa_aprendizaje * (y[idx] - prediccion)
                self.pesos += update * x_i
                self.sesgo += update
                error += int(update != 0.0)
            self.errores.append(error)
            
            # Mostrar progreso cada 10 épocas
            if (epoca + 1) % 10 == 0:
                print(f"Época {epoca + 1}/{self.n_iteraciones}, Error: {error}")

    def normalizar_dato(self, x: np.array) -> np.array:
        """
        Normaliza un dato usando los parámetros de los datos de entrenamiento.
        
        Args:
            x: Dato a normalizar.
            
        Returns:
            Dato normalizado.
        """
        if self.media is None or self.desviacion is None:
            return x
        return (x - self.media) / self.desviacion

    def guardar_modelo(self, archivo: str = 'perceptron_modelo.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        
        Args:
            archivo: Ruta del archivo donde guardar el modelo.
        """
        with open(archivo, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos, 
                'sesgo': self.sesgo,
                'media': self.media,
                'desviacion': self.desviacion,
                'tasa_aprendizaje': self.tasa_aprendizaje,
                'n_iteraciones': self.n_iteraciones
            }, f)
        print(f"Modelo clásico guardado en {archivo}")

    def cargar_modelo(self, archivo: str = 'perceptron_modelo.pkl') -> bool:
        """
        Carga un modelo previamente entrenado desde un archivo.
        
        Args:
            archivo: Ruta del archivo desde donde cargar el modelo.
            
        Returns:
            True si se cargó exitosamente, False en caso contrario.
        """
        try:
            with open(archivo, 'rb') as f:
                datos = pickle.load(f)
            self.pesos = datos['pesos']
            self.sesgo = datos['sesgo']
            self.media = datos.get('media', None)
            self.desviacion = datos.get('desviacion', None)
            self.tasa_aprendizaje = datos.get('tasa_aprendizaje', 0.01)
            self.n_iteraciones = datos.get('n_iteraciones', 100)
            print(f"Modelo clásico cargado desde {archivo}")
            return True
        except FileNotFoundError:
            print("No se encontró un modelo clásico previo.")
            return False
        except KeyError as e:
            print(f"Error al cargar el modelo: {e}. Iniciando desde cero.")
            return False


if QUANTUM_AVAILABLE:
    class PerceptronCuantico:
        """
        Implementación de un clasificador cuántico variational (VQC) para simulación.
        
        Esta clase implementa un perceptrón cuántico que utiliza un circuito cuántico
        variational para realizar clasificación binaria. Es una simulación en hardware
        clásico del comportamiento cuántico.
        
        Atributos:
            feature_map: Mapa de características cuántico (ZZFeatureMap).
            ansatz: Ansatz para el circuito variational (RealAmplitudes).
            optimizer: Optimizador para el entrenamiento (COBYLA).
            modelo: Modelo VQC de qiskit.
        """
        
        def __init__(self):
            """Inicializa el perceptrón cuántico con parámetros por defecto."""
            # Configurar semilla para reproducibilidad
            algorithm_globals.random_seed = 42
            
            # Mapa de características: transforma datos clásicos a espacio cuántico
            self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
            
            # Ansatz: circuito cuántico parametrizado (como los pesos en una red neuronal)
            self.ansatz = RealAmplitudes(num_qubits=2, reps=1)
            
            # Optimizador para el entrenamiento
            self.optimizer = COBYLA(maxiter=100)
            
            # Modelo VQC (Variational Quantum Classifier)
            self.modelo = None

        def entrenar(self, X: np.array, y: np.array):
            """
            Entrena el clasificador cuántico variational.
            
            Args:
                X: Matriz de características de entrenamiento.
                y: Vector de etiquetas de entrenamiento.
            """
            # Normalizar datos para el modelo cuántico (rango -1 a 1)
            X_normalizado = self._normalizar_datos_cuantico(X)
            
            # Configurar instancia cuántica (simulador)
            quantum_instance = QuantumInstance(backend=None)  # None = simulador
            
            # Crear y entrenar el modelo VQC
            self.modelo = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                quantum_instance=quantum_instance
            )
            
            print("Entrenando modelo cuántico (simulación)...")
            self.modelo.fit(X_normalizado, y)
            print("Entrenamiento cuántico completado.")

        def predecir(self, X: np.array) -> np.array:
            """
            Realiza predicciones con el modelo cuántico entrenado.
            
            Args:
                X: Datos de entrada para predecir.
                
            Returns:
                Vector de predicciones (0 o 1).
            """
            if self.modelo is None:
                raise ValueError("Modelo no entrenado. Llama a entrenar() primero.")
                
            X_normalizado = self._normalizar_datos_cuantico(X)
            return self.modelo.predict(X_normalizado)

        def _normalizar_datos_cuantico(self, X: np.array) -> np.array:
            """
            Normaliza datos para el modelo cuántico (escala -1 a 1).
            
            Args:
                X: Datos a normalizar.
                
            Returns:
                Datos normalizados.
            """
            # Para el modelo cuántico, normalizamos a un rango de -1 a 1
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            return 2 * (X - X_min) / (X_max - X_min) - 1

        def guardar_modelo(self, archivo: str = 'perceptron_cuantico.pkl'):
            """
            Guarda la configuración del modelo cuántico.
            
            Nota: Los parámetros entrenados del modelo cuántico no se guardan
            ya que la serialización de modelos VQC es compleja. En una aplicación
            real se necesitaría una estrategia más sofisticada.
            
            Args:
                archivo: Ruta del archivo donde guardar la configuración.
            """
            # En una implementación real, necesitaríamos una estrategia más
            # sofisticada para guardar el estado del modelo cuántico
            print("Advertencia: El guardado de modelos cuánticos es limitado.")
            print("Solo se guarda la configuración, no los parámetros entrenados.")
            
            with open(archivo, 'wb') as f:
                pickle.dump({
                    'feature_map': self.feature_map,
                    'ansatz': self.ansatz,
                    'optimizer': self.optimizer
                }, f)
            print(f"Configuración cuántica guardada en {archivo}")

        def cargar_modelo(self, archivo: str = 'perceptron_cuantico.pkl') -> bool:
            """
            Carga la configuración del modelo cuántico.
            
            Args:
                archivo: Ruta del archivo desde donde cargar la configuración.
                
            Returns:
                True si se cargó exitosamente, False en caso contrario.
            """
            try:
                with open(archivo, 'rb') as f:
                    datos = pickle.load(f)
                self.feature_map = datos['feature_map']
                self.ansatz = datos['ansatz']
                self.optimizer = datos['optimizer']
                print(f"Configuración cuántica cargada desde {archivo}")
                return True
            except FileNotFoundError:
                print("No se encontró un modelo cuántico previo.")
                return False
            except KeyError as e:
                print(f"Error al cargar el modelo cuántico: {e}")
                return False


def es_respuesta_afirmativa(respuesta: str) -> bool:
    """
    Determina si una respuesta es afirmativa.
    
    Args:
        respuesta: Texto de la respuesta.
        
    Returns:
        True si la respuesta es afirmativa, False en caso contrario.
    """
    return respuesta.lower() in ['sí', 'si', 's', 'yes', 'y', '1', 'true', 'verdadero', 'afirmativo']


def es_respuesta_negativa(respuesta: str) -> bool:
    """
    Determina si una respuesta es negativa.
    
    Args:
        respuesta: Texto de la respuesta.
        
    Returns:
        True si la respuesta es negativa, False en caso contrario.
    """
    return respuesta.lower() in ['no', 'n', '0', 'false', 'falso', 'negativo']


def mostrar_menu_principal():
    """Muestra el menú principal de selección de algoritmo."""
    print("\n" + "="*50)
    print("PERCEPTRÓN PARA PREDICCIÓN DE PAGOS DE PRÉSTAMOS")
    print("="*50)
    print("¿Qué algoritmo deseas usar?")
    print("1. Perceptrón Clásico")
    if QUANTUM_AVAILABLE:
        print("2. Perceptrón Cuántico (Simulación)")
    else:
        print("2. Perceptrón Cuántico (No disponible - Instala qiskit)")
    print("3. Salir")
    

def main():
    """Función principal del programa."""
    # Datos de entrenamiento iniciales
    X_entrenamiento = np.array([
        [1000, 3000], [5000, 4000], [300, 2500], [2000, 3500],
        [1500, 2000], [4000, 5000], [800, 1800], [3000, 4000],
        [1200, 2200], [6000, 3000], [700, 1500], [2500, 3000]
    ])
    y_entrenamiento = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    
    # Bucle principal
    while True:
        mostrar_menu_principal()
        opcion = input("Selecciona una opción (1-3): ")
        
        if opcion == "3":
            print("¡Hasta luego!")
            break
            
        elif opcion == "1":
            # Modo clásico
            print("\n--- MODO CLÁSICO ---")
            perceptron = PerceptronSimple(tasa_aprendizaje=0.01, n_iteraciones=100)
            modelo_cargado = perceptron.cargar_modelo()
            
            if not modelo_cargado:
                print("Entrenando modelo clásico...")
                perceptron.entrenar(X_entrenamiento, y_entrenamiento)
                perceptron.guardar_modelo()
                
            # Bucle de interacción
            ejecutar_modo_clasico(perceptron)
            
        elif opcion == "2" and QUANTUM_AVAILABLE:
            # Modo cuántico
            print("\n--- MODO CUÁNTICO (SIMULACIÓN) ---")
            perceptron = PerceptronCuantico()
            modelo_cargado = perceptron.cargar_modelo()
            
            if not modelo_cargado:
                print("Entrenando modelo cuántico...")
                perceptron.entrenar(X_entrenamiento, y_entrenamiento)
                perceptron.guardar_modelo()
                
            # Bucle de interacción
            ejecutar_modo_cuantico(perceptron)
            
        else:
            print("Opción no válida. Por favor, selecciona 1, 2 o 3.")


def ejecutar_modo_clasico(perceptron: PerceptronSimple):
    """
    Ejecuta el bucle de interacción para el modo clásico.
    
    Args:
        perceptron: Instancia del perceptrón clásico.
    """
    while True:
        print("\n--- Predicción de Pago (Clásico) ---")
        try:
            monto = float(input("Monto del préstamo: "))
            ingreso = float(input("Ingreso mensual del usuario: "))
        except ValueError:
            print("Por favor, ingresa números válidos.")
            continue

        # Normalizar y predecir
        X_nuevo = np.array([monto, ingreso])
        X_nuevo_norm = perceptron.normalizar_dato(X_nuevo)
        prediccion = perceptron.predecir(X_nuevo_norm)
        
        resultado = "PAGARÁ" if prediccion == 1 else "NO PAGARÁ"
        print(f"Predicción: {resultado}")

        # Feedback y aprendizaje online
        corroboracion = input("¿La predicción fue correcta? (sí/no): ")
        
        if es_respuesta_negativa(corroboracion):
            try:
                verdadera_etiqueta = int(input("Ingresa la etiqueta correcta (1 para PAGÓ, 0 para NO PAGÓ): "))
                
                # Ajustar pesos
                prediccion_actual = perceptron.predecir(X_nuevo_norm)
                update = perceptron.tasa_aprendizaje * (verdadera_etiqueta - prediccion_actual)
                perceptron.pesos += update * X_nuevo_norm
                perceptron.sesgo += update
                
                print("Modelo actualizado con tu feedback.")
                perceptron.guardar_modelo()
            except ValueError:
                print("Etiqueta no válida. No se actualizó el modelo.")

        # Preguntar si desea continuar
        continuar = input("¿Predecir otro? (sí/no): ")
        if es_respuesta_negativa(continuar):
            break


def ejecutar_modo_cuantico(perceptron):
    """
    Ejecuta el bucle de interacción para el modo cuántico.
    
    Args:
        perceptron: Instancia del perceptrón cuántico.
    """
    while True:
        print("\n--- Predicción de Pago (Cuántico) ---")
        try:
            monto = float(input("Monto del préstamo: "))
            ingreso = float(input("Ingreso mensual del usuario: "))
        except ValueError:
            print("Por favor, ingresa números válidos.")
            continue

        # Predecir (la normalización se hace internamente)
        X_nuevo = np.array([[monto, ingreso]])
        prediccion = perceptron.predecir(X_nuevo)[0]
        
        resultado = "PAGARÁ" if prediccion == 1 else "NO PAGARÁ"
        print(f"Predicción: {resultado}")

        # Nota: El aprendizaje online no está implementado para el modo cuántico
        # ya que requeriría reentrenar el modelo completo, lo cual es costoso
        
        # Preguntar si desea continuar
        continuar = input("¿Predecir otro? (sí/no): ")
        if es_respuesta_negativa(continuar):
            break


if __name__ == "__main__":
    # Ejecutar el programa principal
    main()