"""
Perceptrón Simple para Predicción de Pagos de Préstamos
Este script implementa un perceptrón simple que predice si un usuario pagará o no un préstamo
basándose en el monto del préstamo y el ingreso mensual. El modelo utiliza aprendizaje en línea,
guardando el conocimiento adquirido después de cada interacción, y puede ser extendido a una
implementación cuántica.
Autor: [Tu Nombre]
Fecha: [Fecha]
Licencia: MIT
"""
import numpy as np
import pickle
import os
class PerceptronSimple:
    """
    Una implementación del perceptrón simple con funciones de entrenamiento, predicción y persistencia.
    
    El perceptrón es un algoritmo de clasificación lineal que utiliza una función de activación escalón.
    Es capaz de aprender iterativamente a partir de datos etiquetados y guardar el modelo aprendido.
    
    Atributos:
        tasa_aprendizaje (float): Tasa de aprendizaje para el ajuste de pesos.
        n_iteraciones (int): Número de iteraciones de entrenamiento.
        pesos (numpy.ndarray): Vector de pesos del modelo.
        sesgo (float): Término de sesgo (bias) del modelo.
        errores (list): Lista de errores en cada iteración de entrenamiento.
        media (numpy.ndarray): Media de los datos de entrenamiento para normalización.
        desviacion (numpy.ndarray): Desviación estándar de los datos de entrenamiento para normalización.
    """
    
    def __init__(self, tasa_aprendizaje=0.01, n_iteraciones=100):
        """
        Inicializa el perceptrón con hiperparámetros.
        
        Args:
            tasa_aprendizaje (float): La tasa de aprendizaje para el algoritmo (default 0.01).
            n_iteraciones (int): Número de iteraciones de entrenamiento (default 100).
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.n_iteraciones = n_iteraciones
        self.pesos = None
        self.sesgo = None
        self.errores = []
        self.media = None
        self.desviacion = None
    def funcion_activacion(self, x):
        """
        Función de activación escalón (Heaviside).
        
        Esta función representa una compuerta lógica que activa la neurona cuando el valor
        de la suma ponderada supera el umbral.
        
        Args:
            x (float): Valor de la suma ponderada (producto punto de entradas y pesos más sesgo).
            
        Returns:
            int: 1 si x >= 0, 0 en caso contrario.
        """
        return 1 if x >= 0 else 0
    def predecir(self, x):
        """
        Realiza una predicción para una instancia de entrada.
        
        Args:
            x (numpy.ndarray): Vector de características de entrada normalizadas.
            
        Returns:
            int: La predicción de la clase (0 o 1).
        """
        suma_ponderada = np.dot(x, self.pesos) + self.sesgo
        return self.funcion_activacion(suma_ponderada)
    def entrenar(self, X, y):
        """
        Entrena el perceptrón con datos etiquetados.
        
        El proceso de entrenamiento ajusta iterativamente los pesos y el sesgo para minimizar
        los errores de clasificación. También calcula y almacena parámetros de normalización.
        
        Args:
            X (numpy.ndarray): Matriz de características de entrenamiento.
            y (numpy.ndarray): Vector de etiquetas de entrenamiento (0 o 1).
        """
        n_muestras, n_caracteristicas = X.shape
        
        # Inicializar pesos y sesgo si no existen
        if self.pesos is None:
            self.pesos = np.zeros(n_caracteristicas)
        if self.sesgo is None:
            self.sesgo = 0
        # Calcular y guardar media y desviación para normalización futura
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        
        # Normalizar datos
        X_norm = (X - self.media) / self.desviacion
        # Entrenamiento iterativo
        for _ in range(self.n_iteraciones):
            error = 0
            for idx, x_i in enumerate(X_norm):
                # Calcular predicción
                prediccion = self.predecir(x_i)
                
                # Calcular ajuste
                update = self.tasa_aprendizaje * (y[idx] - prediccion)
                
                # Actualizar pesos y sesgo
                self.pesos += update * x_i
                self.sesgo += update
                
                # Acumular error
                error += int(update != 0.0)
            
            # Registrar error de la iteración
            self.errores.append(error)
    def normalizar_dato(self, x):
        """
        Normaliza un dato usando los parámetros almacenados del conjunto de entrenamiento.
        
        La normalización es crucial para asegurar la convergencia del perceptrón y mantener
        la consistencia entre los datos de entrenamiento y predicción.
        
        Args:
            x (numpy.ndarray): Dato a normalizar.
            
        Returns:
            numpy.ndarray: Dato normalizado.
        """
        # Si no hay parámetros de normalización, devolver el dato sin normalizar
        if self.media is None or self.desviacion is None:
            return x
        return (x - self.media) / self.desviacion
    def guardar_modelo(self, archivo='perceptron_modelo.pkl'):
        """
        Guarda el modelo entrenado en un archivo usando pickle.
        
        Persiste los pesos, sesgo y parámetros de normalización para uso futuro
        sin necesidad de reentrenamiento.
        
        Args:
            archivo (str): Ruta del archivo donde guardar el modelo.
        """
        with open(archivo, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos, 
                'sesgo': self.sesgo,
                'media': self.media,
                'desviacion': self.desviacion
            }, f)
        print(f"Modelo guardado en {archivo}")
    def cargar_modelo(self, archivo='perceptron_modelo.pkl'):
        """
        Carga un modelo previamente entrenado desde un archivo.
        
        Args:
            archivo (str): Ruta del archivo desde donde cargar el modelo.
            
        Returns:
            bool: True si se cargó exitosamente, False en caso contrario.
        """
        try:
            with open(archivo, 'rb') as f:
                datos = pickle.load(f)
            
            # Cargar parámetros con manejo de errores para retrocompatibilidad
            self.pesos = datos['pesos']
            self.sesgo = datos['sesgo']
            self.media = datos.get('media', None)  # Usar get para evitar KeyError
            self.desviacion = datos.get('desviacion', None)  # Usar get para evitar KeyError
            
            print(f"Modelo cargado desde {archivo}")
            return True
        
        except FileNotFoundError:
            print("No se encontró un modelo previo. Iniciando desde cero.")
            return False
        
        except KeyError as e:
            print(f"Error al cargar el modelo: {e}. Iniciando desde cero.")
            return False
def es_respuesta_afirmativa(respuesta):
    """
    Determina si una respuesta del usuario es afirmativa.
    
    Args:
        respuesta (str): Respuesta del usuario.
        
    Returns:
        bool: True si la respuesta es afirmativa, False en caso contrario.
    """
    return respuesta.lower() in ['sí', 'si', 's', 'yes', 'y', '1', 'true', 'verdadero']
def es_respuesta_negativa(respuesta):
    """
    Determina si una respuesta del usuario es negativa.
    
    Args:
        respuesta (str): Respuesta del usuario.
        
    Returns:
        bool: True si la respuesta es negativa, False en caso contrario.
    """
    return respuesta.lower() in ['no', 'n', '0', 'false', 'falso']
# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    """
    Punto de entrada principal del programa.
    
    Esta sección maneja la interacción con el usuario, el ciclo de predicción
    y el aprendizaje continuo del modelo.
    """
    
    # Inicializar el perceptrón
    perceptron = PerceptronSimple(tasa_aprendizaje=0.01, n_iteraciones=100)
    
    # Intentar cargar un modelo existente
    modelo_existente = perceptron.cargar_modelo()
    # Datos de entrenamiento iniciales (ejemplo)
    # Estos datos sirven como base inicial para el modelo
    X_entrenamiento = np.array([
        [1000, 3000],  # Monto bajo, ingreso medio -> Paga
        [5000, 4000],  # Monto alto, ingreso medio -> No paga
        [300, 2500],   # Monto muy bajo, ingreso bajo -> Paga
        [2000, 3500]   # Monto medio, ingreso medio -> Paga
    ])
    y_entrenamiento = np.array([1, 0, 1, 1])  # Etiquetas correspondientes
    # Si no hay modelo existente, entrenar con datos iniciales
    if not modelo_existente:
        print("Entrenando modelo con datos iniciales...")
        perceptron.entrenar(X_entrenamiento, y_entrenamiento)
        perceptron.guardar_modelo()
    # Bucle de interacción con el usuario
    print("\n=== Sistema de Predicción de Pagos ===")
    print("Ingresa los datos del cliente para predecir si pagará el préstamo.")
    print("Después de cada predicción, puedes corregir al modelo para mejorar su aprendizaje.\n")
    
    while True:
        print("\n--- Nueva Predicción ---")
        
        # Obtener datos del usuario con validación
        try:
            monto = float(input("Monto del préstamo solicitado: "))
            ingreso = float(input("Ingreso mensual del cliente: "))
        except ValueError:
            print("Error: Por favor ingresa valores numéricos válidos.")
            continue
        # Normalizar el nuevo dato usando los parámetros del modelo
        X_nuevo = np.array([monto, ingreso])
        X_nuevo_norm = perceptron.normalizar_dato(X_nuevo)
        # Realizar predicción
        prediccion = perceptron.predecir(X_nuevo_norm)
        resultado = "PAGARÁ" if prediccion == 1 else "NO PAGARÁ"
        print(f"\nPredicción: El cliente {resultado} el préstamo")
        # Obtener feedback del usuario
        corroboracion = input("\n¿La predicción fue correcta? (sí/no): ")
        
        if es_respuesta_negativa(corroboracion):
            try:
                verdadera_etiqueta = int(input("Ingresa la etiqueta correcta (1 para PAGÓ, 0 para NO PAGÓ): "))
                
                if verdadera_etiqueta not in [0, 1]:
                    print("Error: La etiqueta debe ser 0 o 1. No se actualizó el modelo.")
                    continue
                
                # Ajustar pesos en tiempo real (aprendizaje online)
                prediccion_actual = perceptron.predecir(X_nuevo_norm)
                update = perceptron.tasa_aprendizaje * (verdadera_etiqueta - prediccion_actual)
                perceptron.pesos += update * X_nuevo_norm
                perceptron.sesgo += update
                
                print("✓ Modelo actualizado con tu feedback")
                perceptron.guardar_modelo()
                
            except ValueError:
                print("Error: Etiqueta no válida. Debe ser 0 o 1. No se actualizó el modelo.")
        # Preguntar si desea continuar
        continuar = input("\n¿Deseas predecir otro cliente? (sí/no): ")
        if es_respuesta_negativa(continuar):
            print("\n¡Gracias por usar el Sistema de Predicción de Pagos!")
            print("Hasta pronto 👋")
            break