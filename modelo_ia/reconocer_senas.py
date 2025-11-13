import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Intentar múltiples formas de importar TensorFlow/Keras
TENSORFLOW_DISPONIBLE = False
load_model = None
image = None

try:
    # TensorFlow 2.16+ con Keras 3.0 separado
    import tensorflow as tf
    import keras
    from keras.models import load_model
    from keras.preprocessing import image
    TENSORFLOW_DISPONIBLE = True
    print(f"Keras {keras.__version__} y TensorFlow {tf.__version__} disponibles")
except ImportError:
    try:
        # TensorFlow 2.x con keras integrado
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        TENSORFLOW_DISPONIBLE = True
        print(f"TensorFlow {tf.__version__} disponible (usando tensorflow.keras)")
    except ImportError:
        try:
            # Solo TensorFlow sin Keras
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image
            TENSORFLOW_DISPONIBLE = True
            print(f"TensorFlow {tf.__version__} disponible")
        except ImportError:
            TENSORFLOW_DISPONIBLE = False
            print("TensorFlow no disponible - reconocimiento simulado")

try:
    import mediapipe as mp
    MEDIAPIPE_DISPONIBLE = True
    print("MediaPipe disponible")
except ImportError:
    MEDIAPIPE_DISPONIBLE = False
    print("MediaPipe no disponible - detección de manos desactivada")


class ReconocedorSenas:
    """
    Clase para reconocer señas del lenguaje de señas peruano
    usando modelos entrenados con TensorFlow
    """
    
    def __init__(self, modelo_path: str = None):
        self.modelo_dir = Path(__file__).parent.parent / "modelos"
        self.modelo = None
        self.clases = []
        self.input_shape = (224, 224)
        
        # Inicializar MediaPipe para detección de manos
        self.mp_hands = None
        self.hands = None
        if MEDIAPIPE_DISPONIBLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Cargar modelo si se proporciona
        if modelo_path:
            self.cargar_modelo(modelo_path)
        else:
            self.cargar_modelo_activo()
    
    def cargar_modelo(self, modelo_path: str) -> bool:
        """Cargar un modelo específico"""
        if not TENSORFLOW_DISPONIBLE or load_model is None:
            print("TensorFlow no disponible - modo simulación")
            return False
        
        try:
            modelo_path = Path(modelo_path)
            if not modelo_path.exists():
                print(f"Modelo no encontrado: {modelo_path}")
                return False
            
            # Cargar modelo con compile=False para evitar problemas
            self.modelo = load_model(str(modelo_path), compile=False)
            
            # Recompilar si es necesario
            if hasattr(self.modelo, 'compile'):
                self.modelo.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            print(f"Modelo cargado: {modelo_path}")
            
            # Cargar clases
            clases_path = self.modelo_dir / "clases_senas.json"
            if clases_path.exists():
                with open(clases_path, 'r', encoding='utf-8') as f:
                    self.clases = json.load(f)
                print(f"Clases cargadas: {len(self.clases)}")
            
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cargar_modelo_activo(self) -> bool:
        """Cargar el modelo activo más reciente"""
        if not self.modelo_dir.exists():
            print("Directorio de modelos no existe")
            return False
        
        # Buscar el modelo .h5 más reciente
        modelos = list(self.modelo_dir.glob("*.h5"))
        if not modelos:
            print("No hay modelos disponibles")
            return False
        
        # Ordenar por fecha de modificación (más reciente primero)
        modelo_reciente = max(modelos, key=lambda p: p.stat().st_mtime)
        print(f"Cargando modelo más reciente: {modelo_reciente.name}")
        return self.cargar_modelo(str(modelo_reciente))
    
    def preprocesar_imagen(self, img: np.ndarray) -> np.ndarray:
        """Preprocesar imagen para el modelo"""
        # Redimensionar
        img_resized = cv2.resize(img, self.input_shape)
        
        # Normalizar
        img_array = img_resized.astype('float32') / 255.0
        
        # Expandir dimensiones para batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch
    
    def detectar_mano(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Detectar región de la mano en el frame"""
        if not MEDIAPIPE_DISPONIBLE or self.hands is None:
            return None, None
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Obtener landmarks de la primera mano
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calcular bounding box
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min = int(max(0, min(x_coords) - 20))
            y_min = int(max(0, min(y_coords) - 20))
            x_max = int(min(w, max(x_coords) + 20))
            y_max = int(min(h, max(y_coords) + 20))
            
            # Recortar región de la mano
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            return hand_region, (x_min, y_min, x_max, y_max)
        
        return None, None
    
    def predecir_imagen(self, img: np.ndarray) -> Dict[str, Any]:
        """Predecir la seña en una imagen"""
        if not TENSORFLOW_DISPONIBLE or self.modelo is None:
            return self._simular_prediccion()
        
        try:
            # Preprocesar
            img_prep = self.preprocesar_imagen(img)
            
            # Predecir
            predicciones = self.modelo.predict(img_prep, verbose=0)[0]
            
            # Obtener top 3 predicciones
            top_indices = np.argsort(predicciones)[-3:][::-1]
            
            resultados = []
            for idx in top_indices:
                if idx < len(self.clases):
                    resultados.append({
                        "clase": self.clases[idx],
                        "confianza": float(predicciones[idx]),
                        "porcentaje": float(predicciones[idx] * 100)
                    })
            
            return {
                "exito": True,
                "prediccion_principal": resultados[0] if resultados else None,
                "todas_predicciones": resultados
            }
            
        except Exception as e:
            return {
                "exito": False,
                "error": str(e)
            }
    
    def predecir_desde_ruta(self, ruta_imagen: str) -> Dict[str, Any]:
        """Predecir desde una ruta de imagen"""
        try:
            img = cv2.imread(ruta_imagen)
            if img is None:
                return {"exito": False, "error": "No se pudo cargar la imagen"}
            
            return self.predecir_imagen(img)
            
        except Exception as e:
            return {"exito": False, "error": str(e)}
    
    def predecir_con_webcam(self, duracion_segundos: int = 10):
        """Reconocer señas desde webcam en tiempo real"""
        if not TENSORFLOW_DISPONIBLE or self.modelo is None:
            print("Modelo no disponible")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la cámara")
            return
        
        print(f"Reconocimiento iniciado por {duracion_segundos} segundos...")
        print("Presiona 'q' para salir antes")
        
        import time
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar mano
            hand_region, bbox = self.detectar_mano(frame)
            
            if hand_region is not None and bbox is not None and hand_region.size > 0:
                # Predecir
                resultado = self.predecir_imagen(hand_region)
                
                if resultado["exito"] and resultado["prediccion_principal"]:
                    pred = resultado["prediccion_principal"]
                    
                    # Dibujar bounding box
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Mostrar predicción
                    texto = f"{pred['clase']}: {pred['porcentaje']:.1f}%"
                    cv2.putText(frame, texto, (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Reconocimiento de Señas', frame)
            
            # Verificar salida
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Verificar tiempo
            if time.time() - start_time > duracion_segundos:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Reconocimiento finalizado")
    
    def _simular_prediccion(self) -> Dict[str, Any]:
        """Simulación para desarrollo sin TensorFlow"""
        import random
        
        clases_simuladas = ['A', 'B', 'C', 'Hola', 'Gracias']
        clase = random.choice(clases_simuladas)
        confianza = random.uniform(0.7, 0.95)
        
        return {
            "exito": True,
            "modo": "simulacion",
            "prediccion_principal": {
                "clase": clase,
                "confianza": confianza,
                "porcentaje": confianza * 100
            },
            "advertencia": "Predicción simulada - TensorFlow no disponible"
        }
    
    def obtener_info_modelo(self) -> Dict[str, Any]:
        """Obtener información del modelo cargado"""
        if not TENSORFLOW_DISPONIBLE or self.modelo is None:
            return {
                "cargado": False,
                "mensaje": "No hay modelo cargado"
            }
        
        return {
            "cargado": True,
            "num_clases": len(self.clases),
            "clases": self.clases,
            "input_shape": self.input_shape,
            "arquitectura": str(type(self.modelo).__name__)
        }
    
    def __del__(self):
        """Liberar recursos"""
        if MEDIAPIPE_DISPONIBLE and self.hands:
            self.hands.close()


if __name__ == "__main__":
    # Ejemplo de uso
    reconocedor = ReconocedorSenas()
    
    print("\n=== Información del Modelo ===")
    info = reconocedor.obtener_info_modelo()
    print(json.dumps(info, indent=2, ensure_ascii=False))