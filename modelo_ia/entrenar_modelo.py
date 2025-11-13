# modelo_ia/datos_entrenamiento/entrenar_modelo.py
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential, Model  # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
    from tensorflow.keras.applications import MobileNetV2  # type: ignore
    TENSORFLOW_DISPONIBLE = True
except ImportError:
    TENSORFLOW_DISPONIBLE = False
    print("TensorFlow no disponible - ejecutando en modo simulación")


class EntrenadorModelo:
    """
    Clase para entrenar modelos de reconocimiento de señas
    """
    
    def __init__(self, datos_dir: str = None, modelo_dir: str = None):
        self.datos_dir = Path(datos_dir) if datos_dir else Path(__file__).parent
        self.modelo_dir = Path(modelo_dir) if modelo_dir else Path(__file__).parent.parent / "modelos"
        
        # Crear directorio de modelos si no existe
        self.modelo_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración del modelo
        self.input_shape = (224, 224, 3)
        self.batch_size = 32
        self.learning_rate = 0.001
        
        print(f"Directorio de datos: {self.datos_dir}")
        print(f"Directorio de modelos: {self.modelo_dir}")
    
    def obtener_estadisticas_dataset(self) -> Dict[str, Any]:
        """Obtener estadísticas del dataset disponible"""
        estadisticas = {
            "total_imagenes": 0,
            "clases": {},
            "clases_disponibles": [],
            "estado": "vacio"
        }
        
        if not self.datos_dir.exists():
            return estadisticas
        
        for categoria_dir in self.datos_dir.iterdir():
            if categoria_dir.is_dir():
                nombre_clase = categoria_dir.name
                imagenes = list(categoria_dir.glob("*.jpg")) + \
                          list(categoria_dir.glob("*.jpeg")) + \
                          list(categoria_dir.glob("*.png"))
                
                cantidad = len(imagenes)
                if cantidad > 0:
                    estadisticas["clases"][nombre_clase] = cantidad
                    estadisticas["clases_disponibles"].append(nombre_clase)
                    estadisticas["total_imagenes"] += cantidad
        
        # Determinar estado
        if estadisticas["total_imagenes"] == 0:
            estadisticas["estado"] = "vacio"
        elif len(estadisticas["clases"]) < 2:
            estadisticas["estado"] = "insuficiente"
        elif estadisticas["total_imagenes"] < 50:
            estadisticas["estado"] = "minimo"
        else:
            estadisticas["estado"] = "listo"
        
        return estadisticas
    
    def validar_dataset(self) -> Dict[str, Any]:
        """Validar que el dataset esté listo para entrenamiento"""
        estadisticas = self.obtener_estadisticas_dataset()
        
        validacion = {
            "valido": False,
            "errores": [],
            "advertencias": [],
            "recomendaciones": []
        }
        
        # Verificar número mínimo de clases
        if len(estadisticas.get("clases", {})) < 2:
            validacion["errores"].append("Se necesitan al menos 2 clases diferentes para entrenar")
        
        # Verificar número mínimo de imágenes por clase
        for clase, cantidad in estadisticas.get("clases", {}).items():
            if cantidad < 10:
                validacion["errores"].append(f"La clase '{clase}' tiene solo {cantidad} imágenes (mínimo 10)")
            elif cantidad < 20:
                validacion["advertencias"].append(f"La clase '{clase}' tiene pocas imágenes ({cantidad}). Recomendado: 20+")
        
        # Verificar total de imágenes
        total = estadisticas.get("total_imagenes", 0)
        if total < 50:
            validacion["errores"].append(f"Dataset muy pequeño: {total} imágenes (mínimo recomendado: 50)")
        elif total < 100:
            validacion["advertencias"].append(f"Dataset pequeño: {total} imágenes (recomendado: 100+)")
        
        validacion["valido"] = len(validacion["errores"]) == 0
        
        return validacion
    
    def crear_data_generators(self) -> Tuple[Any, Any]:
        """Crear generadores de datos con augmentación"""
        if not TENSORFLOW_DISPONIBLE:
            raise Exception("TensorFlow no disponible")
        
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            str(self.datos_dir),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            str(self.datos_dir),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        return train_generator, validation_generator
    
    def crear_modelo(self, num_clases: int, usar_transfer_learning: bool = True):
        """Crear modelo CNN"""
        if not TENSORFLOW_DISPONIBLE:
            raise Exception("TensorFlow no disponible")
        
        if usar_transfer_learning:
            # Transfer learning con MobileNetV2
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            modelo = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(num_clases, activation='softmax')
            ])
        else:
            # CNN desde cero
            modelo = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(num_clases, activation='softmax')
            ])
        
        modelo.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return modelo
    
    def entrenar(
        self,
        nombre_modelo: str = None,
        usar_transfer_learning: bool = True,
        epochs: int = 50
    ) -> Dict[str, Any]:
        """Entrenar el modelo"""
        
        if not TENSORFLOW_DISPONIBLE:
            return self._simular_entrenamiento(nombre_modelo)
        
        # Validar dataset
        validacion = self.validar_dataset()
        if not validacion["valido"]:
            return {
                "exito": False,
                "error": "Dataset no válido",
                "detalles": validacion
            }
        
        # Obtener estadísticas
        stats = self.obtener_estadisticas_dataset()
        num_clases = len(stats["clases"])
        clases = stats["clases_disponibles"]
        
        print(f"Iniciando entrenamiento con {num_clases} clases: {clases}")
        
        # Crear generadores
        train_gen, val_gen = self.crear_data_generators()
        
        # Crear modelo
        modelo = self.crear_modelo(num_clases, usar_transfer_learning)
        
        # Configurar callbacks
        if not nombre_modelo:
            nombre_modelo = f"modelo_senas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        modelo_path = self.modelo_dir / f"{nombre_modelo}.h5"
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
            ModelCheckpoint(str(modelo_path), monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # Entrenar
        print("Iniciando entrenamiento...")
        history = modelo.fit(
            train_gen,
            validation_data=val_gen,
            epochs=min(epochs, 100),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        evaluacion = modelo.evaluate(val_gen, verbose=0)
        accuracy = evaluacion[1]
        loss = evaluacion[0]
        
        # Guardar clases
        clases_path = self.modelo_dir / "clases_senas.json"
        with open(clases_path, 'w', encoding='utf-8') as f:
            json.dump(clases, f, ensure_ascii=False, indent=2)
        
        resultado = {
            "exito": True,
            "nombre_modelo": nombre_modelo,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "num_clases": num_clases,
            "clases": clases,
            "epochs_entrenadas": len(history.history['accuracy']),
            "ruta_modelo": str(modelo_path)
        }
        
        print(f"Entrenamiento completado - Accuracy: {accuracy:.4f}")
        return resultado
    
    def _simular_entrenamiento(self, nombre_modelo: str = None) -> Dict[str, Any]:
        """Simulación para desarrollo sin TensorFlow"""
        import random
        import time
        
        stats = self.obtener_estadisticas_dataset()
        
        if not nombre_modelo:
            nombre_modelo = f"modelo_simulado_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("Simulando entrenamiento...")
        time.sleep(2)
        
        return {
            "exito": True,
            "modo": "simulacion",
            "nombre_modelo": nombre_modelo,
            "accuracy": random.uniform(0.85, 0.95),
            "loss": random.uniform(0.05, 0.15),
            "num_clases": len(stats.get("clases", {})),
            "clases": stats.get("clases_disponibles", []),
            "epochs_entrenadas": random.randint(15, 30),
            "advertencia": "Entrenamiento simulado - TensorFlow no disponible"
        }


if __name__ == "__main__":
    # Ejemplo de uso
    entrenador = EntrenadorModelo()
    
    print("\n=== Estadísticas del Dataset ===")
    stats = entrenador.obtener_estadisticas_dataset()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n=== Validación del Dataset ===")
    validacion = entrenador.validar_dataset()
    print(json.dumps(validacion, indent=2, ensure_ascii=False))
    
    if validacion["valido"]:
        print("\n=== Iniciando Entrenamiento ===")
        resultado = entrenador.entrenar(
            nombre_modelo="modelo_signafree",
            usar_transfer_learning=True,
            epochs=50
        )
        print(json.dumps(resultado, indent=2, ensure_ascii=False))