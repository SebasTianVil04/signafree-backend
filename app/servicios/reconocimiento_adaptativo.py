import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import cv2
import time
import logging
from typing import Tuple, Dict, Optional, List
import torch.nn.functional as F

from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.servicios.config_tipo_senas import detectar_tipo_sena, obtener_config_sena

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalizar_imagenet(frame: np.ndarray) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    frame_float = np.clip(frame_float, 0.0, 1.0)
    frame_normalized = (frame_float - MEAN) / STD
    return frame_normalized

class ReconocimientoAdaptativoIA:
    def __init__(self, ruta_modelo: str = None):
        self.model = None
        self.clases = []
        self.num_clases = 0
        self.num_frames = 16
        self.accuracy = 0.0
        self.arquitectura = ""
        self.tipos_senas = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo_dir = Path('modelos_entrenados')
        
        # CONFIGURACIÓN CORREGIDA PARA CUDA/CUDNN
        self.enable_amp = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            # Configuración más conservadora para evitar errores
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False  # Deshabilitar benchmark temporalmente
            torch.backends.cudnn.deterministic = True
            
            # Limpiar cache de CUDA
            torch.cuda.empty_cache()
            
            # Verificar que CUDA funcione
            try:
                test_tensor = torch.randn(1, 3, 224, 224).cuda()
                del test_tensor
                logger.info("✓ CUDA inicializada correctamente")
            except Exception as e:
                logger.warning(f"⚠ Problema con CUDA: {e}")
                self.device = torch.device('cpu')
                self.enable_amp = False
        
        if ruta_modelo is None:
            pth_files = list(self.modelo_dir.glob('*.pth'))
            if not pth_files:
                raise ValueError("No se encontraron modelos .pth")
            ruta_modelo = max(pth_files, key=os.path.getctime)
            logger.info(f"Usando modelo más reciente: {ruta_modelo}")
        
        self.cargar_modelo(ruta_modelo)
    
    def cargar_modelo(self, ruta_modelo: str):
        try:
            if not Path(ruta_modelo).exists():
                raise FileNotFoundError(f"Modelo no encontrado: {ruta_modelo}")
            
            checkpoint = torch.load(
                ruta_modelo, 
                map_location=self.device,
                weights_only=False
            )
            
            self.num_clases = checkpoint['num_clases']
            self.clases = checkpoint['clases']
            self.num_frames = checkpoint.get('num_frames', 16)
            self.accuracy = checkpoint.get('accuracy', 0.0)
            self.arquitectura = checkpoint.get('arquitectura', 'ModeloAdaptativoSenas')
            self.tipos_senas = checkpoint.get('tipos_senas', {})
            
            if not self.tipos_senas:
                self.tipos_senas = {clase: detectar_tipo_sena(clase) for clase in self.clases}
            
            self.model = ModeloAdaptativoSenas(self.num_clases, self.tipos_senas).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Modelo adaptativo cargado: {ruta_modelo}")
            logger.info(f"Arquitectura: {self.arquitectura}")
            logger.info(f"Clases: {self.clases}")
            logger.info(f"Tipos: {self.tipos_senas}")
            logger.info(f"Frames: {self.num_frames}")
            logger.info(f"Accuracy: {self.accuracy:.4f}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.model = None
            self.clases = []
            raise
    
    def procesar_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[torch.Tensor, str]:
        cap = None
        try:
            if not os.path.exists(ruta_video):
                raise FileNotFoundError(f"Video no encontrado: {ruta_video}")
            
            cap = cv2.VideoCapture(ruta_video)
            if not cap.isOpened():
                raise ValueError(f"No se puede abrir video: {ruta_video}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video sin frames válidos")
            
            tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
            config = obtener_config_sena(sena_esperada) if sena_esperada else None
            
            if tipo_sena == 'ESTATICA' and config:
                num_frames_extraer = config['num_frames_recomendado']
                frame_central = total_frames // 2
                indices_cercanos = [
                    max(0, frame_central - 2),
                    max(0, frame_central - 1),
                    frame_central,
                    min(total_frames - 1, frame_central + 1),
                    min(total_frames - 1, frame_central + 2)
                ]
                frame_indices = np.array(indices_cercanos[:num_frames_extraer])
            else:
                num_frames_extraer = self.num_frames
                frame_indices = np.linspace(0, total_frames - 1, num_frames_extraer, dtype=int)
            
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    if len(frames) > 0:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            while len(frames) < self.num_frames:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames = frames[:self.num_frames]
            
            frames_array = np.stack(frames)
            frames_normalized = normalizar_imagenet(frames_array)
            tensor = torch.from_numpy(frames_normalized.transpose(0, 3, 1, 2)).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor, tipo_sena
            
        except Exception as e:
            logger.error(f"Error procesando video: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()
    
    def procesar_frame_individual(self, frame: np.ndarray, es_estatica: bool = True) -> Tuple[torch.Tensor, str]:
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = normalizar_imagenet(frame)
        frame_array = frame_normalized.transpose(2, 0, 1)
        frame_array = np.expand_dims(frame_array, axis=0)
        
        if es_estatica:
            frames_array = np.tile(frame_array, (5, 1, 1, 1))
            tipo_sena = 'ESTATICA'
        else:
            frames_array = np.tile(frame_array, (self.num_frames, 1, 1, 1))
            tipo_sena = 'DINAMICA'
        
        while frames_array.shape[0] < self.num_frames:
            frames_array = np.concatenate([frames_array, frame_array], axis=0)
        
        frames_array = frames_array[:self.num_frames]
        
        tensor = torch.from_numpy(frames_array).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, tipo_sena
    
    def procesar_frames_secuencia(self, frames_list: List[np.ndarray], tipo_sena: str = 'DINAMICA') -> torch.Tensor:
        frames_procesados = []
        
        for frame in frames_list:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_normalized = normalizar_imagenet(frame)
            frames_procesados.append(frame_normalized)
        
        if tipo_sena == 'ESTATICA':
            if len(frames_procesados) > 5:
                indices = [len(frames_procesados)//2 - 2, 
                          len(frames_procesados)//2 - 1,
                          len(frames_procesados)//2,
                          len(frames_procesados)//2 + 1,
                          len(frames_procesados)//2 + 2]
                frames_procesados = [frames_procesados[max(0, min(i, len(frames_procesados)-1))] for i in indices]
        
        while len(frames_procesados) < self.num_frames:
            if len(frames_procesados) > 0:
                frames_procesados.append(frames_procesados[-1].copy())
            else:
                frames_procesados.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        if len(frames_procesados) > self.num_frames:
            if tipo_sena == 'ESTATICA':
                centro = len(frames_procesados) // 2
                inicio = max(0, centro - self.num_frames//2)
                frames_procesados = frames_procesados[inicio:inicio+self.num_frames]
            else:
                indices = np.linspace(0, len(frames_procesados) - 1, self.num_frames, dtype=int)
                frames_procesados = [frames_procesados[i] for i in indices]
        
        frames_array = np.stack(frames_procesados[:self.num_frames])
        tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2)).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predecir(self, tensor_frames: torch.Tensor, tipo_sena: str = None) -> Tuple[str, float, Dict]:
        if self.model is None or len(self.clases) == 0:
            return "desconocido", 0.0, {}
        
        try:
            if tensor_frames.device != self.device:
                tensor_frames = tensor_frames.to(self.device)
            
            if tipo_sena is None:
                tipo_sena = 'DINAMICA'
            
            with torch.no_grad():
                # INTENTAR PRIMERO CON CUDNN
                try:
                    if self.enable_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(tensor_frames, tipo_sena=tipo_sena)
                    else:
                        outputs = self.model(tensor_frames, tipo_sena=tipo_sena)
                        
                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        logger.warning("⚠ Error cuDNN detectado, reintentando con configuración alternativa...")
                        # Reintentar con cuDNN deshabilitado temporalmente
                        torch.backends.cudnn.enabled = False
                        try:
                            if self.enable_amp:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(tensor_frames, tipo_sena=tipo_sena)
                            else:
                                outputs = self.model(tensor_frames, tipo_sena=tipo_sena)
                        finally:
                            torch.backends.cudnn.enabled = True
                    else:
                        raise e
                
                probabilities = F.softmax(outputs, dim=1)[0]
                
                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    logger.warning("NaN/Inf detectado en probabilidades")
                    probabilities = torch.ones_like(probabilities) / len(self.clases)
                
                probabilities_np = probabilities.cpu().numpy()
            
            indice_predicho = int(np.argmax(probabilities_np))
            confianza = float(probabilities_np[indice_predicho])
            sena_detectada = self.clases[indice_predicho]
            
            top_k = min(5, len(self.clases))
            top_indices = np.argsort(probabilities_np)[-top_k:][::-1]
            
            alternativas = []
            for idx in top_indices:
                alternativas.append({
                    "sena": self.clases[int(idx)],
                    "confianza": float(probabilities_np[int(idx)]),
                    "tipo": self.tipos_senas.get(self.clases[int(idx)], 'DINAMICA')
                })
            
            detalles = {
                "tipo_sena": tipo_sena,
                "tipo_real": self.tipos_senas.get(sena_detectada, 'DINAMICA'),
                "alternativas": alternativas,
                "modelo": self.arquitectura,
                "num_frames_usado": self.num_frames
            }
            
            return sena_detectada, confianza, detalles
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            # Devolver predicción por defecto en lugar de error
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}
    
    def predecir_desde_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tensor, tipo_sena = self.procesar_video(ruta_video, sena_esperada)
        return self.predecir(tensor, tipo_sena)
    
    def predecir_desde_frame(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        es_estatica = (tipo_sena == 'ESTATICA')
        tensor, tipo_sena = self.procesar_frame_individual(frame, es_estatica)
        return self.predecir(tensor, tipo_sena)
    
    def predecir_desde_secuencia(self, frames_list: List[np.ndarray], sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        tensor = self.procesar_frames_secuencia(frames_list, tipo_sena)
        return self.predecir(tensor, tipo_sena)