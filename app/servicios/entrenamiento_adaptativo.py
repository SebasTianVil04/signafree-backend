import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# CORRECCIÓN: Desactivar cuDNN benchmark para evitar errores de memoria
os.environ['CUDNN_DETERMINISTIC'] = '1'

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedShuffleSplit
import gc
import time
import cv2
import random
from collections import Counter
from threading import Lock

from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.modelos.dataset import VideoDataset
from app.utilidades.base_datos import SessionLocal
from app.servicios.config_tipo_senas import (
    detectar_tipo_sena, 
    obtener_config_sena, 
    tipo_sena_config,
    validar_consistencia_categoria,
    obtener_parametros_entrenamiento,
    es_categoria_alfabeto
)

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

progresos_entrenamiento: Dict[str, Any] = {}
_lock_progresos = Lock()

def normalizar_imagenet(frame: np.ndarray) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    frame_float = np.clip(frame_float, 0.0, 1.0)
    frame_normalized = (frame_float - MEAN) / STD
    return frame_normalized

class VideoFrameDatasetAdaptativo(Dataset):
    
    def __init__(self, video_paths: List[str], labels: List[int], 
                 senas: List[str], num_frames: int = 20, is_training: bool = True):
        self.video_paths = video_paths
        self.labels = labels
        self.senas = senas
        self.num_frames = num_frames
        self.is_training = is_training
        
        self.tipos_senas = [detectar_tipo_sena(sena) for sena in senas]
        self.configs_senas = [obtener_config_sena(sena) for sena in senas]
        
        if len(video_paths) != len(labels) or len(video_paths) != len(senas):
            raise ValueError(f"Mismatch: {len(video_paths)} paths vs {len(labels)} labels vs {len(senas)} senas")
        
        estaticas = sum(1 for t in self.tipos_senas if t == 'ESTATICA')
        dinamicas = sum(1 for t in self.tipos_senas if t == 'DINAMICA')
        
        logger.info(f"Dataset Adaptativo: {len(video_paths)} videos")
        logger.info(f"  - Estaticas: {estaticas} ({estaticas/len(video_paths)*100:.1f}%)")
        logger.info(f"  - Dinamicas: {dinamicas} ({dinamicas/len(video_paths)*100:.1f}%)")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        sena = self.senas[idx]
        tipo_sena = self.tipos_senas[idx]
        config = self.configs_senas[idx]
        
        try:
            num_frames_usar = config['num_frames_recomendado'] if tipo_sena == 'ESTATICA' else self.num_frames
            
            frames = self._load_video_frames(video_path, num_frames_usar, tipo_sena, config)
            
            if frames.shape[0] != self.num_frames:
                if frames.shape[0] < self.num_frames:
                    frames = np.pad(frames, 
                                  ((0, self.num_frames - frames.shape[0]), (0,0), (0,0), (0,0)),
                                  mode='edge')
                else:
                    frames = frames[:self.num_frames]
            
            if self.is_training:
                if tipo_sena == 'DINAMICA':
                    frames = self._aplicar_augmentation_dinamica(frames, config)
                else:
                    frames = self._aplicar_augmentation_estatica(frames, config)
            
            frames_normalized = normalizar_imagenet(frames)
            
            frames_tensor = torch.from_numpy(
                frames_normalized.transpose(0, 3, 1, 2).copy()
            ).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            expected_shape = (self.num_frames, 3, 224, 224)
            if frames_tensor.shape != expected_shape:
                raise ValueError(f"Shape invalido: {frames_tensor.shape}")
            
            return frames_tensor, label_tensor, tipo_sena
            
        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {e}")
            dummy_frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            dummy_normalized = normalizar_imagenet(dummy_frames)
            dummy_tensor = torch.from_numpy(
                dummy_normalized.transpose(0, 3, 1, 2).copy()
            ).float()
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_tensor, dummy_label, 'DINAMICA'

    def _load_video_frames(self, video_path: str, num_frames: int, 
                        tipo_sena: str, config: Dict) -> np.ndarray:
        cap = None
        try:
            if not os.path.exists(video_path):
                return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
            
            if tipo_sena == 'ESTATICA':
                frame_central = total_frames // 2
                ventana = num_frames // 2
                inicio = max(0, frame_central - ventana)
                fin = min(total_frames, frame_central + ventana)
                frame_indices = np.linspace(inicio, fin - 1, num_frames, dtype=int)
            else:
                if total_frames <= num_frames:
                    frame_indices = np.arange(total_frames)
                else:
                    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frame_indices = np.unique(frame_indices)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    scale = 224 / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    if new_w > 224:
                        new_w = 224
                    if new_h > 224:
                        new_h = 224
                    
                    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
                    y_offset = (224 - new_h) // 2
                    x_offset = (224 - new_w) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
                    
                    frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    if len(frames) > 0:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            while len(frames) < num_frames:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames_array = np.array(frames[:num_frames], dtype=np.uint8)
            
            return frames_array
            
        except Exception as e:
            logger.error(f"Error cargando frames: {e}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
        finally:
            if cap is not None:
                cap.release()

    def _aplicar_augmentation_estatica(self, frames: np.ndarray, config: Dict) -> np.ndarray:
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() > 0.4:
            brightness_range = config.get('augmentation_brightness', (0.7, 1.3))
            brightness = random.uniform(*brightness_range)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        if random.random() > 0.4:
            max_angle = config.get('augmentation_rotation', 15)
            angle = random.uniform(-max_angle, max_angle)
            h, w = frames.shape[1:3]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frames = np.array([cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT) 
                              for frame in frames])
        
        if random.random() > 0.6:
            alpha = random.uniform(0.8, 1.2)
            frames = np.clip(alpha * frames, 0, 255).astype(np.uint8)
        
        return frames

    def _aplicar_augmentation_dinamica(self, frames: np.ndarray, config: Dict) -> np.ndarray:
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        if random.random() > 0.3:
            brightness_range = config.get('augmentation_brightness', (0.6, 1.4))
            brightness = random.uniform(*brightness_range)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        if random.random() > 0.4:
            max_angle = config.get('augmentation_rotation', 10)
            angle = random.uniform(-max_angle, max_angle)
            h, w = frames.shape[1:3]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frames = np.array([cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT) 
                              for frame in frames])
        
        if random.random() > 0.5:
            noise_level = random.uniform(5, 10)
            noise = np.random.normal(0, noise_level, frames.shape).astype(np.int16)
            frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        if random.random() > 0.7 and len(frames) > 5:
            num_drop = random.randint(1, 2)
            drop_indices = random.sample(range(1, len(frames) - 1), num_drop)
            for idx in drop_indices:
                frames[idx] = (frames[idx - 1] + frames[idx + 1]) // 2
        
        return frames

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class EntrenamientoAdaptativoService:
    
    def __init__(self):
        self.pytorch_disponible = torch is not None
        self.modelo_dir = Path('modelos_entrenados')
        self.modelo_dir.mkdir(exist_ok=True)
        
        self.device, self.gpu_info = self.detectar_y_configurar_gpu()
        
        if torch.cuda.is_available():
            memoria_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # CORRECCIÓN: Reducir batch size para evitar errores de memoria en CuDNN
            if memoria_gpu >= 8:
                self.batch_size = 8  # Reducido de 12
                self.num_frames = 20  # Reducido de 24
            elif memoria_gpu >= 6:
                self.batch_size = 6  # Reducido de 8
                self.num_frames = 20  # Reducido de 20
            elif memoria_gpu >= 4:
                self.batch_size = 4  # Reducido de 6
                self.num_frames = 16
            else:
                self.batch_size = 2  # Reducido de 4
                self.num_frames = 16
            
            import platform
            self.num_workers = 0 if platform.system() == 'Windows' else 2  # Reducido de 4
        else:
            self.batch_size = 2
            self.num_frames = 16
            self.num_workers = 0
        
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.min_videos_por_sena = 8
        self.min_accuracy = 0.50
        
        logger.info(f"Servicio Adaptativo inicializado en {self.device}")
        logger.info(f"Configuracion: batch_size={self.batch_size}, frames={self.num_frames}")

    def detectar_y_configurar_gpu(self) -> Tuple[torch.device, Optional[Dict[str, Any]]]:
        logger.info("Detectando GPU...")
        
        if not torch.cuda.is_available():
            logger.info("GPU no disponible, usando CPU")
            return torch.device('cpu'), None
        
        gpu_idx = 0
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f'cuda:{gpu_idx}')
        
        nombre = torch.cuda.get_device_name(gpu_idx)
        props = torch.cuda.get_device_properties(gpu_idx)
        memoria_total = props.total_memory / (1024**3)
        
        logger.info(f"GPU detectada: {nombre} - {memoria_total:.2f} GB")
        
        torch.cuda.empty_cache()
        # CORRECCIÓN: Desactivar benchmark para operaciones determinísticas
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        
        return device, {"nombre": nombre, "memoria_total_gb": round(memoria_total, 2)}

    def generar_nombre_modelo(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"modelo_adaptativo_{timestamp}"

    def validar_dataset_entrenamiento(self, categoria_ids: List[int], db: Session) -> Dict[str, Any]:
        try:
            videos_db = db.query(VideoDataset).filter(
                VideoDataset.aprobado == True,
                VideoDataset.categoria_id.in_(categoria_ids),
                VideoDataset.ruta_video.isnot(None)
            ).all()
            
            videos_validos = []
            videos_no_encontrados = []
            
            for video in videos_db:
                if os.path.exists(video.ruta_video):
                    videos_validos.append(video)
                else:
                    videos_no_encontrados.append(video.ruta_video)
            
            videos_por_sena = {}
            tipos_por_sena = {}
            for video in videos_validos:
                sena = video.sena or "sin_clasificar"
                if sena not in videos_por_sena:
                    videos_por_sena[sena] = []
                    tipos_por_sena[sena] = detectar_tipo_sena(sena)
                videos_por_sena[sena].append(video)
            
            total_videos = len(videos_validos)
            total_senas = len(videos_por_sena)
            
            senas_con_minimo = []
            senas_sin_minimo = []
            senas_estaticas = []
            senas_dinamicas = []
            
            for sena, videos in videos_por_sena.items():
                tipo = tipos_por_sena[sena]
                
                if tipo == 'ESTATICA':
                    senas_estaticas.append(sena)
                else:
                    senas_dinamicas.append(sena)
                
                if len(videos) >= self.min_videos_por_sena:
                    senas_con_minimo.append(sena)
                else:
                    senas_sin_minimo.append(sena)
            
            consistencia = validar_consistencia_categoria(list(videos_por_sena.keys()))
            es_alfabeto = es_categoria_alfabeto(list(videos_por_sena.keys()))
            
            valido = total_videos >= 20 and len(senas_con_minimo) >= 2
            
            resultado = {
                "valido": valido,
                "total_videos": total_videos,
                "videos_validos": len(videos_validos),
                "videos_no_encontrados": len(videos_no_encontrados),
                "total_senas": total_senas,
                "senas_con_minimo": senas_con_minimo,
                "senas_sin_minimo": senas_sin_minimo,
                "senas_estaticas": senas_estaticas,
                "senas_dinamicas": senas_dinamicas,
                "distribucion": {
                    "estaticas": len(senas_estaticas),
                    "dinamicas": len(senas_dinamicas),
                    "tipo_predominante": consistencia['tipo_mayoritario']
                },
                "consistencia": consistencia,
                "es_categoria_alfabeto": es_alfabeto,
                "minimo_videos_requerido": 20,
                "minimo_senas_requerido": 2,
                "minimo_videos_por_sena": self.min_videos_por_sena,
                "cumple_requisitos": {
                    "videos_totales": total_videos >= 20,
                    "senas_suficientes": len(senas_con_minimo) >= 2,
                    "videos_por_sena": len(senas_sin_minimo) == 0
                }
            }
            
            logger.info(f"Validacion dataset: {total_videos} videos, {total_senas} senas")
            logger.info(f"  - Estaticas: {len(senas_estaticas)}, Dinamicas: {len(senas_dinamicas)}")
            logger.info(f"  - Tipo predominante: {consistencia['tipo_mayoritario']}")
            if es_alfabeto:
                logger.info(f"  - Categoria de ALFABETO detectada")
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error validando dataset: {str(e)}")
            return {
                "valido": False,
                "error": str(e),
                "total_videos": 0,
                "videos_validos": 0,
                "total_senas": 0,
                "senas_con_minimo": [],
                "senas_sin_minimo": []
            }

    def obtener_progreso_entrenamiento(self, nombre_modelo: str) -> Dict[str, Any]:
        global progresos_entrenamiento
        
        with _lock_progresos:
            if nombre_modelo in progresos_entrenamiento:
                progress = progresos_entrenamiento[nombre_modelo]
                resultado = {
                    "nombre_modelo": progress.get("nombre_modelo", nombre_modelo),
                    "estado": progress.get("estado", "desconocido"),
                    "progreso": progress.get("progreso", 0.0),
                    "epoch_actual": progress.get("epoch_actual", 0),
                    "total_epochs": progress.get("total_epochs", 0),
                    "accuracy": progress.get("accuracy", 0.0),
                    "loss": progress.get("loss", 0.0),
                    "train_loss": progress.get("train_loss", 0.0),
                    "train_accuracy": progress.get("train_accuracy", 0.0),
                    "num_clases": progress.get("num_clases", 0),
                    "clases": progress.get("clases", []),
                    "total_videos": progress.get("total_videos", 0),
                    "frames_procesados": progress.get("frames_procesados", 0),
                    "total_frames": progress.get("total_frames", 0),
                    "mensaje": progress.get("mensaje", ""),
                    "fecha_inicio": progress.get("fecha_inicio", datetime.now().isoformat()),
                    "entrenando": progress.get("entrenando", False)
                }
                logger.debug(f"Progreso en memoria: {nombre_modelo}, estado={resultado['estado']}")
                return resultado
        
        db = SessionLocal()
        try:
            from app.modelos.entrenamiento import ModeloIA
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            
            if modelo_db:
                clases = []
                if modelo_db.clases_json:
                    try:
                        clases = json.loads(modelo_db.clases_json)
                    except:
                        clases = []
                
                resultado = {
                    "nombre_modelo": modelo_db.nombre,
                    "estado": "completado" if modelo_db.activo else "inactivo",
                    "progreso": 100.0 if modelo_db.activo else 0.0,
                    "epoch_actual": modelo_db.epocas_entrenamiento or 0,
                    "total_epochs": modelo_db.epocas_entrenamiento or 0,
                    "accuracy": float(modelo_db.accuracy) if modelo_db.accuracy else 0.0,
                    "loss": 0.0,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "num_clases": modelo_db.num_clases or 0,
                    "clases": clases,
                    "total_videos": modelo_db.total_imagenes or 0,
                    "frames_procesados": (modelo_db.total_imagenes or 0) * self.num_frames,
                    "total_frames": (modelo_db.total_imagenes or 0) * self.num_frames,
                    "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db.fecha_entrenamiento else None,
                    "mensaje": "Modelo completado previamente" if modelo_db.activo else "Modelo inactivo",
                    "entrenando": False
                }
                logger.info(f"Modelo encontrado en BD: {nombre_modelo}")
                return resultado
            
            else:
                logger.warning(f"Modelo no encontrado: {nombre_modelo}, retornando estado 'preparando'")
                resultado = {
                    "nombre_modelo": nombre_modelo,
                    "estado": "preparando",
                    "progreso": 0.0,
                    "epoch_actual": 0,
                    "total_epochs": 0,
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "num_clases": 0,
                    "clases": [],
                    "total_videos": 0,
                    "frames_procesados": 0,
                    "total_frames": 0,
                    "mensaje": "Inicializando entrenamiento...",
                    "entrenando": True
                }
                return resultado
        
        except Exception as e:
            logger.error(f"Error consultando modelo: {str(e)}", exc_info=True)
            return {
                "nombre_modelo": nombre_modelo,
                "estado": "error",
                "progreso": 0.0,
                "accuracy": 0.0,
                "loss": 0.0,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "num_clases": 0,
                "clases": [],
                "total_videos": 0,
                "frames_procesados": 0,
                "total_frames": 0,
                "mensaje": f"Error consultando: {str(e)}",
                "entrenando": False
            }
        
        finally:
            db.close()

    def entrenar(self, nombre_modelo: str, categoria_ids: List[int], epochs: int):
        global progresos_entrenamiento
        
        db = None
        model = None
        train_loader = None
        val_loader = None
        
        try:
            if not epochs or epochs <= 0:
                epochs = 150
            elif epochs > 500:
                epochs = 500
            
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo] = {
                    "nombre_modelo": nombre_modelo,
                    "estado": "iniciando",
                    "progreso": 0.0,
                    "epoch_actual": 0,
                    "total_epochs": epochs,
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "num_clases": 0,
                    "clases": [],
                    "total_videos": 0,
                    "frames_procesados": 0,
                    "total_frames": 0,
                    "mensaje": "Inicializando entrenamiento...",
                    "fecha_inicio": datetime.now().isoformat(),
                    "entrenando": True
                }
            
            logger.info(f"Entrenamiento iniciado: {nombre_modelo}, epochs={epochs}")
            
            time.sleep(0.5)
            
            # CORRECCIÓN: Limpiar memoria GPU antes de comenzar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            db = SessionLocal()
            
            videos_db = db.query(VideoDataset).filter(
                VideoDataset.aprobado == True,
                VideoDataset.categoria_id.in_(categoria_ids),
                VideoDataset.ruta_video.isnot(None)
            ).all()
            
            videos_validos = [v for v in videos_db if os.path.exists(v.ruta_video)]
            
            if len(videos_validos) < 20:
                raise Exception(f"Insuficientes videos: {len(videos_validos)} (minimo 20)")
            
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo]["estado"] = "preparando_datos"
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Preparando dataset..."
                progresos_entrenamiento[nombre_modelo]["total_videos"] = len(videos_validos)
            
            videos_por_sena = {}
            for video in videos_validos:
                sena = video.sena or "sin_clasificar"
                if sena not in videos_por_sena:
                    videos_por_sena[sena] = []
                videos_por_sena[sena].append(video)
            
            senas_validas = {k: v for k, v in videos_por_sena.items() 
                        if len(v) >= self.min_videos_por_sena}
            
            if len(senas_validas) < 2:
                raise Exception(
                    f"Se necesitan al menos 2 senas con {self.min_videos_por_sena}+ videos. "
                    f"Senas validas: {list(senas_validas.keys())}"
                )
            
            video_paths = []
            labels = []
            senas_list = []
            clases = sorted(senas_validas.keys())
            clase_a_idx = {clase: i for i, clase in enumerate(clases)}
            
            tipos_senas_dict = {}
            for clase in clases:
                tipos_senas_dict[clase] = detectar_tipo_sena(clase)
            
            for clase, videos_clase in senas_validas.items():
                for video in videos_clase:
                    video_paths.append(video.ruta_video)
                    labels.append(clase_a_idx[clase])
                    senas_list.append(clase)
            
            total_videos = len(video_paths)
            total_frames = total_videos * self.num_frames
            
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo].update({
                    "num_clases": len(clases),
                    "clases": clases,
                    "total_videos": total_videos,
                    "total_frames": total_frames,
                    "tipos_senas": tipos_senas_dict,
                    "estado": "cargando_datos",
                    "mensaje": f"Cargando {total_videos} videos de {len(clases)} senas..."
                })
            
            logger.info(f"Dataset preparado: {len(clases)} senas, {total_videos} videos")
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(video_paths, labels))
            
            train_paths = [video_paths[i] for i in train_idx]
            val_paths = [video_paths[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            train_senas = [senas_list[i] for i in train_idx]
            val_senas = [senas_list[i] for i in val_idx]
            
            class_counts = Counter(train_labels)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            train_dataset = VideoFrameDatasetAdaptativo(
                train_paths, train_labels, train_senas,
                self.num_frames, is_training=True
            )
            val_dataset = VideoFrameDatasetAdaptativo(
                val_paths, val_labels, val_senas,
                self.num_frames, is_training=False
            )
            
            dataloader_config = {
                'batch_size': self.batch_size,
                'sampler': sampler,
                'num_workers': self.num_workers,
                'drop_last': True
            }
            
            if self.num_workers > 0:
                dataloader_config['pin_memory'] = True if self.device.type == 'cuda' else False
            
            train_loader = DataLoader(train_dataset, **dataloader_config)
            
            val_dataloader_config = {
                'batch_size': self.batch_size,
                'shuffle': False,
                'num_workers': self.num_workers,
                'drop_last': False
            }
            
            if self.num_workers > 0:
                val_dataloader_config['pin_memory'] = True if self.device.type == 'cuda' else False
            
            val_loader = DataLoader(val_dataset, **val_dataloader_config)
            
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo]["estado"] = "creando_modelo"
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Inicializando arquitectura..."
            
            model = ModeloAdaptativoSenas(len(clases), tipos_senas_dict).to(self.device)
            
            criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            
            warmup_epochs = min(15, epochs // 10)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs/epochs,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
            
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo]["estado"] = "entrenando"
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Entrenamiento en progreso..."
            
            logger.info(f"Iniciando loop de entrenamiento: {epochs} epocas")
            
            best_accuracy = 0.0
            best_model_state = None
            patience = 30
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (videos, labels_batch, tipos_batch) in enumerate(train_loader):
                    try:
                        videos = videos.to(self.device, non_blocking=True)
                        labels_batch = labels_batch.to(self.device, non_blocking=True)
                        
                        optimizer.zero_grad(set_to_none=True)
                        
                        if scaler:
                            with torch.cuda.amp.autocast():
                                outputs = model.forward_batch_mixto(videos, tipos_batch)
                                loss = criterion(outputs, labels_batch)
                            
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model.forward_batch_mixto(videos, tipos_batch)
                            loss = criterion(outputs, labels_batch)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        
                        scheduler.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        train_total += labels_batch.size(0)
                        train_correct += (predicted == labels_batch).sum().item()
                        
                        frames_procesados = (epoch * len(train_loader) + batch_idx + 1) * self.batch_size * self.num_frames
                        
                        with _lock_progresos:
                            progresos_entrenamiento[nombre_modelo]["frames_procesados"] = min(frames_procesados, total_frames)
                        
                        del videos, labels_batch, outputs, loss, predicted
                        
                        # CORRECCIÓN: Limpiar caché con más frecuencia
                        if (batch_idx + 1) % 2 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e) or "CuDNN" in str(e):
                            logger.error(f"Error de memoria/CuDNN en batch {batch_idx}: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise
                
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for videos, labels_batch, tipos_batch in val_loader:
                        try:
                            videos = videos.to(self.device, non_blocking=True)
                            labels_batch = labels_batch.to(self.device, non_blocking=True)
                            
                            if scaler:
                                with torch.cuda.amp.autocast():
                                    outputs = model.forward_batch_mixto(videos, tipos_batch)
                                    loss = criterion(outputs, labels_batch)
                            else:
                                outputs = model.forward_batch_mixto(videos, tipos_batch)
                                loss = criterion(outputs, labels_batch)
                            
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += labels_batch.size(0)
                            val_correct += (predicted == labels_batch).sum().item()
                            
                            del videos, labels_batch, outputs, loss, predicted
                        
                        except RuntimeError as e:
                            if "out of memory" in str(e) or "CuDNN" in str(e):
                                logger.error(f"Error de memoria/CuDNN en validacion: {e}")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            else:
                                raise
                
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                
                with _lock_progresos:
                    progresos_entrenamiento[nombre_modelo].update({
                        "epoch_actual": epoch + 1,
                        "total_epochs": epochs,
                        "accuracy": float(val_acc),
                        "loss": float(avg_val_loss),
                        "train_accuracy": float(train_acc),
                        "train_loss": float(avg_train_loss),
                        "progreso": ((epoch + 1) / epochs) * 100.0,
                        "mensaje": f"Epoca {epoch+1}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}"
                    })
                
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(
                        f"Epoca {epoch+1:3d}/{epochs} | "
                        f"Train: Loss {avg_train_loss:.4f}, Acc {train_acc:.4f} | "
                        f"Val: Loss {avg_val_loss:.4f}, Acc {val_acc:.4f} | "
                        f"Best: {best_accuracy:.4f}"
                    )
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping activado en epoca {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if best_accuracy >= self.min_accuracy:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                ruta_modelo = self._guardar_modelo(model, nombre_modelo, clases, best_accuracy, tipos_senas_dict)
                
                self._actualizar_bd(db, nombre_modelo, str(ruta_modelo), clases, 
                                best_accuracy, epochs, len(video_paths), tipos_senas_dict)
                
                with _lock_progresos:
                    progresos_entrenamiento[nombre_modelo].update({
                        "estado": "completado",
                        "entrenando": False,
                        "mensaje": "Entrenamiento completado exitosamente",
                        "frames_procesados": total_frames,
                        "progreso": 100.0,
                        "accuracy": float(best_accuracy)
                    })
                
                logger.info(f"Entrenamiento completado: {nombre_modelo}, accuracy={best_accuracy:.4f}")
            else:
                with _lock_progresos:
                    progresos_entrenamiento[nombre_modelo].update({
                        "estado": "error",
                        "entrenando": False,
                        "mensaje": f"Accuracy insuficiente: {best_accuracy*100:.2f}%"
                    })
                raise Exception(f"Accuracy insuficiente: {best_accuracy*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
            with _lock_progresos:
                if nombre_modelo in progresos_entrenamiento:
                    progresos_entrenamiento[nombre_modelo].update({
                        "estado": "error",
                        "entrenando": False,
                        "mensaje": f"Error: {str(e)}"
                    })
            raise
        
        finally:
            if train_loader:
                del train_loader
            if val_loader:
                del val_loader
            if model:
                del model
            if db:
                try:
                    db.close()
                except:
                    pass
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _guardar_modelo(self, model, nombre_modelo: str, clases: List[str], 
                       accuracy: float, tipos_senas: dict) -> Path:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            ruta_modelo = self.modelo_dir / f"{nombre_modelo}.pth"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'clases': clases,
                'num_clases': len(clases),
                'accuracy': float(accuracy),
                'num_frames': self.num_frames,
                'arquitectura': 'ModeloAdaptativoSenas',
                'framework': 'pytorch_adaptativo',
                'tipos_senas': tipos_senas,
                'fecha_entrenamiento': datetime.now().isoformat(),
                'normalizacion': 'imagenet',
                'mean': MEAN.tolist(),
                'std': STD.tolist()
            }, str(ruta_modelo))
            
            logger.info(f"Modelo guardado: {ruta_modelo}, accuracy={accuracy:.4f}")
            
            return ruta_modelo
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise

    def _actualizar_bd(self, db: Session, nombre_modelo: str, ruta_modelo: str, 
                      clases: List[str], accuracy: float, epochs: int, total_videos: int,
                      tipos_senas: dict):
        try:
            from app.modelos.entrenamiento import ModeloIA
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            
            if not modelo_db:
                modelo_db = ModeloIA(nombre=nombre_modelo)
                db.add(modelo_db)
            
            estaticas = sum(1 for t in tipos_senas.values() if t == 'ESTATICA')
            dinamicas = sum(1 for t in tipos_senas.values() if t == 'DINAMICA')
            
            # CORRECCIÓN: Asegurar que accuracy se guarde correctamente
            accuracy_decimal = round(float(accuracy), 6)
            
            modelo_db.ruta_archivo = ruta_modelo
            modelo_db.accuracy = accuracy_decimal
            modelo_db.num_clases = len(clases)
            modelo_db.clases_json = json.dumps(clases, ensure_ascii=False)
            modelo_db.total_imagenes = total_videos
            modelo_db.epocas_entrenamiento = epochs
            modelo_db.arquitectura = "ModeloAdaptativoSenas"
            modelo_db.entrenando = False
            modelo_db.activo = True
            modelo_db.fecha_entrenamiento = datetime.now()
            modelo_db.version = "3.0"
            modelo_db.descripcion = (
                f"Modelo adaptativo con procesamiento diferenciado. "
                f"{len(clases)} senas ({estaticas} estaticas, {dinamicas} dinamicas), "
                f"{total_videos} videos, accuracy: {accuracy_decimal*100:.2f}%"
            )
            
            db.commit()
            db.refresh(modelo_db)
            
            logger.info(f"Modelo registrado en BD: {nombre_modelo}, accuracy={accuracy_decimal:.4f}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error actualizando BD: {str(e)}", exc_info=True)
            raise

entrenamiento_adaptativo_service = EntrenamientoAdaptativoService()