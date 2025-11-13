from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import base64
from datetime import datetime
import logging
from pydantic import BaseModel
import time
import asyncio

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA

# IMPORTAR EL RECONOCEDOR CORRECTO
from ..servicios.reconocimiento_adaptativo import ReconocimientoAdaptativoIA
from ..servicios.config_tipo_senas import detectar_tipo_sena

router = APIRouter(prefix="/reconocimiento-video", tags=["Reconocimiento Video"])

logger = logging.getLogger(__name__)

# Cache global optimizado
_reconocedor_video_cache = {
    'reconocedor': None,
    'modelo_id': None,
    'ultimo_uso': None
}

class FrameData(BaseModel):
    frame_base64: str
    timestamp: int
    landmarks: Optional[List[Any]] = None

class SolicitudVideoSecuencia(BaseModel):
    frames: List[FrameData]
    configuracion: Dict[str, Any] = {}
    sena_esperada: Optional[str] = None

def obtener_reconocedor_video(db: Session) -> ReconocimientoAdaptativoIA:
    """
    Obtiene el reconocedor adaptativo para video - OPTIMIZADO
    """
    try:
        modelo_activo = db.query(ModeloIA).filter(ModeloIA.activo == True).first()
        
        if not modelo_activo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay modelo activo para reconocimiento de video"
            )
        
        # Reutilizar si ya está cargado (más agresivo)
        if (_reconocedor_video_cache['reconocedor'] is not None and 
            _reconocedor_video_cache['modelo_id'] == modelo_activo.id):
            
            _reconocedor_video_cache['ultimo_uso'] = datetime.now()
            return _reconocedor_video_cache['reconocedor']
        
        # Cargar nuevo reconocedor
        logger.info(f"📂 Cargando reconocedor: {modelo_activo.nombre}")
        
        reconocedor = ReconocimientoAdaptativoIA(ruta_modelo=modelo_activo.ruta_archivo)
        
        _reconocedor_video_cache['reconocedor'] = reconocedor
        _reconocedor_video_cache['modelo_id'] = modelo_activo.id
        _reconocedor_video_cache['ultimo_uso'] = datetime.now()
        
        logger.info(f"✓ Reconocedor cargado: {len(reconocedor.clases)} clases, {reconocedor.accuracy*100:.1f}%")
        
        return reconocedor
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error obteniendo reconocedor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {str(e)}"
        )

def procesar_frame_base64_optimizado(frame_base64: str) -> np.ndarray:
    """Decodificación optimizada de frame"""
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
            
        # Redimensionar inmediatamente para mejor performance
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        return frame
        
    except Exception as e:
        logger.warning(f"⚠ Error procesando frame: {e}")
        return None

@router.post("/secuencia", response_model=Dict[str, Any])
async def reconocer_secuencia_video(
    solicitud: SolicitudVideoSecuencia,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Reconoce una secuencia de frames - VERSIÓN OPTIMIZADA PARA TIEMPO REAL
    """
    start_time = time.time()
    
    try:
        num_frames_recibidos = len(solicitud.frames)
        
        # VALIDACIÓN CRÍTICA: No procesar si hay demasiados frames
        if num_frames_recibidos > 16:
            # Tomar solo los últimos 16 frames (más relevantes)
            frames_a_procesar = solicitud.frames[-16:]
            logger.info(f"📹 Optimizando: {num_frames_recibidos} → 16 frames")
        else:
            frames_a_procesar = solicitud.frames
            
        # Cargar reconocedor
        reconocedor = obtener_reconocedor_video(db)
        
        # Procesar frames en lote (más eficiente)
        frames_procesados = []
        
        for i, frame_data in enumerate(frames_a_procesar):
            frame = procesar_frame_base64_optimizado(frame_data.frame_base64)
            if frame is not None:
                frames_procesados.append(frame)
        
        frames_exitosos = len(frames_procesados)
        
        # Validación más estricta para tiempo real
        if frames_exitosos < 6:  # Reducido de 8 a 6
            return {
                "exito": False,
                "mensaje": f"Frames insuficientes para reconocimiento ({frames_exitosos}/6)",
                "datos": {
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": frames_exitosos
                }
            }
        
        # DETERMINAR TIPO DE SEÑA MÁS INTELIGENTEMENTE
        tipo_sena = 'DINAMICA'
        if solicitud.sena_esperada:
            tipo_sena = detectar_tipo_sena(solicitud.sena_esperada)
        
        # USAR MÉTODO MÁS RÁPIDO según el tipo de seña
        if tipo_sena == 'ESTATICA' and frames_exitosos >= 8:
            # Para señas estáticas, usar menos frames pero más representativos
            centro = len(frames_procesados) // 2
            frames_estaticos = frames_procesados[max(0, centro-3):centro+3]
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(
                frames_estaticos, 
                tipo_sena
            )
        else:
            # Para señas dinámicas, usar todos los frames disponibles
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(
                frames_procesados, 
                tipo_sena
            )
        
        # VALIDACIÓN DE CONFIANZA MÁS ESTRICTA
        confianza_validada = max(0.0, min(1.0, confianza))
        
        # Calcular métricas de performance
        processing_time = time.time() - start_time
        
        # CONFIGURACIÓN DE CALIDAD MÁS ESTRICTA
        if confianza_validada >= 0.85:
            calidad = "excelente"
            mensaje = "Alta confianza"
        elif confianza_validada >= 0.75:
            calidad = "buena" 
            mensaje = "Buena confianza"
        elif confianza_validada >= 0.65:
            calidad = "moderada"
            mensaje = "Confianza moderada"
        elif confianza_validada >= 0.55:
            calidad = "baja"
            mensaje = "Baja confianza - necesita mejorar"
        else:
            calidad = "muy_baja"
            mensaje = "Seña no reconocida claramente"
        
        # Filtrar alternativas con confianza mínima
        alternativas_filtradas = []
        if "alternativas" in detalles:
            for alt in detalles["alternativas"]:
                if alt.get("confianza", 0) >= 0.3:  # Solo mostrar alternativas con >30%
                    alternativas_filtradas.append(alt)
        
        # Construir respuesta optimizada
        resultado = {
            "sena_detectada": sena_detectada,
            "texto_traducido": sena_detectada,
            "confianza": round(confianza_validada, 4),
            "confianza_raw": round(confianza_validada, 4),  # Siempre incluir
            "porcentaje": round(confianza_validada * 100, 2),
            "calidad": calidad,
            "mensaje": mensaje,
            "modo": "video_secuencia",
            "num_frames_procesados": frames_exitosos,
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "fps": round(frames_exitosos / processing_time, 2) if processing_time > 0 else 0,
            "alternativas": alternativas_filtradas[:3],  # Máximo 3 alternativas
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✓ Reconocimiento: {sena_detectada} ({confianza_validada*100:.1f}%) - {processing_time*1000:.1f}ms")
        
        return {
            "exito": True,
            "mensaje": "Reconocimiento completado",
            "datos": resultado
        }
        
    except Exception as e:
        logger.error(f"✗ Error en reconocimiento: {str(e)}")
        return {
            "exito": False,
            "mensaje": "Error en el reconocimiento",
            "datos": {
                "confianza": 0.0,
                "sena_detectada": "",
                "num_frames_procesados": 0
            },
            "errores": [f"Error interno: {str(e)}"]
        }

# ENDPOINT NUEVO: Reconocimiento rápido para tiempo real
@router.post("/rapido", response_model=Dict[str, Any])
async def reconocimiento_rapido(
    solicitud: SolicitudVideoSecuencia,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Reconocimiento ultra-rápido para tiempo real - MÁXIMO 8 FRAMES
    """
    start_time = time.time()
    
    try:
        # LIMITAR A 8 FRAMES MÁXIMO para velocidad
        frames_a_procesar = solicitud.frames[-8:]  # Solo últimos 8 frames
        
        reconocedor = obtener_reconocedor_video(db)
        
        # Procesamiento ultra-rápido
        frames_procesados = []
        for frame_data in frames_a_procesar:
            frame = procesar_frame_base64_optimizado(frame_data.frame_base64)
            if frame is not None:
                frames_procesados.append(frame)
        
        if len(frames_procesados) < 4:  # Mínimo muy bajo para velocidad
            return {
                "exito": False,
                "mensaje": "Frames insuficientes",
                "datos": {"confianza": 0.0, "sena_detectada": ""}
            }
        
        # PREDICCIÓN RÁPIDA
        sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(
            frames_procesados, 
            'DINAMICA'  # Asumir dinámica para mayor velocidad
        )
        
        processing_time = time.time() - start_time
        
        # Respuesta mínima para velocidad
        return {
            "exito": True,
            "mensaje": "Reconocimiento rápido completado",
            "datos": {
                "sena_detectada": sena_detectada,
                "confianza": round(max(0, confianza), 4),
                "confianza_raw": round(max(0, confianza), 4),
                "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
                "num_frames_procesados": len(frames_procesados)
            }
        }
        
    except Exception as e:
        logger.error(f"✗ Error en reconocimiento rápido: {str(e)}")
        return {
            "exito": False,
            "mensaje": "Error en reconocimiento rápido",
            "datos": {"confianza": 0.0, "sena_detectada": ""}
        }

@router.post("/video-completo", response_model=Dict[str, Any])
async def reconocer_video_completo(
    data: Dict[str, Any],
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Reconoce un video completo en base64
    """
    try:
        video_base64 = data.get("video_base64")
        configuracion = data.get("configuracion", {})
        sena_esperada = data.get("sena_esperada")
        
        if not video_base64:
            return {
                "exito": False,
                "mensaje": "Se requiere video en base64",
                "errores": ["Falta parámetro video_base64"]
            }
        
        logger.info("📹 Procesando video completo en base64")
        
        # Guardar video temporalmente
        import tempfile
        import os
        
        try:
            if ',' in video_base64:
                video_base64 = video_base64.split(',')[1]
            
            video_bytes = base64.b64decode(video_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(video_bytes)
                temp_video_path = temp_file.name
            
            # Cargar reconocedor
            reconocedor = obtener_reconocedor_video(db)
            
            # Reconocer desde video
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_video(
                temp_video_path,
                sena_esperada=sena_esperada
            )
            
            # Construir respuesta
            resultado = {
                "sena_detectada": sena_detectada,
                "texto_traducido": sena_detectada,
                "confianza": round(confianza, 4),
                "porcentaje": round(confianza * 100, 2),
                "alternativas": detalles.get("alternativas", []),
                "detalles_modelo": {
                    "arquitectura": detalles.get("modelo", "ModeloAdaptativoSenas"),
                    "tipo_procesamiento": detalles.get("tipo_sena", "DINAMICA"),
                    "tipo_real": detalles.get("tipo_real", "DINAMICA"),
                    "num_frames_usado": detalles.get("num_frames_usado", reconocedor.num_frames)
                }
            }
            
            logger.info(f"✓ Video reconocido: {sena_detectada} ({confianza*100:.1f}%)")
            
            return {
                "exito": True,
                "mensaje": "Reconocimiento completado",
                "datos": resultado
            }
            
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
        
    except Exception as e:
        logger.error(f"✗ Error en video completo: {str(e)}")
        return {
            "exito": False,
            "mensaje": "Error procesando video completo",
            "errores": [f"Error: {str(e)}"]
        }

@router.get("/info-modelo")
async def obtener_info_modelo(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Obtiene información del modelo cargado
    """
    try:
        reconocedor = obtener_reconocedor_video(db)
        
        modelo_db = db.query(ModeloIA).filter(
            ModeloIA.id == _reconocedor_video_cache['modelo_id']
        ).first()
        
        info = {
            "exito": True,
            "modelo": {
                "id": modelo_db.id if modelo_db else None,
                "nombre": modelo_db.nombre if modelo_db else "desconocido",
                "arquitectura": reconocedor.arquitectura,
                "num_clases": reconocedor.num_clases,
                "clases": reconocedor.clases,
                "num_frames": reconocedor.num_frames,
                "accuracy_entrenamiento": round(reconocedor.accuracy * 100, 2),
                "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db and modelo_db.fecha_entrenamiento else None,
                "tipos_senas": reconocedor.tipos_senas
            },
            "sistema": {
                "device": str(reconocedor.device),
                "cuda_disponible": str(reconocedor.device).startswith('cuda'),
                "amp_enabled": reconocedor.enable_amp
            }
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error obteniendo info: {e}")
        return {
            "exito": False,
            "mensaje": f"Error: {str(e)}"
        }

@router.get("/estado")
async def obtener_estado_reconocimiento(
    db: Session = Depends(obtener_bd)
):
    """
    Obtiene el estado del sistema de reconocimiento - CORREGIDO
    """
    try:
        reconocedor = _reconocedor_video_cache.get('reconocedor')
        
        if reconocedor is None:
            # Intentar cargar el reconocedor
            try:
                reconocedor = obtener_reconocedor_video(db)
            except Exception as e:
                logger.warning(f"No se pudo cargar reconocedor: {e}")
        
        estado = {
            "estado": "activo" if reconocedor is not None else "inactivo",
            "modelo_cargado": reconocedor is not None,
            "arquitectura": reconocedor.arquitectura if reconocedor else "ninguna",
            "clases_cargadas": len(reconocedor.clases) if reconocedor else 0,
            "clases": reconocedor.clases if reconocedor else [],
            "num_frames": reconocedor.num_frames if reconocedor else 0,
            "device": str(reconocedor.device) if reconocedor else "ninguno",
            "ultimo_uso": _reconocedor_video_cache['ultimo_uso'].isoformat() if _reconocedor_video_cache['ultimo_uso'] else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return estado
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return {
            "estado": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/limpiar-cache")
async def limpiar_cache_modelo():
    """
    Limpia el cache del modelo
    """
    try:
        _reconocedor_video_cache['reconocedor'] = None
        _reconocedor_video_cache['modelo_id'] = None
        _reconocedor_video_cache['ultimo_uso'] = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✓ Cache del modelo limpiado exitosamente")
        
        return {
            "exito": True,
            "mensaje": "Cache limpiado exitosamente"
        }
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return {
            "exito": False,
            "mensaje": f"Error: {str(e)}"
        }

@router.get("/test-conexion")
async def test_conexion():
    """
    Endpoint simple para test de conexión
    """
    return {
        "exito": True,
        "mensaje": "Servicio de reconocimiento activo",
        "timestamp": datetime.now().isoformat()
    }