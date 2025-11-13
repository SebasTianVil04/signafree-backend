from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import base64
import cv2
import numpy as np
import json
from datetime import datetime
import re
import unicodedata
import logging
import tempfile
import os
import time

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA
from ..esquemas.respuesta_schemas import RespuestaAPI

# IMPORTAR LA CLASE CORRECTA DE RECONOCIMIENTO
from ..servicios.reconocimiento_adaptativo import ReconocimientoAdaptativoIA

router = APIRouter(prefix="/traductor", tags=["Traductor"])

logger = logging.getLogger(__name__)

DICCIONARIO_SENAS = {
    'A': {'categoria': 'alfabeto', 'descripcion': 'Letra A', 'dificultad': 'facil'},
    'B': {'categoria': 'alfabeto', 'descripcion': 'Letra B', 'dificultad': 'facil'},
    'C': {'categoria': 'alfabeto', 'descripcion': 'Letra C', 'dificultad': 'facil'},
    'D': {'categoria': 'alfabeto', 'descripcion': 'Letra D', 'dificultad': 'facil'},
    'E': {'categoria': 'alfabeto', 'descripcion': 'Letra E', 'dificultad': 'facil'},
    'F': {'categoria': 'alfabeto', 'descripcion': 'Letra F', 'dificultad': 'facil'},
    'G': {'categoria': 'alfabeto', 'descripcion': 'Letra G', 'dificultad': 'facil'},
    'H': {'categoria': 'alfabeto', 'descripcion': 'Letra H', 'dificultad': 'facil'},
    'I': {'categoria': 'alfabeto', 'descripcion': 'Letra I', 'dificultad': 'facil'},
    'J': {'categoria': 'alfabeto', 'descripcion': 'Letra J', 'dificultad': 'facil'},
    'K': {'categoria': 'alfabeto', 'descripcion': 'Letra K', 'dificultad': 'facil'},
    'L': {'categoria': 'alfabeto', 'descripcion': 'Letra L', 'dificultad': 'facil'},
    'M': {'categoria': 'alfabeto', 'descripcion': 'Letra M', 'dificultad': 'facil'},
    'N': {'categoria': 'alfabeto', 'descripcion': 'Letra N', 'dificultad': 'facil'},
    'O': {'categoria': 'alfabeto', 'descripcion': 'Letra O', 'dificultad': 'facil'},
    'P': {'categoria': 'alfabeto', 'descripcion': 'Letra P', 'dificultad': 'facil'},
    'Q': {'categoria': 'alfabeto', 'descripcion': 'Letra Q', 'dificultad': 'facil'},
    'R': {'categoria': 'alfabeto', 'descripcion': 'Letra R', 'dificultad': 'facil'},
    'S': {'categoria': 'alfabeto', 'descripcion': 'Letra S', 'dificultad': 'facil'},
    'T': {'categoria': 'alfabeto', 'descripcion': 'Letra T', 'dificultad': 'facil'},
    'U': {'categoria': 'alfabeto', 'descripcion': 'Letra U', 'dificultad': 'facil'},
    'V': {'categoria': 'alfabeto', 'descripcion': 'Letra V', 'dificultad': 'facil'},
    'W': {'categoria': 'alfabeto', 'descripcion': 'Letra W', 'dificultad': 'facil'},
    'X': {'categoria': 'alfabeto', 'descripcion': 'Letra X', 'dificultad': 'facil'},
    'Y': {'categoria': 'alfabeto', 'descripcion': 'Letra Y', 'dificultad': 'facil'},
    'Z': {'categoria': 'alfabeto', 'descripcion': 'Letra Z', 'dificultad': 'facil'},
}

# Cache global para el reconocedor adaptativo
_reconocedor_cache = {
    'reconocedor': None,
    'modelo_id': None,
    'ultimo_uso': None
}

def obtener_reconocedor(bd: Session) -> ReconocimientoAdaptativoIA:
    """
    Obtiene el reconocedor adaptativo, cargándolo si es necesario
    """
    try:
        modelo_activo = bd.query(ModeloIA).filter(ModeloIA.activo == True).first()
        
        if not modelo_activo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No hay modelo activo para reconocimiento"
            )
        
        # Si ya está cargado y es el mismo modelo, reutilizarlo
        if (_reconocedor_cache['reconocedor'] is not None and 
            _reconocedor_cache['modelo_id'] == modelo_activo.id):
            logger.info(f"✓ Reutilizando reconocedor cargado (modelo {modelo_activo.id})")
            _reconocedor_cache['ultimo_uso'] = datetime.now()
            return _reconocedor_cache['reconocedor']
        
        # Cargar nuevo reconocedor
        logger.info(f"📂 Cargando reconocedor adaptativo: {modelo_activo.nombre}")
        
        reconocedor = ReconocimientoAdaptativoIA(ruta_modelo=modelo_activo.ruta_archivo)
        
        _reconocedor_cache['reconocedor'] = reconocedor
        _reconocedor_cache['modelo_id'] = modelo_activo.id
        _reconocedor_cache['ultimo_uso'] = datetime.now()
        
        logger.info(f"✓ Reconocedor cargado: {len(reconocedor.clases)} clases")
        logger.info(f"  Arquitectura: {reconocedor.arquitectura}")
        logger.info(f"  Accuracy: {reconocedor.accuracy*100:.2f}%")
        logger.info(f"  Device: {reconocedor.device}")
        
        return reconocedor
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error obteniendo reconocedor: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {str(e)}"
        )

@router.post("/voz-a-senas", response_model=RespuestaAPI)
async def traducir_voz_a_senas(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Traduce texto/voz a señas"""
    try:
        texto = data.get("texto", "").strip()
        
        if not texto:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requiere texto para traducir"
            )
        
        palabras = procesar_texto_para_traduccion(texto)
        senas_traducidas = []
        
        for palabra in palabras:
            palabra_normalizada = normalizar_palabra(palabra)
            
            if palabra_normalizada in DICCIONARIO_SENAS:
                info_sena = DICCIONARIO_SENAS[palabra_normalizada]
                sena_traducida = {
                    "palabra": palabra,
                    "sena": palabra_normalizada,
                    "confianza": 1.0,
                    "categoria": info_sena["categoria"],
                    "descripcion": info_sena["descripcion"],
                    "dificultad": info_sena["dificultad"]
                }
                senas_traducidas.append(sena_traducida)
            else:
                # Deletrear palabra larga
                if len(palabra_normalizada) <= 10:
                    for letra in palabra_normalizada:
                        if letra.upper() in DICCIONARIO_SENAS:
                            info_letra = DICCIONARIO_SENAS[letra.upper()]
                            sena_letra = {
                                "palabra": letra,
                                "sena": letra.upper(),
                                "confianza": 0.9,
                                "categoria": "alfabeto",
                                "descripcion": f"Deletreo de '{letra}'",
                                "dificultad": "facil"
                            }
                            senas_traducidas.append(sena_letra)
        
        resultado = {
            "modo": "voz-a-senas",
            "texto_original": texto,
            "senas_detectadas": senas_traducidas,
            "palabras_procesadas": len(palabras),
            "senas_encontradas": len(senas_traducidas)
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Traduccion completada exitosamente",
            datos=resultado
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en traduccion voz-a-senas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en traduccion: {str(e)}"
        )

@router.post("/senas-a-texto", response_model=RespuestaAPI)
async def traducir_senas_a_texto(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Traduce señas a texto usando ReconocimientoAdaptativoIA - VERSIÓN MEJORADA
    """
    start_time = time.time()
    
    try:
        video_base64 = data.get("video_base64")
        frames_base64 = data.get("frames_base64", [])
        imagen_base64 = data.get("imagen_base64")
        sena_esperada = data.get("sena_esperada")
        configuracion = data.get("configuracion", {})
        
        if not video_base64 and not frames_base64 and not imagen_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requiere video, frames o imagen para reconocimiento"
            )
        
        logger.info("🔍 Iniciando reconocimiento de senas - MODO OPTIMIZADO")
        
        # USAR EL RECONOCEDOR ADAPTATIVO
        reconocedor = obtener_reconocedor(bd)
        
        # CONFIGURACIÓN MEJORADA - CONFIANZA MÍNIMA MÁS ALTA
        confianza_minima = configuracion.get("confianza_minima", 0.70)  # AUMENTADO de 0.4 a 0.70
        
        # Procesar según tipo de entrada
        sena_detectada = "desconocido"
        confianza = 0.0
        detalles = {}
        
        try:
            if video_base64:
                logger.info("📹 Procesando video base64")
                # Guardar video temporalmente
                video_path = guardar_video_temporal(video_base64)
                try:
                    sena_detectada, confianza, detalles = reconocedor.predecir_desde_video(
                        video_path, 
                        sena_esperada=sena_esperada
                    )
                finally:
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                
            elif frames_base64 and len(frames_base64) > 0:
                logger.info(f"🖼️ Procesando {len(frames_base64)} frames")
                
                # OPTIMIZACIÓN: Limitar número de frames procesados
                frames_a_procesar = frames_base64[-12:]  # Solo últimos 12 frames
                
                frames = []
                for i, frame_b64 in enumerate(frames_a_procesar):
                    try:
                        frame = procesar_frame_base64_optimizado(frame_b64)
                        if frame is not None:
                            frames.append(frame)
                    except Exception as e:
                        logger.warning(f"⚠️ Error procesando frame {i}: {e}")
                        continue
                
                if len(frames) >= 6:  # Mínimo 6 frames válidos
                    sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(
                        frames,
                        sena_esperada=sena_esperada
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Frames insuficientes para reconocimiento ({len(frames)}/6)"
                    )
                    
            elif imagen_base64:
                logger.info("📸 Procesando imagen única")
                frame = procesar_frame_base64_optimizado(imagen_base64)
                if frame is not None:
                    sena_detectada, confianza, detalles = reconocedor.predecir_desde_frame(
                        frame,
                        sena_esperada=sena_esperada
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No se pudo procesar la imagen"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No se pudieron procesar los datos de entrada"
                )
                
        except Exception as e:
            logger.error(f"✗ Error en procesamiento: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en procesamiento: {str(e)}"
            )
        
        # VALIDACIÓN MEJORADA DE CONFIANZA
        processing_time = time.time() - start_time
        
        # Log detallado del resultado
        logger.info(f"📊 Resultado crudo: {sena_detectada} - {confianza:.4f}")
        logger.info(f"⏱️ Tiempo procesamiento: {processing_time:.2f}s")
        
        # Obtener información de la seña
        info_sena = DICCIONARIO_SENAS.get(sena_detectada, {
            "categoria": detalles.get("tipo_real", "desconocido"),
            "descripcion": f"Seña {sena_detectada}",
            "dificultad": "facil"
        })
        
        # FILTRAR ALTERNATIVAS CON CONFIANZA MÍNIMA
        alternativas_filtradas = []
        if "alternativas" in detalles:
            for alt in detalles["alternativas"]:
                if alt.get("confianza", 0) >= 0.3:  # Solo mostrar >30%
                    alternativas_filtradas.append(alt)
        
        # CLASIFICACIÓN MEJORADA DE RESULTADOS
        if confianza < confianza_minima:
            mensaje = f"Confianza insuficiente ({confianza * 100:.1f}% < {confianza_minima * 100}%)"
            calidad = "baja"
            texto_traducido = "?"
            
            # Si hay alternativas con buena confianza, sugerir la mejor
            if alternativas_filtradas and alternativas_filtradas[0]["confianza"] >= confianza_minima:
                mejor_alternativa = alternativas_filtradas[0]
                mensaje = f"Posible: {mejor_alternativa['sena']} ({mejor_alternativa['confianza'] * 100:.1f}%)"
                sena_detectada = mejor_alternativa['sena']
                confianza = mejor_alternativa['confianza']
                texto_traducido = sena_detectada
                calidad = "moderada"
                
        elif confianza >= 0.85:
            mensaje = "Reconocimiento excelente"
            calidad = "excelente"
            texto_traducido = sena_detectada
        elif confianza >= 0.75:
            mensaje = "Reconocimiento bueno"
            calidad = "buena"
            texto_traducido = sena_detectada
        elif confianza >= 0.70:
            mensaje = "Reconocimiento aceptable"
            calidad = "aceptable"
            texto_traducido = sena_detectada
        else:
            mensaje = "Confianza baja"
            calidad = "baja"
            texto_traducido = sena_detectada
        
        # Construir resultado
        resultado = {
            "modo": "senas-a-texto",
            "tipo_entrada": detalles.get("tipo_sena", "desconocido"),
            "texto_traducido": texto_traducido,
            "sena_detectada": sena_detectada,
            "confianza": round(confianza, 4),
            "confianza_raw": round(confianza, 4),
            "calidad": calidad,
            "mensaje": mensaje,
            "categoria": info_sena.get("categoria", "desconocido"),
            "descripcion": info_sena.get("descripcion", f"Seña {sena_detectada}"),
            "alternativas": alternativas_filtradas[:3],  # Máximo 3 alternativas
            "tiempo_procesamiento": round(processing_time, 2),
            "detalles_modelo": {
                "arquitectura": detalles.get("modelo", "desconocido"),
                "tipo_procesamiento": detalles.get("tipo_sena", "desconocido"),
                "tipo_real": detalles.get("tipo_real", "desconocido"),
                "num_frames": detalles.get("num_frames_usado", 0)
            }
        }
        
        logger.info(f"✅ Reconocimiento final: {sena_detectada} - {confianza*100:.1f}% - {calidad}")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Reconocimiento completado",
            datos=resultado
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error en reconocimiento: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en reconocimiento: {str(e)}"
        )

def guardar_video_temporal(video_base64: str) -> str:
    """Guarda video base64 en archivo temporal"""
    try:
        if ',' in video_base64:
            video_base64 = video_base64.split(',')[1]
        
        video_bytes = base64.b64decode(video_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(video_bytes)
            return temp_file.name
            
    except Exception as e:
        logger.error(f"Error guardando video temporal: {e}")
        raise

def procesar_frame_base64(frame_base64: str) -> np.ndarray:
    """Decodifica frame de base64 a numpy array"""
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_base64)
        frame_np = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("No se pudo decodificar el frame")
        
        return frame
        
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        raise

def procesar_frame_base64_optimizado(frame_base64: str) -> Optional[np.ndarray]:
    """Decodificación optimizada de frame con redimensionamiento"""
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_base64)
        frame_np = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        # REDIMENSIONAR a tamaño estándar para mejor consistencia
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return frame
        
    except Exception as e:
        logger.warning(f"⚠️ Error procesando frame optimizado: {e}")
        return None

def procesar_texto_para_traduccion(texto: str) -> List[str]:
    """Procesa texto para traducción"""
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto.strip())
    return texto.split()

def normalizar_palabra(palabra: str) -> str:
    """Normaliza palabra removiendo acentos"""
    palabra = unicodedata.normalize('NFD', palabra.lower())
    palabra = ''.join(c for c in palabra if unicodedata.category(c) != 'Mn')
    return palabra.title()

@router.get("/estado-reconocedor")
async def obtener_estado_reconocedor(
    bd: Session = Depends(obtener_bd)
):
    """Obtiene el estado del reconocedor cargado"""
    try:
        reconocedor = _reconocedor_cache.get('reconocedor')
        
        if reconocedor is None:
            # Intentar cargar el reconocedor
            try:
                reconocedor = obtener_reconocedor(bd)
            except Exception as e:
                return {
                    "estado": "inactivo",
                    "mensaje": f"No se pudo cargar el reconocedor: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        
        estado = {
            "estado": "activo",
            "modelo_cargado": True,
            "arquitectura": reconocedor.arquitectura,
            "clases_cargadas": len(reconocedor.clases),
            "num_frames": reconocedor.num_frames,
            "accuracy_entrenamiento": round(reconocedor.accuracy * 100, 2),
            "device": str(reconocedor.device),
            "amp_habilitado": reconocedor.enable_amp,
            "ultimo_uso": _reconocedor_cache['ultimo_uso'].isoformat() if _reconocedor_cache['ultimo_uso'] else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return estado
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return {
            "estado": "error",
            "mensaje": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/limpiar-cache")
async def limpiar_cache_reconocedor():
    """Limpia el cache del reconocedor"""
    try:
        _reconocedor_cache['reconocedor'] = None
        _reconocedor_cache['modelo_id'] = None
        _reconocedor_cache['ultimo_uso'] = None
        
        # Limpiar memoria de GPU si está disponible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✓ Memoria CUDA liberada")
        except ImportError:
            pass
        
        logger.info("✓ Cache del reconocedor limpiado")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Cache limpiado exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error limpiando cache: {str(e)}"
        )

@router.get("/diccionario") 
async def obtener_diccionario_senas():
    """Obtiene el diccionario de señas disponibles"""
    return {
        "total_senas": len(DICCIONARIO_SENAS),
        "senas": list(DICCIONARIO_SENAS.keys()),
        "categorias": {
            "alfabeto": len([s for s in DICCIONARIO_SENAS if DICCIONARIO_SENAS[s]['categoria'] == 'alfabeto'])
        },
        "timestamp": datetime.now().isoformat()
    }