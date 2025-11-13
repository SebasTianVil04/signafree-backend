import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import tempfile
import gc
import subprocess
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
import time

from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..servicios.config_tipo_senas import detectar_tipo_sena, obtener_config_sena

logger = logging.getLogger(__name__)

class DatasetService:
    
    def __init__(self):
        self.video_dir = Path("archivos_subidos") / "videos_dataset"
        self.frames_dir = Path("archivos_subidos") / "frames_dataset"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.resolucion_frame = (224, 224)
        self.ffmpeg_disponible = self._verificar_ffmpeg()
        logger.info(f"DatasetService inicializado - FFmpeg: {'Disponible' if self.ffmpeg_disponible else 'No disponible'}")

    def _verificar_ffmpeg(self) -> bool:
        """Verifica si FFmpeg está disponible"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _convertir_webm_a_mp4(self, ruta_webm: str, ruta_mp4: str) -> bool:
        """Convierte WebM a MP4 usando FFmpeg - MEJORADO"""
        if not self.ffmpeg_disponible:
            logger.warning("FFmpeg no disponible, saltando conversión")
            return False
        
        try:
            # Comando FFmpeg simplificado y más robusto
            cmd = [
                'ffmpeg',
                '-i', ruta_webm,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                '-pix_fmt', 'yuv420p',  # Formato de píxeles compatible
                '-movflags', '+faststart',  # Optimización para streaming
                '-vf', 'fps=30',
                '-an',  # Sin audio
                '-y',  # Sobrescribir
                ruta_mp4
            ]
            
            logger.info(f"Convirtiendo WebM a MP4: {os.path.basename(ruta_webm)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Aumentado a 60 segundos
            )
            
            if result.returncode == 0 and os.path.exists(ruta_mp4):
                tamaño_mp4 = os.path.getsize(ruta_mp4)
                if tamaño_mp4 > 1024:  # Al menos 1KB
                    logger.info(f"✓ Conversión exitosa: {tamaño_mp4:,} bytes")
                    return True
                else:
                    logger.error(f"✗ MP4 muy pequeño: {tamaño_mp4} bytes")
                    if os.path.exists(ruta_mp4):
                        os.remove(ruta_mp4)
                    return False
            else:
                logger.error(f"✗ FFmpeg falló con código {result.returncode}")
                if result.stderr:
                    logger.error(f"Error FFmpeg: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ Timeout en conversión FFmpeg (60s)")
            return False
        except Exception as e:
            logger.error(f"✗ Error en conversión: {e}")
            return False

    def _extraer_frames_opencv_robusto(self, ruta_video: str, nombre_sena: str, max_frames: int = 20) -> Tuple[int, float]:
        """
        Extracción ROBUSTA de frames con múltiples estrategias
        """
        # Estrategia 1: Intentar con diferentes backends de OpenCV
        backends = [
            ('CAP_FFMPEG', cv2.CAP_FFMPEG),
            ('CAP_ANY (default)', cv2.CAP_ANY),
        ]
        
        for backend_name, backend_flag in backends:
            logger.info(f"🔍 Intentando extraer frames con {backend_name}...")
            resultado = self._extraer_frames_con_backend(
                ruta_video, nombre_sena, max_frames, backend_flag, backend_name
            )
            
            if resultado[0] > 0:  # Si extrajo al menos 1 frame
                logger.info(f"✓ Extracción exitosa con {backend_name}: {resultado[0]} frames")
                return resultado
            else:
                logger.warning(f"⚠ Falló con {backend_name}, intentando siguiente método...")
        
        # Si llegamos aquí, todos los backends fallaron
        logger.error(f"✗ No se pudieron extraer frames con ningún método")
        return 0, 0.0

    def _extraer_frames_con_backend(
        self, 
        ruta_video: str, 
        nombre_sena: str, 
        max_frames: int,
        backend: int,
        backend_name: str
    ) -> Tuple[int, float]:
        """
        Intenta extraer frames usando un backend específico
        """
        cap = None
        frames_guardados = 0
        calidad_total = 0.0
        
        try:
            # Abrir video con el backend especificado
            cap = cv2.VideoCapture(ruta_video, backend)
            
            if not cap.isOpened():
                return 0, 0.0
            
            # Obtener información del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps is None or fps <= 0 or fps > 500:
                fps = 30.0
            
            logger.info(f"   Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
            
            # VALIDACIÓN: Si las dimensiones son claramente inválidas
            if width <= 0 or height <= 0 or width > 10000 or height > 10000:
                logger.warning(f"   ⚠ Dimensiones inválidas: {width}x{height}")
                return 0, 0.0
            
            # Estrategia de extracción
            frame_count = 0
            frames_saltados = max(1, 3)
            intentos_fallidos_consecutivos = 0
            max_intentos_consecutivos = 10
            max_iteraciones = 150
            
            while frames_guardados < max_frames and frame_count < max_iteraciones:
                # Leer frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    intentos_fallidos_consecutivos += 1
                    if intentos_fallidos_consecutivos >= max_intentos_consecutivos:
                        logger.warning(f"   ⚠ {max_intentos_consecutivos} fallos consecutivos, deteniendo")
                        break
                    frame_count += 1
                    continue
                
                # Solo procesar cada N frames
                if frame_count % frames_saltados != 0:
                    frame_count += 1
                    continue
                
                # VALIDACIÓN CRÍTICA: Verificar shape del frame
                if len(frame.shape) != 3:
                    logger.warning(f"   ⚠ Shape inválido: {frame.shape}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                frame_height, frame_width, frame_channels = frame.shape
                
                # Verificar dimensiones razonables
                if frame_height < 10 or frame_width < 10:
                    logger.warning(f"   ⚠ Frame muy pequeño: {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                # Verificar que no sea corrupto (como el error que vimos: (1, 54632, 3))
                # Este patrón indica que el frame está corrupto y los bytes están mal interpretados
                if frame_height == 1 or frame_width == 1:
                    logger.warning(f"   ⚠ Frame corrupto (dimensión = 1): {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                if frame_height > 10000 or frame_width > 10000:
                    logger.warning(f"   ⚠ Frame corrupto (muy grande): {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                # Verificar canales
                if frame_channels not in [1, 3, 4]:
                    logger.warning(f"   ⚠ Canales inválidos: {frame_channels}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                # Frame válido!
                intentos_fallidos_consecutivos = 0
                
                try:
                    # Normalizar canales
                    if frame_channels == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame_channels == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Redimensionar
                    frame_resized = cv2.resize(frame, self.resolucion_frame, interpolation=cv2.INTER_AREA)
                    
                    # Guardar
                    nombre_frame = f"{nombre_sena}_{frames_guardados:05d}.jpg"
                    ruta_frame = self.frames_dir / nombre_frame
                    
                    if cv2.imwrite(str(ruta_frame), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                        frames_guardados += 1
                        
                        # Calcular calidad
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
                            calidad = min(1.0, nitidez / 100.0)
                            calidad_total += max(0.3, calidad)
                        except:
                            calidad_total += 0.5
                        
                        if frames_guardados % 5 == 0:
                            logger.info(f"   ✓ {frames_guardados}/{max_frames} frames")
                    
                except Exception as e:
                    logger.warning(f"   ⚠ Error procesando frame: {e}")
                    intentos_fallidos_consecutivos += 1
                
                frame_count += 1
            
            calidad_promedio = calidad_total / frames_guardados if frames_guardados > 0 else 0.5
            return frames_guardados, calidad_promedio
            
        except Exception as e:
            logger.error(f"   ✗ Error con {backend_name}: {e}")
            return 0, 0.0
        finally:
            if cap is not None:
                cap.release()

    def _procesar_video_directo(self, ruta_video: str, nombre_sena: str) -> Tuple[int, float, float, int, float]:
        """Procesa video y extrae metadata + frames"""
        cap = None
        
        try:
            # Extraer frames con método robusto
            frames_extraidos, calidad_promedio = self._extraer_frames_opencv_robusto(ruta_video, nombre_sena)
            
            # Obtener metadata del video
            cap = cv2.VideoCapture(ruta_video)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps is None or fps <= 0 or fps > 500:
                    fps = 30.0
                
                if total_frames <= 0 or total_frames > 1000000:
                    total_frames = max(frames_extraidos * 2, 30)
                
                duracion = total_frames / fps if fps > 0 else 1.0
                
                cap.release()
            else:
                fps = 30.0
                total_frames = frames_extraidos * 2
                duracion = 1.0
            
            return frames_extraidos, calidad_promedio, fps, total_frames, duracion
            
        except Exception as e:
            logger.error(f"✗ Error procesando video: {e}", exc_info=True)
            return 0, 0.5, 30.0, 0, 1.0
        finally:
            if cap is not None:
                cap.release()

    async def subir_video_dataset(
        self,
        db: Session,
        archivo: any,
        categoria_id: int,
        sena: str,
        usuario_id: int
    ) -> VideoDataset:
        """Sube y procesa un video para el dataset"""
        temp_video_path = None
        temp_mp4_path = None
        
        try:
            # Leer contenido del archivo
            if hasattr(archivo, 'read'):
                contenido_video = await archivo.read()
            else:
                contenido_video = archivo
            
            if len(contenido_video) < 1024:
                raise ValueError("Archivo de video demasiado pequeño")
            
            logger.info(f"📹 Procesando video: {sena}, {len(contenido_video):,} bytes")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # 1. GUARDAR WEBM ORIGINAL
            nombre_webm = f"{sena}_{timestamp}.webm"
            ruta_webm_final = self.video_dir / nombre_webm
            
            with open(ruta_webm_final, 'wb') as f:
                f.write(contenido_video)
            
            logger.info(f"✓ WebM guardado: {ruta_webm_final}")
            
            # 2. CREAR ARCHIVO TEMPORAL PARA CONVERSIÓN
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(contenido_video)
                temp_video_path = temp_file.name
            
            # 3. INTENTAR CONVERSIÓN A MP4
            conversion_exitosa = False
            formato_guardado = "webm"
            ruta_video_final = str(ruta_webm_final)
            
            if self.ffmpeg_disponible:
                nombre_mp4 = f"{sena}_{timestamp}.mp4"
                ruta_mp4_final = self.video_dir / nombre_mp4
                temp_mp4_path = temp_video_path.replace('.webm', '_temp.mp4')
                
                logger.info(f"🔄 Convirtiendo a MP4...")
                
                if self._convertir_webm_a_mp4(temp_video_path, temp_mp4_path):
                    # Verificar que el MP4 es válido
                    if os.path.exists(temp_mp4_path) and os.path.getsize(temp_mp4_path) > 1024:
                        import shutil
                        shutil.move(temp_mp4_path, str(ruta_mp4_final))
                        conversion_exitosa = True
                        formato_guardado = "mp4"
                        ruta_video_final = str(ruta_mp4_final)
                        
                        logger.info(f"✓ MP4 guardado exitosamente: {ruta_mp4_final}")
                        
                        # Eliminar WebM si MP4 está bien
                        try:
                            os.remove(ruta_webm_final)
                            logger.info("✓ WebM eliminado (MP4 es la versión final)")
                        except Exception as e:
                            logger.warning(f"No se pudo eliminar WebM: {e}")
                    else:
                        logger.warning("⚠ MP4 generado es inválido, intentando con WebM")
                else:
                    logger.warning("⚠ Conversión a MP4 falló, intentando con WebM")
            else:
                logger.warning("⚠ FFmpeg no disponible, intentando procesar WebM directamente")
                logger.warning("   RECOMENDACIÓN: Instale FFmpeg para mejor compatibilidad")
            
            # 4. EXTRAER FRAMES DEL VIDEO FINAL
            logger.info(f"📊 Extrayendo frames de {formato_guardado.upper()}...")
            frames_extraidos, calidad_promedio, fps_real, total_frames, duracion_real = self._procesar_video_directo(
                ruta_video_final, sena
            )
            
            # 5. DETERMINAR APROBACIÓN
            aprobado = frames_extraidos >= 3
            
            # 6. CREAR REGISTRO EN BD
            video_db = VideoDataset(
                categoria_id=categoria_id,
                usuario_id=usuario_id,
                sena=sena.upper(),
                ruta_video=ruta_video_final,
                duracion_segundos=duracion_real,
                fps=fps_real,
                resolucion=json.dumps({"ancho": 640, "alto": 480}),
                tamaño_bytes=len(contenido_video),
                formato=formato_guardado,
                frames_extraidos=frames_extraidos,
                calidad_promedio=calidad_promedio,
                procesado=True,
                aprobado=aprobado,
                fecha_procesado=datetime.utcnow(),
                notas=f"Formato: {formato_guardado.upper()}, Frames: {frames_extraidos}, Calidad: {calidad_promedio:.2f}"
            )
            
            if aprobado:
                video_db.fecha_aprobado = datetime.utcnow()

            db.add(video_db)
            db.commit()
            db.refresh(video_db)

            estado = "✓ APROBADO" if aprobado else "⚠ PENDIENTE"
            logger.info(f"{estado} - Video {video_db.id} - {frames_extraidos} frames - {formato_guardado.upper()}")

            return video_db

        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error procesando video: {str(e)}", exc_info=True)
            
            # Registro de error en BD
            try:
                video_db = VideoDataset(
                    categoria_id=categoria_id,
                    usuario_id=usuario_id,
                    sena=sena.upper(),
                    ruta_video="error",
                    duracion_segundos=0,
                    fps=30.0,
                    resolucion=json.dumps({"ancho": 0, "alto": 0}),
                    tamaño_bytes=len(contenido_video) if 'contenido_video' in locals() else 0,
                    formato="webm",
                    frames_extraidos=0,
                    calidad_promedio=0.0,
                    procesado=False,
                    aprobado=False,
                    fecha_procesado=datetime.utcnow(),
                    notas=f"ERROR: {str(e)}"
                )
                db.add(video_db)
                db.commit()
                db.refresh(video_db)
                return video_db
            except Exception as inner_e:
                logger.error(f"✗ Error en fallback: {inner_e}")
                raise Exception(f"Error procesando video: {str(e)}")
            
        finally:
            # Limpiar archivos temporales
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                try:
                    os.unlink(temp_mp4_path)
                except:
                    pass
            gc.collect()

    def obtener_directorio_frames(self, sena: str) -> Optional[Path]:
        """Obtiene el directorio donde están los frames"""
        return self.frames_dir

    def preparar_dataset_entrenamiento(
        self,
        db: Session,
        categoria_ids: Optional[List[int]] = None
    ) -> Dict[str, List[str]]:
        """Prepara dataset para entrenamiento"""
        query = db.query(VideoDataset).filter(VideoDataset.aprobado == True)
        
        if categoria_ids:
            query = query.filter(VideoDataset.categoria_id.in_(categoria_ids))
        
        dataset = {}
        videos = query.all()
        
        for video in videos:
            sena = video.sena
            if sena not in dataset:
                dataset[sena] = []
            if os.path.exists(video.ruta_video):
                dataset[sena].append(video.ruta_video)
        
        logger.info(f"📊 Dataset preparado: {len(dataset)} señas, {sum(len(v) for v in dataset.values())} videos")
        return dataset

    def obtener_estadisticas_dataset(self, db: Session) -> Dict[str, any]:
        """Obtiene estadísticas del dataset"""
        try:
            total_videos = db.query(VideoDataset).count()
            videos_aprobados = db.query(VideoDataset).filter(VideoDataset.aprobado == True).count()
            total_frames = db.query(func.sum(VideoDataset.frames_extraidos)).scalar() or 0
            
            por_sena = db.query(
                VideoDataset.sena,
                func.count(VideoDataset.id).label('videos'),
                func.sum(VideoDataset.frames_extraidos).label('frames'),
                func.avg(VideoDataset.calidad_promedio).label('calidad_promedio')
            ).group_by(VideoDataset.sena).all()
            
            por_sena_list = []
            for item in por_sena:
                por_sena_list.append({
                    "sena": str(item[0]),
                    "videos": int(item[1]),
                    "frames": int(item[2]) if item[2] else 0,
                    "calidad_promedio": float(item[3]) if item[3] else 0.0
                })
            
            return {
                "total_videos": total_videos,
                "videos_aprobados": videos_aprobados,
                "total_frames": int(total_frames),
                "por_sena": sorted(por_sena_list, key=lambda x: x['videos'], reverse=True),
                "total_senas": len(por_sena_list),
                "tasa_aprobacion": round(videos_aprobados / total_videos * 100, 2) if total_videos > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"✗ Error obteniendo estadísticas: {str(e)}")
            return {
                "total_videos": 0,
                "videos_aprobados": 0,
                "total_frames": 0,
                "por_sena": [],
                "total_senas": 0,
                "tasa_aprobacion": 0
            }
    
    def aprobar_video(self, db: Session, video_id: int, aprobar: bool, notas: str = None) -> VideoDataset:
        """Aprueba o rechaza un video"""
        try:
            video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
            
            if not video:
                raise Exception("Video no encontrado")
            
            video.aprobado = aprobar
            video.fecha_aprobado = datetime.utcnow() if aprobar else None
            
            if notas is not None and hasattr(video, 'notas'):
                video.notas = notas
            
            db.commit()
            db.refresh(video)
            
            logger.info(f"✓ Video {video_id} {'aprobado' if aprobar else 'rechazado'}")
            
            return video
            
        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error aprobando video: {str(e)}")
            raise Exception(f"Error aprobando video: {str(e)}")

    def reprocesar_video(self, db: Session, video_id: int) -> VideoDataset:
        """Reprocesa un video existente - CORREGIDO"""
        try:
            video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
            
            if not video:
                raise Exception("Video no encontrado")
            
            if not os.path.exists(video.ruta_video):
                raise Exception("Archivo de video no encontrado")
            
            logger.info(f"🔄 Reprocesando video {video_id}: {video.sena}")
            
            # Extraer frames nuevamente
            frames_extraidos, calidad_promedio, fps_real, total_frames, duracion_real = self._procesar_video_directo(
                video.ruta_video, video.sena
            )
            
            # Actualizar registro
            video.frames_extraidos = frames_extraidos
            video.calidad_promedio = calidad_promedio
            video.fps = fps_real
            video.duracion_segundos = duracion_real
            video.procesado = True
            video.aprobado = frames_extraidos >= 3
            video.fecha_procesado = datetime.utcnow()
            video.notas = f"Reprocesado: {frames_extraidos} frames, Calidad: {calidad_promedio:.2f}"
            
            if video.aprobado:
                video.fecha_aprobado = datetime.utcnow()
            
            db.commit()
            db.refresh(video)  # CORREGIDO: era video_db
            
            logger.info(f"✓ Video {video_id} reprocesado: {frames_extraidos} frames")
            
            return video
            
        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error reprocesando video: {str(e)}")
            raise Exception(f"Error reprocesando video: {str(e)}")

dataset_service = DatasetService()