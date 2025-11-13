from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from datetime import datetime
from enum import Enum

class TipoVideo(str, Enum):
    YOUTUBE = "youtube"
    GOOGLE_DRIVE = "google_drive"
    VIMEO = "vimeo"


class ClaseCrear(BaseModel):
    leccion_id: int = Field(..., gt=0)
    titulo: str = Field(..., min_length=3, max_length=255)
    descripcion: Optional[str] = None
    contenido_texto: Optional[str] = None
    sena: Union[str, None] = None
    tipo_video: TipoVideo = TipoVideo.YOUTUBE
    video_url: Optional[str] = Field(None, max_length=500)
    video_id: Optional[str] = Field(None, max_length=255)
    imagen_referencia: Optional[str] = Field(None, max_length=500)
    gif_demostracion: Optional[str] = Field(None, max_length=500)
    orden: int = Field(..., gt=0)
    duracion_estimada: Optional[int] = Field(10, gt=0)
    tips: Optional[str] = None
    errores_comunes: Optional[str] = None
    requiere_practica: bool = True
    intentos_minimos: int = Field(3, gt=0)
    precision_minima: float = Field(0.7, ge=0, le=1)
    activa: bool = True

    @validator('sena')
    def validar_sena(cls, v, values):
        requiere_practica = values.get('requiere_practica', True)
        
        if requiere_practica:
            if v is None:
                raise ValueError('La seña es obligatoria cuando se requiere práctica')
            if isinstance(v, str) and not v.strip():
                raise ValueError('La seña es obligatoria cuando se requiere práctica')
        
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        
        return v

    @validator('descripcion', 'contenido_texto', 'tips', 'errores_comunes', 'video_url', 'video_id', 'imagen_referencia', 'gif_demostracion')
    def convertir_vacios_a_none(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

    class Config:
        use_enum_values = True


class ClaseActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=3, max_length=255)
    descripcion: Optional[str] = None
    contenido_texto: Optional[str] = None
    sena: Union[str, None] = None
    tipo_video: Optional[TipoVideo] = None
    video_url: Optional[str] = Field(None, max_length=500)
    video_id: Optional[str] = Field(None, max_length=255)
    imagen_referencia: Optional[str] = Field(None, max_length=500)
    gif_demostracion: Optional[str] = Field(None, max_length=500)
    orden: Optional[int] = Field(None, gt=0)
    duracion_estimada: Optional[int] = Field(None, gt=0)
    tips: Optional[str] = None
    errores_comunes: Optional[str] = None
    requiere_practica: Optional[bool] = None
    intentos_minimos: Optional[int] = Field(None, gt=0)
    precision_minima: Optional[float] = Field(None, ge=0, le=1)
    activa: Optional[bool] = None

    @validator('sena')
    def validar_sena(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

    @validator('descripcion', 'contenido_texto', 'tips', 'errores_comunes', 'video_url', 'video_id', 'imagen_referencia', 'gif_demostracion')
    def convertir_vacios_a_none(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

    class Config:
        use_enum_values = True


class ClaseRespuesta(BaseModel):
    id: int
    leccion_id: int
    titulo: str
    descripcion: Optional[str] = None
    contenido_texto: Optional[str] = None
    sena: Optional[str] = None
    tipo_video: str
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    imagen_referencia: Optional[str] = None
    gif_demostracion: Optional[str] = None
    orden: int
    duracion_estimada: Optional[int] = None
    tips: Optional[str] = None
    errores_comunes: Optional[str] = None
    requiere_practica: bool
    intentos_minimos: int
    precision_minima: float
    activa: bool
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None

    class Config:
        from_attributes = True
        from_attributes = True
        use_enum_values = True

    @validator('tipo_video', pre=True)
    def convert_enum(cls, v):
        if hasattr(v, 'value'):
            return v.value
        return v