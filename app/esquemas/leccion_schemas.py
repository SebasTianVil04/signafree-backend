# app/esquemas/leccion.py
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class LeccionBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    sena: str = Field(..., min_length=1, max_length=100)
    orden: int = Field(..., ge=1)
    nivel_dificultad: int = Field(default=1, ge=1, le=3)
    activa: bool = True
    bloqueada: bool = False
    leccion_previa_id: Optional[int] = None
    puntos_base: int = Field(default=10, ge=0)
    puntos_perfecto: int = Field(default=20, ge=0)

class LeccionCrear(LeccionBase):
    categoria_id: int = Field(..., ge=1)
    
    @validator('puntos_perfecto')
    def validar_puntos(cls, v, values):
        if 'puntos_base' in values and v < values['puntos_base']:
            raise ValueError('puntos_perfecto debe ser mayor o igual a puntos_base')
        return v

class LeccionActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    sena: Optional[str] = Field(None, min_length=1, max_length=100)
    orden: Optional[int] = Field(None, ge=1)
    nivel_dificultad: Optional[int] = Field(None, ge=1, le=3)
    activa: Optional[bool] = None
    bloqueada: Optional[bool] = None
    leccion_previa_id: Optional[int] = None
    puntos_base: Optional[int] = Field(None, ge=0)
    puntos_perfecto: Optional[int] = Field(None, ge=0)

class LeccionRespuesta(LeccionBase):
    id: int
    categoria_id: int
    categoria_nombre: Optional[str] = None
    total_clases: int = 0
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ClaseBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    contenido: str = Field(..., min_length=1)
    orden: int = Field(..., ge=1)
    video_id: Optional[str] = None
    tipo_video: Optional[str] = None
    duracion: Optional[int] = Field(None, ge=0)
    activa: bool = True

class ClaseCrear(ClaseBase):
    leccion_id: int = Field(..., ge=1)

class ClaseActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    contenido: Optional[str] = Field(None, min_length=1)
    orden: Optional[int] = Field(None, ge=1)
    video_id: Optional[str] = None
    tipo_video: Optional[str] = None
    duracion: Optional[int] = Field(None, ge=0)
    activa: Optional[bool] = None

class ClaseRespuesta(ClaseBase):
    id: int
    leccion_id: int
    fecha_creacion: datetime
    
    class Config:
        from_attributes = True

class ExamenBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    tipo: str = Field(..., pattern='^(nivel|final)$')
    nivel: Optional[int] = Field(None, ge=1)
    orden: int = Field(..., ge=1)
    clases_requeridas: int = Field(default=0, ge=0)
    requiere_todas_clases: bool = False
    tiempo_limite: Optional[int] = Field(None, ge=0)
    puntuacion_minima: int = Field(default=70, ge=0, le=100)
    activo: bool = True

class ExamenCrear(ExamenBase):
    leccion_id: int = Field(..., ge=1)

class ExamenActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    tipo: Optional[str] = Field(None, pattern='^(nivel|final)$')
    nivel: Optional[int] = Field(None, ge=1)
    orden: Optional[int] = Field(None, ge=1)
    clases_requeridas: Optional[int] = Field(None, ge=0)
    requiere_todas_clases: Optional[bool] = None
    tiempo_limite: Optional[int] = Field(None, ge=0)
    puntuacion_minima: Optional[int] = Field(None, ge=0, le=100)
    activo: Optional[bool] = None

class ExamenRespuesta(ExamenBase):
    id: int
    leccion_id: int
    total_preguntas: int = 0
    fecha_creacion: datetime
    disponible: bool = False
    completado: bool = False
    mejor_calificacion: Optional[int] = None
    
    class Config:
        from_attributes = True

class ProgresoExamen(BaseModel):
    examen_id: int
    completado: bool = False
    mejor_calificacion: int = 0
    intentos_realizados: int = 0
    ultimo_intento: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class RespuestaAPI(BaseModel):
    exito: bool
    mensaje: str
    datos: Optional[dict] = None

class RespuestaLista(BaseModel):
    exito: bool
    mensaje: str
    datos: list
    total: int
    pagina: int = 1
    por_pagina: int = 10