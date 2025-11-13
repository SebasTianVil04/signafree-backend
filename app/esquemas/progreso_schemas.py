# app/esquemas/progreso_schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ProgresoClaseBase(BaseModel):
    vista: bool = False
    completada: bool = False
    intentos_realizados: int = 0
    mejor_precision: float = 0.0
    ultima_precision: float = 0.0
    tiempo_video: int = 0
    tiempo_practica: int = 0

class ProgresoClaseCrear(ProgresoClaseBase):
    usuario_id: int
    clase_id: int

class ProgresoClaseActualizar(BaseModel):
    vista: Optional[bool] = None
    completada: Optional[bool] = None
    intentos_realizados: Optional[int] = None
    mejor_precision: Optional[float] = None
    ultima_precision: Optional[float] = None
    tiempo_video: Optional[int] = None
    tiempo_practica: Optional[int] = None

class ProgresoClaseRespuesta(ProgresoClaseBase):
    id: int
    usuario_id: int
    clase_id: int
    fecha_primera_vista: Optional[datetime] = None
    fecha_completada: Optional[datetime] = None
    ultima_practica: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ProgresoLeccionBase(BaseModel):
    desbloqueada: bool = False
    iniciada: bool = False
    completada: bool = False
    total_clases: int = 0
    clases_completadas: int = 0
    mejor_precision: float = 0.0
    total_intentos: int = 0
    total_puntos: int = 0
    estrellas: int = Field(0, ge=0, le=3)

class ProgresoLeccionCrear(ProgresoLeccionBase):
    usuario_id: int
    leccion_id: int

class ProgresoLeccionRespuesta(ProgresoLeccionBase):
    id: int
    usuario_id: int
    leccion_id: int
    porcentaje_completado: float
    porcentaje_precision: str
    tiene_estrella_dorada: bool
    fecha_desbloqueo: Optional[datetime] = None
    fecha_inicio: Optional[datetime] = None
    fecha_completada: Optional[datetime] = None
    ultima_practica: Optional[datetime] = None
    
    class Config:
        from_attributes = True