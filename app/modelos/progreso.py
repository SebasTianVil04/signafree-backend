# app/modelos/progreso.py
from sqlalchemy import Column, Integer, Boolean, DateTime, Float, ForeignKey, JSON, String
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

# app/modelos/progreso.py - Campos faltantes
class ProgresoClase(Base):
    __tablename__ = "progresos_clases"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(Integer, ForeignKey("clases.id"), nullable=False)
    
    # 🔥 AÑADIR CAMPOS FALTANTES
    tiempo_total_practica = Column(Integer, default=0)  # Faltaba
    fecha_primera_vista = Column(DateTime(timezone=True), nullable=True)
    fecha_completada = Column(DateTime(timezone=True), nullable=True)
    ultima_practica = Column(DateTime(timezone=True), nullable=True)
    
    # Campos existentes
    vista = Column(Boolean, default=False)
    completada = Column(Boolean, default=False)
    intentos_realizados = Column(Integer, default=0)
    mejor_precision = Column(Float, default=0.0)
    ultima_precision = Column(Float, default=0.0)  # 🔥 NUEVO: Última precisión
    
    # Timestamps
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relaciones
    usuario = relationship("Usuario", back_populates="progresos_clases")
    clase = relationship("Clase", back_populates="progresos_clase")

class ProgresoLeccion(Base):
    """Progreso general por lección"""
    __tablename__ = "progresos_lecciones"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relaciones
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)
    
    # Estado del progreso
    desbloqueada = Column(Boolean, default=False)
    iniciada = Column(Boolean, default=False)
    completada = Column(Boolean, default=False)
    
    # Estadísticas generales
    total_clases = Column(Integer, default=0)
    clases_completadas = Column(Integer, default=0)
    mejor_precision = Column(Float, default=0.0)
    total_intentos = Column(Integer, default=0)
    total_puntos = Column(Integer, default=0)
    
    # Estrellas ganadas (1-3 según desempeño)
    estrellas = Column(Integer, default=0)
    
    # Fechas
    fecha_desbloqueo = Column(DateTime(timezone=True), nullable=True)
    fecha_inicio = Column(DateTime(timezone=True), nullable=True)
    fecha_completada = Column(DateTime(timezone=True), nullable=True)
    ultima_practica = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relaciones
    usuario = relationship("Usuario", back_populates="progresos_lecciones")
    leccion = relationship("Leccion", back_populates="progresos")
    
    @property
    def porcentaje_completado(self):
        """Retorna el porcentaje de clases completadas"""
        if self.total_clases == 0:
            return 0
        return (self.clases_completadas / self.total_clases) * 100
    
    @property
    def porcentaje_precision(self):
        """Retorna la mejor precisión como porcentaje"""
        return f"{self.mejor_precision * 100:.1f}%"
    
    @property
    def tiene_estrella_dorada(self):
        """Verifica si merece estrella dorada (100% precisión)"""
        return self.estrellas == 3
    
    def __repr__(self):
        return f"<ProgresoLeccion(usuario_id={self.usuario_id}, leccion_id={self.leccion_id}, completada={self.completada})>"