from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base


class Categoria(Base):
    __tablename__ = "categorias"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    
    tipo_id = Column(Integer, ForeignKey("tipos_categoria.id"), nullable=False, index=True)
    
    descripcion = Column(Text, nullable=True)
    icono = Column(String(255), nullable=True)
    color = Column(String(20), nullable=True)
    orden = Column(Integer, nullable=False)
    nivel_requerido = Column(Integer, default=1)
    activa = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relaciones existentes
    tipo_rel = relationship("TipoCategoria", back_populates="categorias")
    lecciones = relationship("Leccion", back_populates="categoria_rel", cascade="all, delete-orphan", order_by="Leccion.orden")
    
    # NUEVA: Relación con CategoriaDataset
    dataset_categoria = relationship("CategoriaDataset", back_populates="categoria_rel", uselist=False)
    
    def __repr__(self):
        return f"<Categoria(id={self.id}, nombre='{self.nombre}', tipo_id={self.tipo_id})>"
    
    @property
    def total_lecciones(self):
        return len(self.lecciones) if self.lecciones else 0
    
    @property
    def lecciones_activas(self):
        return [leccion for leccion in self.lecciones if leccion.activa]
    
    @property
    def total_lecciones_activas(self):
        return len(self.lecciones_activas)
