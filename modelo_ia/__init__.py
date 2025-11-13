"""Módulo de inteligencia artificial"""
# modelo_ia/datos_entrenamiento/__init__.py
"""
Módulo de datos de entrenamiento para el modelo de IA
Gestiona la carga, procesamiento y organización de datos para entrenar el modelo
"""

from pathlib import Path

# Directorio base de datos de entrenamiento
DATOS_DIR = Path(__file__).parent

# Categorías de señas disponibles
CATEGORIAS_SENAS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'hola', 'gracias', 'por_favor', 'adios', 'si', 'no'
]

__all__ = ['DATOS_DIR', 'CATEGORIAS_SENAS']