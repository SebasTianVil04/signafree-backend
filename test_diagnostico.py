import sys
import os

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test 1: Imports básicos
try:
    import fastapi
    print("FastAPI importado correctamente")
except Exception as e:
    print(f"Error importando FastAPI: {e}")

# Test 2: Pydantic
try:
    from pydantic import BaseModel
    print("Pydantic importado correctamente")
except Exception as e:
    print(f"Error importando Pydantic: {e}")

# Test 3: SQLAlchemy
try:
    import sqlalchemy
    print("SQLAlchemy importado correctamente")
except Exception as e:
    print(f"Error importando SQLAlchemy: {e}")

# Test 4: Nuestros módulos
try:
    from app.utilidades.configuracion import configuracion
    print("Configuración importada correctamente")
except Exception as e:
    print(f"Error importando configuración: {e}")

# Test 5: Base de datos
try:
    from app.utilidades.base_datos import Base
    print("Base de datos importada correctamente")
except Exception as e:
    print(f"Error importando base de datos: {e}")

# Test 6: Modelos
try:
    from app.modelos.usuario import Usuario
    print("Modelo Usuario importado correctamente")
except Exception as e:
    print(f"Error importando modelo Usuario: {e}")

# Test 7: Esquemas
try:
    from app.esquemas.usuario_schemas import UsuarioRegistro
    print("Esquemas importados correctamente")
except Exception as e:
    print(f"Error importando esquemas: {e}")

# Test 8: Main app
try:
    from app.main import app
    print("App principal importada correctamente")
except Exception as e:
    print(f"Error importando app principal: {e}")
    print(f"   Detalles: {str(e)}")

print("\n Diagnóstico completo.")