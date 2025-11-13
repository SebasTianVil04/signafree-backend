from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import hashlib
import bcrypt as bcrypt_lib  # Importar bcrypt directamente

from .configuracion import configuracion
from .base_datos import obtener_bd
from ..modelos.usuario import Usuario

# Configuración de contraseñas - Usar bcrypt directamente
contexto_password = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuración de tokens
security = HTTPBearer()

def obtener_hash_password(password: str) -> str:
    """
    Genera el hash de una contraseña usando bcrypt de forma directa
    """
    try:
        print(f"\n{'='*60}")
        print(f" GENERANDO HASH DE CONTRASEÑA")
        print(f"{'='*60}")
        print(f"1. Contraseña original: {len(password)} caracteres, {len(password.encode('utf-8'))} bytes")
        
        # SOLUCIÓN: Usar bcrypt directamente en lugar de passlib
        # Codificar la contraseña a bytes
        password_bytes = password.encode('utf-8')
        
        # Generar salt y hash con bcrypt
        salt = bcrypt_lib.gensalt()
        hashed = bcrypt_lib.hashpw(password_bytes, salt)
        
        # Convertir a string para almacenar
        bcrypt_hash = hashed.decode('utf-8')
        
        print(f"2. Hash Bcrypt generado: {len(bcrypt_hash)} caracteres")
        print(f"   Primeros 30 chars: {bcrypt_hash[:30]}...")
        print(f" Hash generado exitosamente")
        print(f"{'='*60}\n")
        
        return bcrypt_hash
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f" ERROR AL GENERAR HASH")
        print(f"{'='*60}")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print(f"Contraseña recibida: {len(password)} caracteres")
        print(f"{'='*60}\n")
        raise

def verificar_password(password_plano: str, password_hash: str) -> bool:
    """
    Verifica si una contraseña plana coincide con su hash
    """
    try:
        # Verificación directa con bcrypt
        password_bytes = password_plano.encode('utf-8')
        hash_bytes = password_hash.encode('utf-8')
        return bcrypt_lib.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        print(f" Error al verificar contraseña: {str(e)}")
        # Fallback a passlib si bcrypt directo falla
        try:
            return contexto_password.verify(password_plano, password_hash)
        except:
            return False

# El resto del código permanece igual...
def crear_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crear token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=configuracion.access_token_expire_minutes)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, configuracion.secret_key, algorithm=configuracion.algorithm)
    return encoded_jwt

def verificar_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar y decodificar token JWT"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            configuracion.secret_key, 
            algorithms=[configuracion.algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )

def obtener_usuario_actual(
    email: str = Depends(verificar_token),
    bd: Session = Depends(obtener_bd)
) -> Usuario:
    """Obtener usuario actual basado en el token"""
    usuario = bd.query(Usuario).filter(Usuario.email == email).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado"
        )
    if not usuario.activo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario inactivo"
        )
    return usuario

def verificar_admin(usuario_actual: Usuario = Depends(obtener_usuario_actual)):
    """Verificar si el usuario es administrador"""
    if not usuario_actual.es_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos de administrador"
        )
    return usuario_actual

def verificar_token_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    bd: Session = Depends(obtener_bd)
) -> Usuario:
    """Verificar token y permisos de administrador"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            configuracion.secret_key, 
            algorithms=[configuracion.algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        usuario = bd.query(Usuario).filter(Usuario.email == email).first()
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuario no encontrado"
            )
        
        if not usuario.activo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Usuario inactivo"
            )
        
        if not usuario.es_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos de administrador"
            )
        
        return usuario
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )