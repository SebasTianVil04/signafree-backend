from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import hashlib

from .configuracion import configuracion
from .base_datos import obtener_bd
from ..modelos.usuario import Usuario

# Configuración de contraseñas
contexto_password = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuración de tokens
security = HTTPBearer()

# Constante para límite de bcrypt
MAX_PASSWORD_LENGTH = 72


def obtener_hash_password(password: str) -> str:
    """
    Genera el hash de una contraseña usando bcrypt
    """
    try:
        print(f"\n{'='*60}")
        print(f" GENERANDO HASH DE CONTRASEÑA")
        print(f"{'='*60}")
        print(f"1. Contraseña original: {len(password)} caracteres, {len(password.encode('utf-8'))} bytes")
        
        # Verificar si la contraseña es demasiado larga para bcrypt
        if len(password.encode('utf-8')) > MAX_PASSWORD_LENGTH:
            print("ADVERTENCIA: Contraseña demasiado larga, truncando...")
            password = password[:MAX_PASSWORD_LENGTH]
        
        # Hash directo con bcrypt (más seguro)
        bcrypt_hash = contexto_password.hash(password)
        print(f"2. Hash Bcrypt: {len(bcrypt_hash)} caracteres, {len(bcrypt_hash.encode('utf-8'))} bytes")
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
        return contexto_password.verify(password_plano, password_hash)
    except Exception as e:
        print(f" Error al verificar contraseña: {str(e)}")
        return False


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