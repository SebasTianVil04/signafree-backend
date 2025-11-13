from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ..utilidades.base_datos import obtener_bd
from ..modelos.clase import Clase, TipoVideo
from ..modelos.leccion import Leccion
from ..modelos.usuario import Usuario
from ..esquemas.clase_schemas import ClaseCrear, ClaseActualizar, ClaseRespuesta
from ..utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/clases", tags=["Clases"])


@router.post("/", response_model=dict)
def crear_clase(
    clase: ClaseCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO CREACIÓN DE CLASE")
        print(f"{'='*60}")
        
        if not usuario_actual.es_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos para crear clases"
            )
        
        # Validar que la lección existe
        leccion = db.query(Leccion).filter(Leccion.id == clase.leccion_id).first()
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        # Validar orden duplicado
        clase_existente = db.query(Clase).filter(
            Clase.leccion_id == clase.leccion_id,
            Clase.orden == clase.orden
        ).first()
        
        if clase_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe una clase con el orden {clase.orden} en esta lección"
            )
        
        # IMPORTANTE: Los validadores de Pydantic ya validaron la seña
        # Convertir a diccionario (exclude_unset mantiene solo los valores enviados)
        clase_data = clase.dict(exclude_unset=False, exclude_none=False)
        
        print(f"\n📥 DATOS DESPUÉS DE PYDANTIC:")
        print(f"   requiere_practica: {clase_data.get('requiere_practica')}")
        print(f"   sena: {repr(clase_data.get('sena'))}")
        print(f"   sena type: {type(clase_data.get('sena'))}")
        print(f"   sena is None: {clase_data.get('sena') is None}")
        
        # Validar tipo de video
        if 'tipo_video' in clase_data and clase_data['tipo_video']:
            try:
                clase_data['tipo_video'] = TipoVideo(clase_data['tipo_video'])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tipo de video inválido. Debe ser: {', '.join([e.value for e in TipoVideo])}"
                )
        
        print(f"\n💾 CREANDO INSTANCIA DE CLASE...")
        print(f"   Datos a insertar: {clase_data}")
        
        # Crear la clase
        nueva_clase = Clase(**clase_data)
        
        print(f"\n📝 INSTANCIA CREADA:")
        print(f"   ID: {nueva_clase.id}")
        print(f"   Titulo: {nueva_clase.titulo}")
        print(f"   Seña: {repr(nueva_clase.sena)}")
        print(f"   Requiere práctica: {nueva_clase.requiere_practica}")
        
        db.add(nueva_clase)
        db.commit()
        db.refresh(nueva_clase)
        
        print(f"\n✅ CLASE CREADA EXITOSAMENTE")
        print(f"   ID en BD: {nueva_clase.id}")
        print(f"{'='*60}\n")
        
        return {
            "exito": True,
            "mensaje": "Clase creada exitosamente",
            "datos": ClaseRespuesta.from_orm(nueva_clase)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR EN CREACIÓN DE CLASE")
        print(f"{'='*60}")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        # Usar logging en lugar de traceback.print_exc() para evitar problemas en Windows
        import sys
        import traceback
        error_details = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"\nStacktrace:\n{error_details}")
        print(f"{'='*60}\n")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al crear clase: {str(e)}"
        )


@router.get("/leccion/{leccion_id}", response_model=dict)
def obtener_clases_por_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        print(f"🔍 Buscando lección con ID: {leccion_id}")
        
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            print(f"❌ Lección {leccion_id} no encontrada")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        print(f"✅ Lección encontrada: {leccion.titulo}")
        
        if usuario_actual.es_admin:
            clases = db.query(Clase).filter(
                Clase.leccion_id == leccion_id
            ).order_by(Clase.orden).all()
        else:
            clases = db.query(Clase).filter(
                Clase.leccion_id == leccion_id,
                Clase.activa == True
            ).order_by(Clase.orden).all()
        
        print(f"📚 Clases encontradas: {len(clases)}")
        
        clases_respuesta = []
        for clase in clases:
            try:
                clase_respuesta = ClaseRespuesta.from_orm(clase)
                clases_respuesta.append(clase_respuesta)
            except Exception as e:
                print(f"❌ Error convirtiendo clase {clase.id}: {e}")
                continue
        
        return {
            "exito": True,
            "mensaje": f"Se encontraron {len(clases_respuesta)} clases",
            "datos": clases_respuesta
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inesperado en obtener_clases_por_leccion: {e}")
        import sys
        import traceback
        error_details = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"Stacktrace:\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al obtener clases: {str(e)}"
        )


@router.get("/{clase_id}", response_model=dict)
def obtener_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        if not usuario_actual.es_admin and not clase.activa:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes acceso a esta clase"
            )
        
        return {
            "exito": True,
            "mensaje": "Clase obtenida exitosamente",
            "datos": ClaseRespuesta.from_orm(clase)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inesperado en obtener_clase: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al obtener clase: {str(e)}"
        )


@router.put("/{clase_id}", response_model=dict)
def actualizar_clase(
    clase_id: int,
    clase_data: ClaseActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        if not usuario_actual.es_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos para actualizar clases"
            )
        
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        # Validar orden duplicado
        if clase_data.orden is not None and clase_data.orden != clase.orden:
            orden_duplicado = db.query(Clase).filter(
                Clase.leccion_id == clase.leccion_id,
                Clase.orden == clase_data.orden,
                Clase.id != clase_id
            ).first()
            
            if orden_duplicado:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe una clase con el orden {clase_data.orden} en esta lección"
                )
        
        datos_actualizar = clase_data.dict(exclude_unset=True)
        
        # Validar tipo de video
        if 'tipo_video' in datos_actualizar and datos_actualizar['tipo_video']:
            try:
                datos_actualizar['tipo_video'] = TipoVideo(datos_actualizar['tipo_video'])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tipo de video inválido. Debe ser: {', '.join([e.value for e in TipoVideo])}"
                )
        
        # Validar seña si requiere práctica
        requiere_practica_nuevo = datos_actualizar.get('requiere_practica')
        sena_nueva = datos_actualizar.get('sena')
        
        if requiere_practica_nuevo is not None:
            if requiere_practica_nuevo:
                sena_final = sena_nueva if sena_nueva is not None else clase.sena
                if not sena_final or (isinstance(sena_final, str) and not sena_final.strip()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La seña es obligatoria cuando se requiere práctica"
                    )
        elif clase.requiere_practica:
            if sena_nueva is not None:
                if not sena_nueva or (isinstance(sena_nueva, str) and not sena_nueva.strip()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La seña es obligatoria cuando se requiere práctica"
                    )
        
        # Actualizar campos
        for campo, valor in datos_actualizar.items():
            setattr(clase, campo, valor)
        
        db.commit()
        db.refresh(clase)
        
        return {
            "exito": True,
            "mensaje": "Clase actualizada exitosamente",
            "datos": ClaseRespuesta.from_orm(clase)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inesperado en actualizar_clase: {e}")
        import sys
        import traceback
        error_details = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"Stacktrace:\n{error_details}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al actualizar clase: {str(e)}"
        )


@router.delete("/{clase_id}", response_model=dict)
def eliminar_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        if not usuario_actual.es_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos para eliminar clases"
            )
        
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        db.delete(clase)
        db.commit()
        
        return {
            "exito": True,
            "mensaje": "Clase eliminada exitosamente",
            "datos": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inesperado en eliminar_clase: {e}")
        import sys
        import traceback
        error_details = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"Stacktrace:\n{error_details}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al eliminar clase: {str(e)}"
        )