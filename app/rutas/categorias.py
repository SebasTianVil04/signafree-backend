from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from typing import Optional

from app.modelos.dataset import CategoriaDataset
from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual, verificar_admin
from ..modelos.usuario import Usuario
from ..modelos.categoria import Categoria
from ..modelos.tipo_categoria import TipoCategoria
from ..esquemas.categoria import CategoriaCrear, CategoriaActualizar
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista


router = APIRouter(prefix="/categorias", tags=["Categorías"])


@router.get("", response_model=RespuestaLista)
def listar_categorias(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    activas_solo: bool = Query(True),
    tipo_id: Optional[int] = Query(None)
):
    try:
        query = db.query(Categoria).options(
            joinedload(Categoria.lecciones),
            joinedload(Categoria.tipo_rel)
        )
        
        if activas_solo:
            query = query.filter(Categoria.activa == True)
        
        if tipo_id:
            query = query.filter(Categoria.tipo_id == tipo_id)
        
        categorias = query.order_by(Categoria.orden).all()
        
        categorias_data = []
        for categoria in categorias:
            categorias_data.append({
                "id": categoria.id,
                "nombre": categoria.nombre,
                "tipo_id": categoria.tipo_id,
                "tipo_valor": categoria.tipo_rel.valor if categoria.tipo_rel else None,
                "tipo_etiqueta": categoria.tipo_rel.etiqueta if categoria.tipo_rel else None,
                "descripcion": categoria.descripcion,
                "icono": categoria.icono,
                "color": categoria.color,
                "orden": categoria.orden,
                "nivel_requerido": categoria.nivel_requerido,
                "activa": categoria.activa,
                "total_lecciones": categoria.total_lecciones,
                "fecha_creacion": categoria.fecha_creacion,
                "fecha_actualizacion": categoria.fecha_actualizacion
            })
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(categorias_data)} categorías",
            datos=categorias_data,
            total=len(categorias_data),
            pagina=1,
            por_pagina=len(categorias_data)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar categorías: {str(e)}"
        )


@router.get("/{categoria_id}", response_model=RespuestaAPI)
def obtener_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        categoria = db.query(Categoria).options(
            joinedload(Categoria.lecciones),
            joinedload(Categoria.tipo_rel)
        ).filter(Categoria.id == categoria_id).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría con ID {categoria_id} no encontrada"
            )
        
        categoria_data = {
            "id": categoria.id,
            "nombre": categoria.nombre,
            "tipo_id": categoria.tipo_id,
            "tipo_valor": categoria.tipo_rel.valor if categoria.tipo_rel else None,
            "tipo_etiqueta": categoria.tipo_rel.etiqueta if categoria.tipo_rel else None,
            "descripcion": categoria.descripcion,
            "icono": categoria.icono,
            "color": categoria.color,
            "orden": categoria.orden,
            "nivel_requerido": categoria.nivel_requerido,
            "activa": categoria.activa,
            "total_lecciones": categoria.total_lecciones,
            "fecha_creacion": categoria.fecha_creacion,
            "fecha_actualizacion": categoria.fecha_actualizacion
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Categoría obtenida exitosamente",
            datos=categoria_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener categoría: {str(e)}"
        )


@router.post("", response_model=RespuestaAPI)
def crear_categoria(
    categoria: CategoriaCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual = Depends(verificar_admin)
):
    try:
        tipo = db.query(TipoCategoria).filter(
            TipoCategoria.id == categoria.tipo_id
        ).first()
        
        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de categoría {categoria.tipo_id} no existe"
            )
        
        existente = db.query(Categoria).filter(
            Categoria.nombre == categoria.nombre
        ).first()
        
        if existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe una categoría con el nombre '{categoria.nombre}'"
            )
        
        nueva_categoria = Categoria(
            nombre=categoria.nombre,
            tipo_id=categoria.tipo_id,
            descripcion=categoria.descripcion,
            icono=categoria.icono,
            color=categoria.color,
            orden=categoria.orden,
            nivel_requerido=categoria.nivel_requerido or 1,
            activa=True
        )
        
        db.add(nueva_categoria)
        db.flush()
        
        print(f"Categoría principal creada: ID={nueva_categoria.id}, nombre='{nueva_categoria.nombre}'")
        
        nueva_categoria_dataset = CategoriaDataset(
            nombre=nueva_categoria.nombre,
            descripcion=nueva_categoria.descripcion,
            categoria_id=nueva_categoria.id,
            activa=True,
            orden=nueva_categoria.orden
        )
        
        db.add(nueva_categoria_dataset)
        db.commit()
        db.refresh(nueva_categoria)
        db.refresh(nueva_categoria_dataset)
        
        print(f"Categoría dataset creada: ID={nueva_categoria_dataset.id}, categoria_id={nueva_categoria_dataset.categoria_id}")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Categoría creada exitosamente (incluye dataset)",
            datos={
                "id": nueva_categoria.id,
                "nombre": nueva_categoria.nombre,
                "tipo_id": nueva_categoria.tipo_id,
                "descripcion": nueva_categoria.descripcion,
                "icono": nueva_categoria.icono,
                "color": nueva_categoria.color,
                "orden": nueva_categoria.orden,
                "nivel_requerido": nueva_categoria.nivel_requerido,
                "activa": nueva_categoria.activa,
                "total_lecciones": 0,
                "dataset_id": nueva_categoria_dataset.id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        print(f"Error al crear categoría: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear categoría: {str(e)}"
        )


@router.put("/{categoria_id}", response_model=RespuestaAPI)
def actualizar_categoria(
    categoria_id: int,
    categoria_actualizar: CategoriaActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        categoria = db.query(Categoria).options(
            joinedload(Categoria.lecciones),
            joinedload(Categoria.tipo_rel)
        ).filter(Categoria.id == categoria_id).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría con ID {categoria_id} no encontrada"
            )
        
        if categoria_actualizar.nombre and categoria_actualizar.nombre != categoria.nombre:
            existe = db.query(Categoria).filter(
                Categoria.nombre == categoria_actualizar.nombre,
                Categoria.id != categoria_id
            ).first()
            if existe:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe una categoría con el nombre '{categoria_actualizar.nombre}'"
                )
        
        if categoria_actualizar.tipo_id and categoria_actualizar.tipo_id != categoria.tipo_id:
            tipo = db.query(TipoCategoria).filter(
                TipoCategoria.id == categoria_actualizar.tipo_id
            ).first()
            
            if not tipo:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El tipo de categoría con ID {categoria_actualizar.tipo_id} no existe"
                )
            
            if not tipo.activo:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El tipo '{tipo.etiqueta}' está inactivo"
                )
        
        datos_actualizacion = categoria_actualizar.dict(exclude_unset=True)
        for campo, valor in datos_actualizacion.items():
            setattr(categoria, campo, valor)
        
        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()
        
        if categoria_dataset:
            if categoria_actualizar.nombre is not None:
                categoria_dataset.nombre = categoria_actualizar.nombre
            if categoria_actualizar.descripcion is not None:
                categoria_dataset.descripcion = categoria_actualizar.descripcion
            if categoria_actualizar.activa is not None:
                categoria_dataset.activa = categoria_actualizar.activa
            if categoria_actualizar.orden is not None:
                categoria_dataset.orden = categoria_actualizar.orden
        
        db.commit()
        db.refresh(categoria)
        db.refresh(categoria, ["tipo_rel"])
        
        categoria_data = {
            "id": categoria.id,
            "nombre": categoria.nombre,
            "tipo_id": categoria.tipo_id,
            "tipo_valor": categoria.tipo_rel.valor if categoria.tipo_rel else None,
            "tipo_etiqueta": categoria.tipo_rel.etiqueta if categoria.tipo_rel else None,
            "descripcion": categoria.descripcion,
            "icono": categoria.icono,
            "color": categoria.color,
            "orden": categoria.orden,
            "nivel_requerido": categoria.nivel_requerido,
            "activa": categoria.activa,
            "total_lecciones": categoria.total_lecciones,
            "fecha_creacion": categoria.fecha_creacion,
            "fecha_actualizacion": categoria.fecha_actualizacion
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Categoría '{categoria.nombre}' actualizada exitosamente",
            datos=categoria_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        print(f"Error al actualizar categoría: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar categoría: {str(e)}"
        )


@router.delete("/{categoria_id}", response_model=RespuestaAPI)
def eliminar_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin),
    forzar: bool = False
):
    try:
        categoria = db.query(Categoria).options(
            joinedload(Categoria.lecciones),
            joinedload(Categoria.tipo_rel)
        ).filter(Categoria.id == categoria_id).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría con ID {categoria_id} no encontrada"
            )
        
        total_lecciones = len(categoria.lecciones)
        
        if total_lecciones > 0 and not forzar:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"La categoría tiene {total_lecciones} lección(es) asociada(s). Usa 'forzar=true' para desactivarla."
            )
        
        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()
        
        if categoria_dataset:
            db.delete(categoria_dataset)
            print(f"Categoría dataset eliminada: ID={categoria_dataset.id}")
        
        if forzar and total_lecciones > 0:
            categoria.activa = False
            db.commit()
            mensaje = f"Categoría desactivada (tenía {total_lecciones} lecciones asociadas)"
        else:
            db.delete(categoria)
            db.commit()
            mensaje = "Categoría eliminada exitosamente"
        
        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        print(f"Error al eliminar categoría: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar categoría: {str(e)}"
        )



@router.patch("/{categoria_id}/toggle", response_model=RespuestaAPI)
def cambiar_estado_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        categoria = db.query(Categoria).options(
            joinedload(Categoria.lecciones),
            joinedload(Categoria.tipo_rel)
        ).filter(Categoria.id == categoria_id).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría con ID {categoria_id} no encontrada"
            )
        
        categoria.activa = not categoria.activa
        
        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()
        
        if categoria_dataset:
            categoria_dataset.activa = categoria.activa
            print(f"Categoría dataset actualizada: ID={categoria_dataset.id}, activa={categoria.activa}")
        
        db.commit()
        db.refresh(categoria)
        db.refresh(categoria, ["tipo_rel"])
        
        categoria_data = {
            "id": categoria.id,
            "nombre": categoria.nombre,
            "tipo_id": categoria.tipo_id,
            "tipo_valor": categoria.tipo_rel.valor if categoria.tipo_rel else None,
            "tipo_etiqueta": categoria.tipo_rel.etiqueta if categoria.tipo_rel else None,
            "descripcion": categoria.descripcion,
            "icono": categoria.icono,
            "color": categoria.color,
            "orden": categoria.orden,
            "nivel_requerido": categoria.nivel_requerido,
            "activa": categoria.activa,
            "total_lecciones": categoria.total_lecciones,
            "fecha_creacion": categoria.fecha_creacion,
            "fecha_actualizacion": categoria.fecha_actualizacion
        }
        
        estado = "activada" if categoria.activa else "desactivada"
        return RespuestaAPI(
            exito=True,
            mensaje=f"Categoría '{categoria.nombre}' {estado} exitosamente",
            datos=categoria_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        print(f"Error al cambiar estado de categoría: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cambiar estado de categoría: {str(e)}"
        )


@router.post("/sincronizar-dataset", response_model=RespuestaAPI)
def sincronizar_categorias_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual = Depends(verificar_admin)
):
    try:
        categorias = db.query(Categoria).all()
        categorias_creadas = 0
        categorias_existentes = 0
        
        for categoria in categorias:
            dataset_existe = db.query(CategoriaDataset).filter(
                CategoriaDataset.categoria_id == categoria.id
            ).first()
            
            if not dataset_existe:
                nueva_categoria_dataset = CategoriaDataset(
                    nombre=categoria.nombre,
                    descripcion=categoria.descripcion,
                    categoria_id=categoria.id,
                    activa=categoria.activa,
                    orden=categoria.orden
                )
                db.add(nueva_categoria_dataset)
                categorias_creadas += 1
                print(f"Categoría dataset creada para: {categoria.nombre}")
            else:
                categorias_existentes += 1
        
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Sincronización completada: {categorias_creadas} creadas, {categorias_existentes} ya existían",
            datos={
                "categorias_creadas": categorias_creadas,
                "categorias_existentes": categorias_existentes,
                "total_categorias": len(categorias)
            }
        )
        
    except Exception as e:
        db.rollback()
        import traceback
        print(f"Error al sincronizar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al sincronizar: {str(e)}"
        )
