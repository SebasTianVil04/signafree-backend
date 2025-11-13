from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Any, Dict, List
from datetime import datetime

from ..utilidades.base_datos import obtener_bd, SessionLocal
from ..modelos.progreso import ProgresoClase, ProgresoLeccion
from ..modelos.clase import Clase
from ..modelos.leccion import Leccion
from ..modelos.usuario import Usuario
from ..esquemas.progreso_schemas import (
    ProgresoClaseRespuesta, 
    ProgresoLeccionRespuesta
)
from ..utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/progreso", tags=["Progreso"])


@router.get("/clase/{clase_id}", response_model=ProgresoClaseRespuesta)
def obtener_progreso_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        print(f"📊 Obteniendo progreso de clase {clase_id} para usuario {usuario_actual.id}")
        
        progreso = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.clase_id == clase_id
        ).first()
        
        if not progreso:
            clase = db.query(Clase).filter(Clase.id == clase_id).first()
            if not clase:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Clase no encontrada"
                )
            
            progreso = ProgresoClase(
                usuario_id=usuario_actual.id,
                clase_id=clase_id,
                vista=False,
                completada=False,
                intentos_realizados=0,
                mejor_precision=0.0,
                tiempo_total_practica=0
            )
            db.add(progreso)
            db.commit()
            db.refresh(progreso)
            print(f"✅ Progreso creado")
        else:
            print(f"✅ Progreso encontrado: {progreso.intentos_realizados} intentos")
        
        return progreso
    except Exception as e:
        print(f"❌ Error obteniendo progreso: {str(e)}")
        db.rollback()
        raise


@router.post("/clase/{clase_id}/marcar-vista")
def marcar_clase_vista(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        progreso = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.clase_id == clase_id
        ).first()
        
        if not progreso:
            progreso = ProgresoClase(
                usuario_id=usuario_actual.id,
                clase_id=clase_id,
                vista=False,
                completada=False,
                intentos_realizados=0,
                mejor_precision=0.0,
                tiempo_total_practica=0
            )
            db.add(progreso)
        
        if not progreso.vista:
            progreso.vista = True
            progreso.fecha_primera_vista = datetime.now()
        
        db.commit()
        db.refresh(progreso)
        
        return {"mensaje": "Clase marcada como vista", "progreso": progreso}
    except Exception as e:
        print(f"❌ Error marcando vista: {str(e)}")
        db.rollback()
        raise


@router.post("/clases/{clase_id}/practica")
async def guardar_resultado_practica(
    clase_id: int,
    resultado: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        precision = float(resultado.get("precision", 0))
        tiempo_practica = int(resultado.get("tiempo_practica", 0))
        sena_reconocida = str(resultado.get("sena_reconocida", "")).upper().strip()
        
        print(f"📊 Guardando práctica - Usuario: {usuario_actual.id}, Clase: {clase_id}")
        
        clase = bd.query(Clase).filter(Clase.id == clase_id).first()
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        leccion_id = clase.leccion_id
        
        intentos_minimos = clase.intentos_minimos or 3
        precision_minima = clase.precision_minima or 0.7
        
        sena_esperada = clase.sena.upper().strip() if clase.sena else ""
        
        if not sena_esperada:
            leccion = bd.query(Leccion).filter(Leccion.id == leccion_id).first()
            if leccion and leccion.sena:
                sena_esperada = leccion.sena.upper().strip()
        
        print(f"   Seña esperada: '{sena_esperada}'")
        print(f"   Seña reconocida: '{sena_reconocida}'")
        print(f"   Requisitos: {intentos_minimos} intentos, {precision_minima*100}% precisión")
        
        progreso = bd.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.clase_id == clase_id
        ).first()
        
        if not progreso:
            print("   ➕ Creando nuevo progreso...")
            progreso = ProgresoClase(
                usuario_id=usuario_actual.id,
                clase_id=clase_id,
                vista=True,
                completada=False,
                intentos_realizados=0,
                mejor_precision=0.0,
                tiempo_total_practica=0,
                ultima_precision=0.0
            )
            bd.add(progreso)
            bd.flush()
        
        intentos_anteriores = progreso.intentos_realizados
        precision_anterior = progreso.mejor_precision
        
        progreso.intentos_realizados += 1
        progreso.tiempo_total_practica += tiempo_practica
        progreso.ultima_practica = datetime.now()
        progreso.ultima_precision = precision
        
        if precision > progreso.mejor_precision:
            progreso.mejor_precision = precision
        
        sena_correcta = (sena_reconocida == sena_esperada)
        cumple_intentos = progreso.intentos_realizados >= intentos_minimos
        cumple_precision = progreso.mejor_precision >= precision_minima
        
        recien_completada = False
        
        if sena_correcta and cumple_intentos and cumple_precision and not progreso.completada:
            progreso.completada = True
            progreso.fecha_completada = datetime.now()
            recien_completada = True
            print(f"   🎉 ¡CLASE COMPLETADA!")
            print(f"      ✅ Seña correcta: {sena_reconocida} == {sena_esperada}")
            print(f"      ✅ Intentos suficientes: {progreso.intentos_realizados} >= {intentos_minimos}")
            print(f"      ✅ Precisión suficiente: {progreso.mejor_precision:.2f} >= {precision_minima}")
        else:
            if progreso.completada:
                print(f"   ℹ️  Clase ya estaba completada")
            else:
                print(f"   ⏸️  Clase NO completada:")
                if not sena_correcta:
                    print(f"      ❌ Seña incorrecta: '{sena_reconocida}' != '{sena_esperada}'")
                else:
                    print(f"      ✅ Seña correcta: '{sena_reconocida}' == '{sena_esperada}'")
                
                if not cumple_intentos:
                    print(f"      ❌ Intentos insuficientes: {progreso.intentos_realizados} < {intentos_minimos}")
                else:
                    print(f"      ✅ Intentos suficientes: {progreso.intentos_realizados} >= {intentos_minimos}")
                
                if not cumple_precision:
                    print(f"      ❌ Precisión insuficiente: {progreso.mejor_precision:.2f} < {precision_minima}")
                else:
                    print(f"      ✅ Precisión suficiente: {progreso.mejor_precision:.2f} >= {precision_minima}")
        
        print(f"   📈 Progreso actualizado: {intentos_anteriores} → {progreso.intentos_realizados} intentos")
        print(f"   ⭐ Mejor precisión: {precision_anterior:.2f} → {progreso.mejor_precision:.2f}")
        
        bd.commit()
        bd.refresh(progreso)
        
        if recien_completada and leccion_id:
            try:
                await actualizar_progreso_leccion_background(usuario_actual.id, leccion_id, recien_completada)
            except Exception as e:
                print(f"⚠️  Error en actualización de lección: {str(e)}")
        
        respuesta = {
            "exito": True,
            "mensaje": "Resultado guardado correctamente",
            "progreso": {
                "completada": progreso.completada,
                "intentos_realizados": progreso.intentos_realizados,
                "mejor_precision": float(progreso.mejor_precision),
                "ultima_precision": float(progreso.ultima_precision),
                "tiempo_total_practica": progreso.tiempo_total_practica,
                "ultima_practica": progreso.ultima_practica.isoformat() if progreso.ultima_practica else None
            },
            "clase_completada": recien_completada,
            "requisitos": {
                "intentos_minimos": intentos_minimos,
                "precision_minima": precision_minima,
                "cumple_intentos": cumple_intentos,
                "cumple_precision": cumple_precision,
                "sena_correcta": sena_correcta,
                "sena_esperada": sena_esperada,
                "sena_reconocida": sena_reconocida
            }
        }
        
        print("✅ Práctica guardada exitosamente")
        return respuesta
        
    except HTTPException:
        bd.rollback()
        raise
    except Exception as e:
        print(f"❌ ERROR guardando práctica: {str(e)}")
        import traceback
        print(traceback.format_exc())
        bd.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar resultado: {str(e)}"
        )

async def actualizar_progreso_leccion_background(usuario_id: int, leccion_id: int, clase_completada: bool):
    db = SessionLocal()
    try:
        print(f"🔄 Actualizando progreso de lección {leccion_id} para usuario {usuario_id}")
        
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            print(f"❌ Lección {leccion_id} no encontrada")
            return
        
        progreso_leccion = db.query(ProgresoLeccion).filter(
            ProgresoLeccion.usuario_id == usuario_id,
            ProgresoLeccion.leccion_id == leccion_id
        ).first()
        
        if not progreso_leccion:
            progreso_leccion = ProgresoLeccion(
                usuario_id=usuario_id,
                leccion_id=leccion_id,
                total_clases=len(leccion.clases) if leccion.clases else 0,
                desbloqueada=True
            )
            db.add(progreso_leccion)
            db.flush()
        
        clases_completadas = db.query(ProgresoClase).join(Clase).filter(
            ProgresoClase.usuario_id == usuario_id,
            Clase.leccion_id == leccion_id,
            ProgresoClase.completada == True
        ).count()
        
        progreso_leccion.clases_completadas = clases_completadas
        
        precision_promedio = db.query(func.avg(ProgresoClase.mejor_precision)).join(Clase).filter(
            ProgresoClase.usuario_id == usuario_id,
            Clase.leccion_id == leccion_id,
            ProgresoClase.mejor_precision > 0
        ).scalar()
        
        if precision_promedio:
            progreso_leccion.mejor_precision = float(precision_promedio)
        
        if (progreso_leccion.clases_completadas >= progreso_leccion.total_clases and 
            progreso_leccion.total_clases > 0 and 
            not progreso_leccion.completada):
            
            progreso_leccion.completada = True
            progreso_leccion.fecha_completada = datetime.now()
            
            if progreso_leccion.mejor_precision >= 0.95:
                progreso_leccion.estrellas = 3
            elif progreso_leccion.mejor_precision >= 0.85:
                progreso_leccion.estrellas = 2
            else:
                progreso_leccion.estrellas = 1
            
            print(f"🎉 ¡LECCIÓN {leccion_id} COMPLETADA! Estrellas: {progreso_leccion.estrellas}")
            
            desbloquear_siguiente_leccion_sync(db, usuario_id, leccion_id)
        
        db.commit()
        print(f"✅ Progreso de lección {leccion_id} actualizado: {clases_completadas}/{progreso_leccion.total_clases} clases")
        
    except Exception as e:
        print(f"❌ Error en actualización de lección: {str(e)}")
        import traceback
        print(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


def desbloquear_siguiente_leccion_sync(db: Session, usuario_id: int, leccion_actual_id: int):
    try:
        leccion_actual = db.query(Leccion).filter(Leccion.id == leccion_actual_id).first()
        
        if not leccion_actual:
            print(f"⚠️  Lección actual {leccion_actual_id} no encontrada")
            return
        
        print(f"📍 Lección actual: ID={leccion_actual.id}, Categoría={leccion_actual.categoria}, Orden={leccion_actual.orden}")
        
        siguiente_leccion = db.query(Leccion).filter(
            Leccion.categoria == leccion_actual.categoria,
            Leccion.orden == leccion_actual.orden + 1,
            Leccion.activa == True
        ).first()
        
        if siguiente_leccion:
            print(f"🔍 Siguiente lección encontrada: ID={siguiente_leccion.id}, Título={siguiente_leccion.titulo}")
            
            progreso_siguiente = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_id,
                ProgresoLeccion.leccion_id == siguiente_leccion.id
            ).first()
            
            if not progreso_siguiente:
                progreso_siguiente = ProgresoLeccion(
                    usuario_id=usuario_id,
                    leccion_id=siguiente_leccion.id,
                    total_clases=len(siguiente_leccion.clases) if siguiente_leccion.clases else 0,
                    desbloqueada=True,
                    fecha_desbloqueo=datetime.now()
                )
                db.add(progreso_siguiente)
                db.flush()
                print(f"🎉 ✅ Siguiente lección {siguiente_leccion.id} DESBLOQUEADA (nueva)")
            else:
                if not progreso_siguiente.desbloqueada:
                    progreso_siguiente.desbloqueada = True
                    progreso_siguiente.fecha_desbloqueo = datetime.now()
                    print(f"🔓 ✅ Siguiente lección {siguiente_leccion.id} RE-DESBLOQUEADA")
                else:
                    print(f"ℹ️  Lección {siguiente_leccion.id} ya estaba desbloqueada")
                    
            db.commit()
        else:
            print(f"ℹ️  No hay siguiente lección en categoría {leccion_actual.categoria}, orden {leccion_actual.orden + 1}")
            
    except Exception as e:
        print(f"❌ Error en desbloquear_siguiente_leccion_sync: {str(e)}")
        import traceback
        print(traceback.format_exc())
        db.rollback()


@router.get("/leccion/{leccion_id}", response_model=ProgresoLeccionRespuesta)
def obtener_progreso_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        progreso = db.query(ProgresoLeccion).filter(
            ProgresoLeccion.usuario_id == usuario_actual.id,
            ProgresoLeccion.leccion_id == leccion_id
        ).first()
        
        if not progreso:
            leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
            if not leccion:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Lección no encontrada"
                )
            
            progreso = ProgresoLeccion(
                usuario_id=usuario_actual.id,
                leccion_id=leccion_id,
                total_clases=len(leccion.clases) if leccion.clases else 0,
                desbloqueada=True if leccion.orden == 1 else False
            )
            db.add(progreso)
            db.commit()
            db.refresh(progreso)
        
        return progreso
    except Exception as e:
        print(f"❌ Error obteniendo progreso lección: {str(e)}")
        db.rollback()
        raise


@router.get("/usuario/resumen")
def obtener_resumen_progreso_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Obtener resumen completo del progreso del usuario.
    ✅ CORREGIDO: Actualiza el conteo de clases completadas en tiempo real
    """
    try:
        print(f"📊 Generando resumen para usuario {usuario_actual.id}")
        
        # Obtener TODAS las lecciones activas, no solo las que tienen progreso
        lecciones = db.query(Leccion).filter(Leccion.activa == True).all()
        
        progresos_data = []
        total_puntos = 0
        total_estrellas = 0
        lecciones_completadas = 0
        
        for leccion in lecciones:
            # Buscar o crear progreso de lección
            progreso = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.leccion_id == leccion.id
            ).first()
            
            # ✅ CORRECCIÓN: Calcular clases completadas en tiempo real
            clases_completadas_count = db.query(ProgresoClase).join(Clase).filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                Clase.leccion_id == leccion.id,
                ProgresoClase.completada == True,
                Clase.activa == True
            ).count()
            
            total_clases = db.query(Clase).filter(
                Clase.leccion_id == leccion.id,
                Clase.activa == True
            ).count()
            
            print(f"   Lección {leccion.id}: {clases_completadas_count}/{total_clases} clases")
            
            # Si no existe progreso, crearlo
            if not progreso:
                progreso = ProgresoLeccion(
                    usuario_id=usuario_actual.id,
                    leccion_id=leccion.id,
                    total_clases=total_clases,
                    clases_completadas=clases_completadas_count,
                    desbloqueada=True if leccion.orden == 1 else False,
                    iniciada=clases_completadas_count > 0,
                    completada=False,
                    mejor_precision=0.0,
                    total_intentos=0,
                    total_puntos=0,
                    estrellas=0
                )
                db.add(progreso)
                db.flush()
            else:
                # ✅ CORRECCIÓN: Actualizar valores en tiempo real
                progreso.clases_completadas = clases_completadas_count
                progreso.total_clases = total_clases
                
                # Actualizar si está iniciada
                if clases_completadas_count > 0 and not progreso.iniciada:
                    progreso.iniciada = True
                
                # Verificar si se completó
                if clases_completadas_count >= total_clases and total_clases > 0 and not progreso.completada:
                    progreso.completada = True
                    progreso.fecha_completada = datetime.now()
                    
                    # Calcular estrellas basado en precisión
                    precision_promedio = db.query(func.avg(ProgresoClase.mejor_precision)).join(Clase).filter(
                        ProgresoClase.usuario_id == usuario_actual.id,
                        Clase.leccion_id == leccion.id,
                        ProgresoClase.mejor_precision > 0
                    ).scalar()
                    
                    if precision_promedio:
                        progreso.mejor_precision = float(precision_promedio)
                        
                        if progreso.mejor_precision >= 0.95:
                            progreso.estrellas = 3
                        elif progreso.mejor_precision >= 0.85:
                            progreso.estrellas = 2
                        else:
                            progreso.estrellas = 1
            
            # Calcular porcentaje
            porcentaje_completado = (clases_completadas_count / total_clases * 100) if total_clases > 0 else 0
            
            # Agregar a la lista
            progresos_data.append({
                "id": progreso.id if progreso.id else None,
                "leccion_id": leccion.id,
                "usuario_id": usuario_actual.id,
                "desbloqueada": progreso.desbloqueada,
                "iniciada": progreso.iniciada,
                "completada": progreso.completada,
                "total_clases": total_clases,
                "clases_completadas": clases_completadas_count,  # ✅ Valor actualizado
                "mejor_precision": float(progreso.mejor_precision or 0),
                "total_intentos": progreso.total_intentos or 0,
                "total_puntos": progreso.total_puntos or 0,
                "estrellas": progreso.estrellas or 0,
                "porcentaje_completado": round(porcentaje_completado, 1),
                "porcentaje_precision": f"{progreso.mejor_precision * 100:.1f}%" if progreso.mejor_precision else "0%",
                "tiene_estrella_dorada": (progreso.mejor_precision or 0) >= 0.95,
                "fecha_desbloqueo": progreso.fecha_desbloqueo,
                "fecha_completada": progreso.fecha_completada
            })
            
            # Sumar totales
            if progreso.completada:
                lecciones_completadas += 1
            total_puntos += progreso.total_puntos or 0
            total_estrellas += progreso.estrellas or 0
        
        # Guardar cambios
        db.commit()
        
        total_lecciones = len(lecciones)
        porcentaje_total = (lecciones_completadas / total_lecciones * 100) if total_lecciones > 0 else 0
        
        resultado = {
            "total_lecciones": total_lecciones,
            "lecciones_completadas": lecciones_completadas,
            "porcentaje_completado": round(porcentaje_total, 1),
            "total_puntos": total_puntos,
            "total_estrellas": total_estrellas,
            "progresos": progresos_data
        }
        
        print(f"✅ Resumen generado: {lecciones_completadas}/{total_lecciones} lecciones")
        return resultado
        
    except Exception as e:
        print(f"❌ Error obteniendo resumen: {str(e)}")
        import traceback
        print(traceback.format_exc())
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener resumen: {str(e)}"
        )