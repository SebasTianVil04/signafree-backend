import logging
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuraciones específicas por tipo de seña
CONFIG_SENAS = {
    'ESTATICA': {
        'num_frames_recomendado': 8,
        'fps_muestreo': 10,
        'enfoque': 'centro_secuencia',
        'requiere_estabilidad': True,
        'umbral_confianza': 0.65,
        'procesamiento': 'frames_centrales',
        'augmentation_brightness': (0.7, 1.3),
        'augmentation_rotation': 15
    },
    'DINAMICA': {
        'num_frames_recomendado': 16,
        'fps_muestreo': 15,
        'enfoque': 'secuencia_completa',
        'requiere_estabilidad': False,
        'umbral_confianza': 0.55,
        'procesamiento': 'distribucion_uniforme',
        'augmentation_brightness': (0.6, 1.4),
        'augmentation_rotation': 10
    }
}

# Señas estáticas conocidas
SENAS_ESTATICAS = {
    # Alfabeto
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # Números
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # Señas estáticas comunes
    'hola', 'gracias', 'si', 'no', 'ayuda', 'casa', 'agua', 'comida', 'baño',
    'familia', 'amigo', 'trabajo', 'escuela', 'doctor', 'hospital', 'dinero'
}

# Patrones para señas dinámicas
PATRONES_DINAMICOS = [
    r'.*familia.*', r'.*tiempo.*', r'.*trabajo.*', r'.*escuela.*',
    r'.*amigo.*', r'.*favorito.*', r'.*gustar.*', r'.*querer.*',
    r'.*necesitar.*', r'.*entender.*', r'.*explicar.*', r'.*preguntar.*',
    r'.*contar.*', r'.*enseñar.*', r'.*aprender.*', r'.*trabajar.*'
]

# Mapeo de confusiones comunes entre señas
CONFUSIONES_COMUNES = {
    'A': ['B', '5', 'S', '1'],
    'B': ['A', '1', 'D', 'P'],
    'C': ['O', 'D', '3', 'G'],
    'D': ['B', 'C', 'P', 'Q'],
    'E': ['3', 'M', 'S', 'A'],
    'F': ['9', 'P', 'R'],
    'G': ['C', 'Q', '6'],
    'H': ['8', 'N', 'M'],
    'I': ['1', 'L', '7'],
    'J': ['Z', '1', '7'],
    'K': ['X', 'R', '4'],
    'L': ['1', 'I', '7'],
    'M': ['N', '3', 'W'],
    'N': ['M', 'Z', '2'],
    'O': ['0', 'C', 'Q'],
    'P': ['B', 'F', 'R'],
    'Q': ['O', 'G', '2'],
    'R': ['K', 'P', 'F'],
    'S': ['5', 'A', '8'],
    'T': ['7', 'J', '1'],
    'U': ['V', 'W', '2'],
    'V': ['U', 'W', '2'],
    'W': ['M', 'U', 'V'],
    'X': ['K', 'Y', '4'],
    'Y': ['X', 'V', '4'],
    'Z': ['2', 'N', '7'],
    'HOLA': ['AYUDA', 'GRACIAS', 'POR FAVOR'],
    'GRACIAS': ['HOLA', 'POR FAVOR', 'ADIOS'],
    'AYUDA': ['HOLA', 'POR FAVOR', 'NECESITO']
}

# Configuración global de tipos de señas
tipo_sena_config = {
    'ESTATICA': {
        'frames_entrenamiento': 8,
        'fps': 10,
        'augmentation': {
            'brightness_range': (0.7, 1.3),
            'rotation_range': 15,
            'flip_prob': 0.5
        }
    },
    'DINAMICA': {
        'frames_entrenamiento': 16,
        'fps': 15,
        'augmentation': {
            'brightness_range': (0.6, 1.4),
            'rotation_range': 10,
            'flip_prob': 0.5,
            'noise_level': 10,
            'frame_drop_prob': 0.3
        }
    }
}

def detectar_tipo_sena(nombre_sena: str) -> str:
    """
    Detecta si una seña es estática o dinámica basado en su nombre y patrones.
    
    Args:
        nombre_sena: Nombre de la seña a analizar
        
    Returns:
        'ESTATICA' o 'DINAMICA'
    """
    if not nombre_sena:
        return 'DINAMICA'
    
    nombre_limpio = nombre_sena.lower().strip()
    
    # Verificar si es una seña estática conocida
    if nombre_limpio in SENAS_ESTATICAS:
        return 'ESTATICA'
    
    # Verificar si es una sola letra
    if len(nombre_limpio) == 1 and nombre_limpio.isalpha():
        return 'ESTATICA'
    
    # Verificar si es un número
    if nombre_limpio.isdigit():
        return 'ESTATICA'
    
    # Verificar patrones dinámicos
    for patron in PATRONES_DINAMICOS:
        if re.match(patron, nombre_limpio):
            return 'DINAMICA'
    
    # Por defecto, considerar estática para señas simples
    if len(nombre_limpio.split()) == 1 and len(nombre_limpio) <= 8:
        return 'ESTATICA'
    
    return 'DINAMICA'

def obtener_config_sena(nombre_sena: str) -> Dict[str, Any]:
    """
    Obtiene la configuración específica para un tipo de seña.
    
    Args:
        nombre_sena: Nombre de la seña
        
    Returns:
        Dict con configuración completa
    """
    tipo_sena = detectar_tipo_sena(nombre_sena)
    config = CONFIG_SENAS[tipo_sena].copy()
    config['tipo_detectado'] = tipo_sena
    config['nombre_sena'] = nombre_sena
    
    logger.info(f"Configuración para '{nombre_sena}': {tipo_sena}")
    
    return config

def obtener_confusiones_comunes(sena: str) -> List[str]:
    """
    Obtiene las señas con las que comúnmente se confunde una seña.
    
    Args:
        sena: Seña a analizar
        
    Returns:
        Lista de señas comúnmente confundidas
    """
    sena_upper = sena.upper()
    return CONFUSIONES_COMUNES.get(sena_upper, [])

def es_confusion_comun(sena_esperada: str, sena_detectada: str) -> bool:
    """
    Verifica si una confusión es común entre dos señas.
    
    Args:
        sena_esperada: Seña que se esperaba
        sena_detectada: Seña que se detectó
        
    Returns:
        True si es una confusión común
    """
    if sena_esperada.upper() == sena_detectada.upper():
        return False
    
    confusiones = obtener_confusiones_comunes(sena_esperada)
    return sena_detectada.upper() in [c.upper() for c in confusiones]

def calibrar_modelo_usuario(usuario_id: int, senas_calibracion: Dict[str, str]) -> Dict[str, Any]:
    """
    Calibra el modelo para un usuario específico basado en señas de calibración.
    
    Args:
        usuario_id: ID del usuario
        senas_calibracion: Dict con {sena_esperada: sena_detectada}
        
    Returns:
        Dict con resultados de calibración
    """
    resultados = {}
    ajustes = {}
    
    for sena_esperada, sena_detectada in senas_calibracion.items():
        if sena_esperada != sena_detectada:
            # Calcular ajuste necesario
            tipo_esperado = detectar_tipo_sena(sena_esperada)
            tipo_detectado = detectar_tipo_sena(sena_detectada)
            
            if tipo_esperado != tipo_detectado:
                ajustes[sena_esperada] = {
                    'tipo_corregido': tipo_esperado,
                    'confianza_ajustada': 0.7,
                    'factor_compensacion': 1.2,
                    'confusion_comun': es_confusion_comun(sena_esperada, sena_detectada)
                }
            elif es_confusion_comun(sena_esperada, sena_detectada):
                ajustes[sena_esperada] = {
                    'confusion_comun': True,
                    'sena_confundida': sena_detectada,
                    'factor_penalizacion': 0.8
                }
    
    return {
        'usuario_id': usuario_id,
        'ajustes_aplicados': ajustes,
        'total_errores': len(ajustes),
        'fecha_calibracion': datetime.now().isoformat()
    }

def generar_recomendaciones_calibracion(errores: Dict[str, Any]) -> List[str]:
    """
    Genera recomendaciones basadas en los errores de calibración.
    
    Args:
        errores: Dict con información de errores
        
    Returns:
        Lista de recomendaciones
    """
    recomendaciones = []
    
    for sena, info_error in errores.items():
        if info_error.get('confusion_comun'):
            sena_confundida = info_error.get('sena_confundida', 'desconocida')
            recomendaciones.append(
                f"Confusión común: '{sena}' se confunde con '{sena_confundida}'. "
                f"Enfócate en las diferencias clave entre estas señas."
            )
        elif info_error.get('tipo_corregido'):
            tipo_esperado = info_error['tipo_corregido']
            recomendaciones.append(
                f"Seña '{sena}' es {tipo_esperado.lower()}. "
                f"Asegúrate de ejecutar la seña de manera {'estática' if tipo_esperado == 'ESTATICA' else 'dinámica'}."
            )
    
    # Recomendaciones generales
    if len(errores) > 3:
        recomendaciones.append(
            "Múltiples errores detectados. Considera practicar las señas básicas "
            "y mejorar la iluminación y ángulo de la cámara."
        )
    
    if not recomendaciones:
        recomendaciones.append("Calibración exitosa. No se detectaron errores significativos.")
    
    return recomendaciones

def validar_consistencia_categoria(senas: List[str]) -> Dict[str, Any]:
    """
    Valida la consistencia de tipos de señas en una categoría.
    
    Args:
        senas: Lista de señas en la categoría
        
    Returns:
        Dict con análisis de consistencia
    """
    tipos = [detectar_tipo_sena(sena) for sena in senas]
    conteo_tipos = {
        'ESTATICA': tipos.count('ESTATICA'),
        'DINAMICA': tipos.count('DINAMICA')
    }
    
    total = len(tipos)
    proporcion_estaticas = conteo_tipos['ESTATICA'] / total if total > 0 else 0
    proporcion_dinamicas = conteo_tipos['DINAMICA'] / total if total > 0 else 0
    
    tipo_mayoritario = 'ESTATICA' if conteo_tipos['ESTATICA'] > conteo_tipos['DINAMICA'] else 'DINAMICA'
    
    return {
        'total_senas': total,
        'conteo_tipos': conteo_tipos,
        'proporcion_estaticas': round(proporcion_estaticas, 3),
        'proporcion_dinamicas': round(proporcion_dinamicas, 3),
        'tipo_mayoritario': tipo_mayoritario,
        'consistente': (proporcion_estaticas > 0.8 or proporcion_dinamicas > 0.8)
    }

def es_categoria_alfabeto(senas: List[str]) -> bool:
    """
    Determina si una categoría contiene principalmente señas de alfabeto.
    
    Args:
        senas: Lista de señas en la categoría
        
    Returns:
        True si es una categoría de alfabeto
    """
    if not senas:
        return False
    
    letras_alfabeto = set('abcdefghijklmnopqrstuvwxyz')
    conteo_letras = sum(1 for sena in senas if sena.lower() in letras_alfabeto)
    
    return conteo_letras / len(senas) > 0.7

def obtener_parametros_entrenamiento(senas: List[str]) -> Dict[str, Any]:
    """
    Obtiene parámetros de entrenamiento optimizados para un conjunto de señas.
    
    Args:
        senas: Lista de señas a entrenar
        
    Returns:
        Dict con parámetros optimizados
    """
    analisis_consistencia = validar_consistencia_categoria(senas)
    es_alfabeto = es_categoria_alfabeto(senas)
    
    # Parámetros base
    parametros = {
        'batch_size': 8,
        'num_frames': 16,
        'learning_rate': 0.001,
        'epochs': 150,
        'weight_decay': 1e-4
    }
    
    # Ajustar según el tipo predominante
    if analisis_consistencia['tipo_mayoritario'] == 'ESTATICA':
        parametros['num_frames'] = 8
        parametros['batch_size'] = 12  # Más batches para señas estáticas
    else:
        parametros['num_frames'] = 16
        parametros['batch_size'] = 6   # Menos batches para señas dinámicas (más memoria)
    
    # Ajustar para alfabeto
    if es_alfabeto:
        parametros['learning_rate'] = 0.0005  # Learning rate más bajo para alfabeto
        parametros['epochs'] = 200  # Más épocas para alfabeto
    
    return {
        **parametros,
        'analisis_consistencia': analisis_consistencia,
        'es_categoria_alfabeto': es_alfabeto,
        'recomendaciones': generar_recomendaciones_entrenamiento(analisis_consistencia, es_alfabeto)
    }

def generar_recomendaciones_entrenamiento(consistencia: Dict[str, Any], es_alfabeto: bool) -> List[str]:
    """
    Genera recomendaciones específicas para el entrenamiento.
    
    Args:
        consistencia: Análisis de consistencia de tipos
        es_alfabeto: Si es categoría de alfabeto
        
    Returns:
        Lista de recomendaciones
    """
    recomendaciones = []
    
    if consistencia['consistente']:
        if consistencia['tipo_mayoritario'] == 'ESTATICA':
            recomendaciones.append("Categoría predominantemente estática - usar procesamiento optimizado para señas estáticas")
        else:
            recomendaciones.append("Categoría predominantemente dinámica - usar procesamiento optimizado para secuencias")
    else:
        recomendaciones.append("Categoría mixta - usar modelo adaptativo con procesamiento diferenciado")
    
    if es_alfabeto:
        recomendaciones.append("Categoría de alfabeto - entrenar con learning rate bajo y más épocas")
    
    return recomendaciones