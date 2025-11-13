import os
import subprocess
import shutil

def get_ffmpeg_path():
    """Obtener la ruta de FFmpeg en Render"""
    # En Render, FFmpeg está en $HOME/bin
    render_ffmpeg = os.path.join(os.environ.get('HOME', ''), 'bin', 'ffmpeg')
    
    if os.path.exists(render_ffmpeg):
        return render_ffmpeg
    
    # Buscar en PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Local/desarrollo
    return 'ffmpeg'

def verificar_ffmpeg():
    """Verificar que FFmpeg está disponible"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_path, '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✅ FFmpeg encontrado en: {ffmpeg_path}")
        print(f"Versión: {result.stdout.split('version')[1].split()[0]}")
        return True
    except Exception as e:
        print(f"❌ Error con FFmpeg: {e}")
        return False

# Configurar variable de entorno
os.environ['FFMPEG_BINARY'] = get_ffmpeg_path()