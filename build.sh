#!/usr/bin/env bash
set -o errexit

echo "📦 Actualizando pip..."
pip install --upgrade pip setuptools wheel

echo "📚 Instalando dependencias Python..."
pip install -r requirements.txt

echo "🎬 Instalando FFmpeg..."
# Crear directorio para binarios
mkdir -p $HOME/bin

# Descargar FFmpeg estático
cd $HOME/bin
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz

# Mover binarios
cp ffmpeg-*-amd64-static/ffmpeg .
cp ffmpeg-*-amd64-static/ffprobe .
chmod +x ffmpeg ffprobe

# Limpiar
rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static

# Verificar instalación
./ffmpeg -version

echo "📁 Creando directorios del proyecto..."
cd /opt/render/project/src
mkdir -p archivos_subidos/temp
mkdir -p logs
mkdir -p modelos_entrenados

echo "✅ Build completado exitosamente"