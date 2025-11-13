#!/usr/bin/env bash
set -o errexit

echo "📦 Instalando dependencias Python..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎬 Instalando FFmpeg..."
# Crear directorio para binarios
mkdir -p $HOME/bin
cd $HOME/bin

# Descargar FFmpeg estático
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extraer
tar -xf ffmpeg-release-amd64-static.tar.xz

# Mover binarios
cp ffmpeg-*-amd64-static/ffmpeg .
cp ffmpeg-*-amd64-static/ffprobe .
chmod +x ffmpeg ffprobe

# Limpiar
rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static

# Verificar
./ffmpeg -version

echo "📁 Creando directorios..."
cd /opt/render/project/src
mkdir -p archivos_subidos/temp
mkdir -p logs  
mkdir -p modelos_entrenados

echo "✅ Build completado"