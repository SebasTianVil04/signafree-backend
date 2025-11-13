#!/usr/bin/env bash
set -o errexit

echo "📦 Actualizando pip..."
pip install --upgrade pip

echo "📚 Instalando dependencias..."
pip install -r requirements.txt

echo "📁 Creando directorios necesarios..."
mkdir -p archivos_subidos/temp
mkdir -p archivos_subidos/videos_dataset
mkdir -p archivos_subidos/frames_dataset
mkdir -p modelos_entrenados
mkdir -p logs

echo "✅ Build completado"