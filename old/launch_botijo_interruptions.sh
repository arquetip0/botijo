#!/bin/bash

# Script de lanzamiento para Botijo con sistema de interrupciones ReSpeaker v2.0
# Optimizado para SEE-14133

clear
echo "🤖 Lanzando Botijo con Sistema de Interrupciones ReSpeaker v2.0"
echo "=================================================================="
echo ""
echo "🎤 Hardware: ReSpeaker Mic Array v2.0 (SEE-14133)"
echo "🔧 Características:"
echo "   - Beamforming circular de 4 micrófonos"
echo "   - Cancelación de eco adaptativa LMS"
echo "   - WebRTC VAD profesional (Modo 3)"
echo "   - Detección de interrupciones por voz humana"
echo "   - Respuestas con límite automático y continuación"
echo ""
echo "💡 Funcionalidades de interrupciones:"
echo "   ✓ Detección en tiempo real durante respuestas"
echo "   ✓ Pregunta automática: '¿Deseas que continúe?'"
echo "   ✓ Continuación inteligente del contexto"
echo "   ✓ Filtrado avanzado de ruido ambiental"
echo ""
echo "🚀 Iniciando en 3 segundos..."
sleep 3

cd /home/jack/botijo
python3 gpt5botijonew.py

echo ""
echo "👋 Botijo desconectado. ¡Hasta pronto!"
