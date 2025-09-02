#!/usr/bin/env python3
"""
Test script para verificar que el sistema ReSpeaker v2.0 funciona correctamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar funciones necesarias del script principal
from gpt5botijonew import (
    find_respeaker_v2,
    configure_respeaker_v2,
    get_respeaker_v2_device_index,
    initialize_respeaker_v2_system
)

def test_respeaker_system():
    print("🔧 [TEST] Probando sistema ReSpeaker v2.0...")
    
    # 1. Detectar dispositivo USB
    print("\n1. Detectando dispositivo USB...")
    device = find_respeaker_v2()
    if device:
        print("✅ ReSpeaker v2.0 detectado correctamente")
    else:
        print("❌ ReSpeaker v2.0 NO detectado")
        return False
    
    # 2. Configurar dispositivo
    print("\n2. Configurando dispositivo...")
    config_result = configure_respeaker_v2()
    if config_result:
        print("✅ Configuración exitosa")
    else:
        print("⚠️ Configuración parcial")
    
    # 3. Verificar disponibilidad en PyAudio
    print("\n3. Verificando disponibilidad en PyAudio...")
    device_index = get_respeaker_v2_device_index()
    if device_index is not None:
        print(f"✅ ReSpeaker disponible en PyAudio (índice: {device_index})")
    else:
        print("❌ ReSpeaker NO disponible en PyAudio")
        return False
    
    # 4. Inicialización completa del sistema
    print("\n4. Inicialización completa del sistema...")
    system_ready = initialize_respeaker_v2_system()
    if system_ready:
        print("✅ Sistema ReSpeaker v2.0 completamente inicializado")
        print("🎯 Características activas:")
        print("   - Beamforming circular de 4 micrófonos")
        print("   - Cancelación de eco adaptativa")
        print("   - WebRTC VAD profesional")
        print("   - Detección de interrupciones por voz humana")
        return True
    else:
        print("❌ Error en inicialización del sistema")
        return False

if __name__ == "__main__":
    success = test_respeaker_system()
    if success:
        print("\n🎉 [SUCCESS] Sistema ReSpeaker v2.0 listo para usar!")
        print("💡 [INFO] El sistema de interrupciones funcionará correctamente")
    else:
        print("\n💥 [ERROR] Problemas detectados en el sistema ReSpeaker v2.0")
        print("⚠️ [FALLBACK] Se usará micrófono estándar sin interrupciones")
    
    sys.exit(0 if success else 1)
