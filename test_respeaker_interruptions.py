#!/usr/bin/env python3
"""
Prueba específica de interrupciones del ReSpeaker v2.0
"""

import sys
sys.path.insert(0, '.')

from gpt5botijonew import *
import time
import threading

def test_interruptions():
    """Prueba específica de interrupciones mientras Botijo habla"""
    
    print("🧪 [TEST] Iniciando prueba de interrupciones ReSpeaker v2.0")
    
    # Inicializar sistema
    global respeaker_v2_available, interruption_detected, is_speaking
    
    print("🔧 [TEST] Inicializando ReSpeaker v2.0...")
    respeaker_v2_available = initialize_respeaker_v2_system()
    
    if not respeaker_v2_available:
        print("❌ [TEST] ReSpeaker v2.0 no disponible")
        return
    
    print("✅ [TEST] ReSpeaker v2.0 inicializado")
    
    # Reset variables globales
    interruption_detected = False
    is_speaking = False
    
    # Texto largo para probar interrupciones
    test_text = """
    Hola, soy Botijo y estoy probando las interrupciones con ReSpeaker Mic Array v2.0. 
    Este es un texto largo para darte tiempo de interrumpirme mientras hablo.
    El sistema de detección de voz nativa del ReSpeaker debería permitirte interrumpir
    mi respuesta simplemente hablando. Esto simula el comportamiento de ChatGPT donde
    puedes cortar la respuesta en cualquier momento. Si me escuchas hablando, 
    intenta decir algo para ver si se detecta la interrupción.
    """
    
    print("\n🎤 [TEST] Iniciando respuesta con monitoreo de interrupciones...")
    print("💡 [TEST] ¡HABLA AHORA para probar la interrupción!")
    
    # Usar la función optimizada con ReSpeaker v2.0
    interruption_success, response = hablar_generador_respeaker_v2_optimized(test_text.strip())
    
    print(f"\n🏁 [TEST] Resultado de la prueba:")
    print(f"   - Interrumpido: {interruption_success}")
    print(f"   - Respuesta completa: {len(response) > len(test_text.strip()) * 0.8}")
    
    if interruption_success:
        print("✅ [TEST] ¡ÉXITO! Las interrupciones funcionan correctamente")
    else:
        print("⚠️ [TEST] No se detectó interrupción (puede ser normal si no hablaste)")
    
    return interruption_success

if __name__ == "__main__":
    try:
        # Configurar logging básico
        import logging
        logging.basicConfig(level=logging.WARNING)
        
        # Ejecutar prueba
        result = test_interruptions()
        
        print(f"\n🎯 [RESULTADO] Prueba {'EXITOSA' if result else 'COMPLETADA'}")
        
    except KeyboardInterrupt:
        print("\n🛑 [TEST] Prueba cancelada por usuario")
    except Exception as e:
        print(f"\n❌ [TEST] Error en prueba: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 [TEST] Limpiando...")
        emergency_shutdown()
