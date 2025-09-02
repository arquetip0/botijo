#!/usr/bin/env python3
"""
Test específico para verificar el sistema de interrupciones ReSpeaker v2.0
"""

import time
import threading
import sys
import os

# Añadir el directorio del proyecto al path
sys.path.append('/home/jack/botijo')

# Importar solo las funciones necesarias
from gpt5botijonew import (
    hablar_generador_respeaker_v2_optimized,
    initialize_respeaker_v2_system,
    is_speaking,
    interruption_detected,
    pending_response
)

def test_generator():
    """Generador de prueba que simula una respuesta larga"""
    test_text = """
    Hola humano, soy Botijo y voy a hablar durante un rato largo para que puedas probar las interrupciones. 
    Este es un texto bastante extenso que debería tomar varios segundos en ser sintetizado y reproducido. 
    Durante este tiempo, deberías poder interrumpirme simplemente hablando al micrófono ReSpeaker v2.0. 
    El sistema utiliza beamforming avanzado y WebRTC VAD para detectar tu voz con alta precisión. 
    Si todo funciona correctamente, debería parar de hablar inmediatamente cuando detecte tu voz. 
    Esto es exactamente como funciona ChatGPT en modo audio, permitiendo conversaciones mucho más naturales. 
    La tecnología ReSpeaker v2.0 con sus 4 micrófonos en configuración circular proporciona una 
    detección de voz superior comparada con micrófonos simples. Puedes probar a interrumpirme ahora 
    diciendo algo como 'para', 'espera' o cualquier otra palabra.
    """
    
    # Simular streaming de texto como haría ChatGPT
    words = test_text.split()
    for word in words:
        yield word + " "
        time.sleep(0.1)  # Simular latencia de streaming

def main():
    print("🎤 Test de Interrupciones ReSpeaker v2.0")
    print("=" * 50)
    
    # Inicializar sistema ReSpeaker
    try:
        print("🔧 Inicializando ReSpeaker v2.0...")
        respeaker_available = initialize_respeaker_v2_system()
        
        if not respeaker_available:
            print("❌ ReSpeaker v2.0 no disponible")
            return
            
        print("✅ ReSpeaker v2.0 inicializado")
        print()
        print("💡 INSTRUCCIONES:")
        print("   1. El sistema comenzará a hablar")
        print("   2. Mientras habla, di algo (ej: 'para', 'espera')")
        print("   3. Debería interrumpirse inmediatamente")
        print("   4. El sistema dirá si fue interrumpido")
        print()
        input("Presiona ENTER para comenzar la prueba...")
        
        # Ejecutar test
        print("\n🤖 Iniciando test de habla larga...")
        was_interrupted, response = hablar_generador_respeaker_v2_optimized(test_generator())
        
        print(f"\n📊 RESULTADOS DEL TEST:")
        print(f"   - Fue interrumpido: {'✅ SÍ' if was_interrupted else '❌ NO'}")
        print(f"   - Respuesta completa: {len(response)} caracteres")
        print(f"   - Respuesta pendiente: {'✅ SÍ' if pending_response else '❌ NO'}")
        
        if was_interrupted:
            print("\n🎉 ¡SISTEMA DE INTERRUPCIONES FUNCIONANDO!")
            print("   El ReSpeaker v2.0 detectó tu voz correctamente")
        else:
            print("\n⚠️  Sistema completó sin interrupciones")
            print("   Puede que no hayas hablado o el micrófono no detectó tu voz")
            
    except KeyboardInterrupt:
        print("\n🛑 Test cancelado por usuario")
    except Exception as e:
        print(f"\n❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
