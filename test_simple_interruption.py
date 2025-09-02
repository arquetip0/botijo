#!/usr/bin/env python3
"""
Test simple y directo para verificar interrupciones ReSpeaker v2.0
"""

import sys
import os
import time
import threading

# Configurar path
sys.path.insert(0, '/home/jack/botijo')
os.chdir('/home/jack/botijo')

def simple_test():
    # Importar después de configurar el path
    import gpt5botijonew as bot
    
    print("🎤 Test Simple de Interrupciones ReSpeaker v2.0")
    print("=" * 60)
    
    # Verificar que ReSpeaker esté disponible
    print("🔍 Verificando ReSpeaker v2.0...")
    respeaker_index = bot.get_respeaker_v2_device_index()
    
    if respeaker_index is None:
        print("❌ ReSpeaker v2.0 no encontrado")
        return False
        
    print(f"✅ ReSpeaker v2.0 encontrado en índice: {respeaker_index}")
    
    # Inicializar variables globales necesarias
    bot.is_speaking = False
    bot.interruption_detected = False
    bot.pending_response = ""
    
    print("\n🧪 Iniciando test de detección...")
    print("💡 HABLA AHORA al micrófono para probar la detección")
    print("   (El test durará 10 segundos)")
    
    # Simular que está hablando
    bot.is_speaking = True
    
    # Iniciar monitor
    bot.start_respeaker_v2_interruption_monitor()
    
    # Esperar 10 segundos para que puedas hablar
    start_time = time.time()
    while time.time() - start_time < 10:
        if bot.interruption_detected:
            print(f"\n🎉 ¡INTERRUPCIÓN DETECTADA!")
            print(f"   - Tiempo transcurrido: {time.time() - start_time:.1f}s")
            break
        time.sleep(0.1)
    
    # Detener monitor
    bot.stop_respeaker_v2_interruption_monitor()
    bot.is_speaking = False
    
    if bot.interruption_detected:
        print("✅ RESULTADO: Sistema de interrupciones FUNCIONANDO")
        return True
    else:
        print("❌ RESULTADO: No se detectó ninguna interrupción")
        print("   Prueba hablando más fuerte o acercándote al micrófono")
        return False

if __name__ == "__main__":
    try:
        success = simple_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test cancelado")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
