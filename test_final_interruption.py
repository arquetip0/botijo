#!/usr/bin/env python3
"""
Test final: Verificar interrupciones durante respuestas reales de Botijo
"""

import sys
import os
import time

# Configurar environment
sys.path.insert(0, '/home/jack/botijo')
os.chdir('/home/jack/botijo')

def test_full_interruption():
    print("🎤 Test Final: Interrupciones Durante Respuesta Real")
    print("=" * 60)
    print()
    print("📋 PLAN DE PRUEBA:")
    print("   1. Hacer una pregunta que genere respuesta larga")
    print("   2. Mientras Botijo habla, interrumpir diciendo 'para'")
    print("   3. Verificar que se detiene inmediatamente")
    print("   4. Confirmar que pregunta si deseas continuar")
    print()
    input("Presiona ENTER para comenzar la prueba...")
    
    # Simular una pregunta que genere respuesta larga
    test_question = "cuéntame una historia muy larga y detallada sobre robots"
    
    print(f"\n🤖 Pregunta de prueba: '{test_question}'")
    print("\n💡 INSTRUCCIONES:")
    print("   - Botijo comenzará a responder")
    print("   - Después de 3-5 segundos, di 'para' o 'espera'")
    print("   - Debería interrumpirse inmediatamente")
    print()
    input("¿Listo? Presiona ENTER para ejecutar...")
    
    # Ejecutar Botijo con la pregunta
    cmd = f"cd /home/jack/botijo && echo '{test_question}' | timeout 30 python3 gpt5botijonew.py"
    print(f"\n🚀 Ejecutando: {cmd}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=35)
        print(f"\n📊 RESULTADO DEL TEST:")
        print(f"   - Código de salida: {result.returncode}")
        print(f"   - Output length: {len(result.stdout)} chars")
        
        if "INTERRUMPIDO" in result.stdout:
            print("   ✅ INTERRUPCIÓN DETECTADA EN LA SALIDA")
        else:
            print("   ❌ No se encontró evidencia de interrupción")
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Test terminado por timeout (normal)")
    except Exception as e:
        print(f"   ❌ Error durante el test: {e}")

if __name__ == "__main__":
    try:
        test_full_interruption()
    except KeyboardInterrupt:
        print("\n🛑 Test cancelado")
    except Exception as e:
        print(f"\n❌ Error: {e}")
