#!/usr/bin/env python3
"""
Script de validación para integratorpico3.py
Verifica que todas las dependencias, archivos y configuraciones estén correctas.
"""

import os
import sys
from dotenv import load_dotenv

def check_file_exists(filepath, description):
    """Verificar que un archivo existe"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NO encontrado: {filepath}")
        return False

def check_env_variable(var_name):
    """Verificar que una variable de entorno existe"""
    value = os.getenv(var_name)
    if value:
        print(f"✅ Variable de entorno {var_name}: {'*' * min(10, len(value))}...")
        return True
    else:
        print(f"❌ Variable de entorno {var_name} NO encontrada")
        return False

def check_imports():
    """Verificar que todas las librerías necesarias se pueden importar"""
    imports_to_check = [
        ("sounddevice", "sounddevice"),
        ("pvporcupine", "pvporcupine"),
        ("openai", "OpenAI"),
        ("elevenlabs", "ElevenLabs"),
        ("google.cloud.speech", "Google Cloud Speech"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("pyaudio", "PyAudio"),
        ("pydub", "Pydub"),
        ("board", "Adafruit CircuitPython"),
        ("neopixel", "Adafruit NeoPixel"),
        ("adafruit_servokit", "Adafruit ServoKit"),
        ("picamera2", "Picamera2"),
    ]
    
    all_imports_ok = True
    print("\n🔍 Verificando importaciones...")
    
    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def main():
    print("🧪 Validación de integratorpico3.py")
    print("=" * 50)
    
    # Cargar variables de entorno
    load_dotenv()
    
    # 1. Verificar archivos críticos
    print("\n📁 Verificando archivos...")
    files_ok = True
    files_to_check = [
        ("/home/jack/botijo/amanece.ppn", "Modelo wake word 'amanece'"),
        ("/home/jack/botijo/botijo.ppn", "Modelo wake word 'botijo'"),
        ("/home/jack/botijo/porcupine_params_es.pv", "Modelo idioma español Picovoice"),
        ("/home/jack/botijo/.env", "Archivo de variables de entorno"),
        ("/home/jack/botijo/packett/network.rpk", "Modelo de seguimiento facial"),
        ("/home/jack/botijo/labels.txt", "Archivo de etiquetas"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            files_ok = False
    
    # 2. Verificar variables de entorno
    print("\n🔐 Verificando variables de entorno...")
    env_ok = True
    env_vars = [
        "PICOVOICE_ACCESS_KEY",
        "OPENAI_API_KEY", 
        "ELEVENLABS_API_KEY"
    ]
    
    for var in env_vars:
        if not check_env_variable(var):
            env_ok = False
    
    # 3. Verificar importaciones
    imports_ok = check_imports()
    
    # 4. Verificar sintaxis del archivo principal
    print("\n📝 Verificando sintaxis de integratorpico3.py...")
    try:
        import ast
        with open("/home/jack/botijo/integratorpico3.py", 'r') as f:
            ast.parse(f.read())
        print("✅ Sintaxis del archivo principal correcta")
        syntax_ok = True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        syntax_ok = False
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        syntax_ok = False
    
    # 5. Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VALIDACIÓN:")
    print(f"  📁 Archivos: {'✅ OK' if files_ok else '❌ FALTAN'}")
    print(f"  🔐 Variables entorno: {'✅ OK' if env_ok else '❌ FALTAN'}")
    print(f"  📦 Dependencias: {'✅ OK' if imports_ok else '❌ FALTAN'}")
    print(f"  📝 Sintaxis: {'✅ OK' if syntax_ok else '❌ ERROR'}")
    
    all_ok = files_ok and env_ok and imports_ok and syntax_ok
    
    if all_ok:
        print("\n🎉 ¡TODO CORRECTO! El sistema está listo para ejecutarse.")
        print("\n🚀 Para ejecutar:")
        print("   python integratorpico3.py")
        print("\n🎤 Flujo de uso:")
        print("   1. Di 'Amanece' para despertar el sistema")
        print("   2. Di 'Botijo' seguido inmediatamente de tu comando")
        print("   3. El sistema procesará tu comando con Google STT")
    else:
        print("\n⚠️  HAY PROBLEMAS QUE CORREGIR ANTES DE EJECUTAR")
        if not files_ok:
            print("   • Verifica que todos los archivos de modelo existen")
        if not env_ok:
            print("   • Configura las variables de entorno faltantes en .env")
        if not imports_ok:
            print("   • Instala las dependencias faltantes")
        if not syntax_ok:
            print("   • Corrige los errores de sintaxis")

if __name__ == "__main__":
    main()
