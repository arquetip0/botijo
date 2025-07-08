#!/usr/bin/env python3
"""
Script de prueba para verificar la detección de dos wake words con Picovoice:
- "amanece" (índice 0) - Para despertar el sistema
- "botijo" (índice 1) - Para comandos
"""

import os
import pvporcupine
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_picovoice_dual_keywords():
    """Probar la carga de Picovoice con dos modelos personalizados"""
    
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("❌ ERROR: No se encontró PICOVOICE_ACCESS_KEY")
        return False
    
    try:
        # Rutas de los modelos
        keyword_paths = [
            "/home/jack/botijo/amanece.ppn",  # Índice 0 - Para despertar
            "/home/jack/botijo/botijo.ppn"    # Índice 1 - Para comandos
        ]
        model_path = "/home/jack/botijo/porcupine_params_es.pv"
        
        print("🔍 Verificando archivos...")
        for i, path in enumerate(keyword_paths):
            if os.path.exists(path):
                print(f"✅ Modelo {i} encontrado: {path}")
            else:
                print(f"❌ Modelo {i} NO encontrado: {path}")
                return False
        
        if os.path.exists(model_path):
            print(f"✅ Modelo de idioma encontrado: {model_path}")
        else:
            print(f"❌ Modelo de idioma NO encontrado: {model_path}")
            return False
        
        print("\n🚀 Inicializando Picovoice con dos wake words...")
        
        # Crear instancia de Porcupine
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            model_path=model_path
        )
        
        print("✅ Picovoice inicializado correctamente!")
        print(f"📊 Frame length: {porcupine.frame_length}")
        print(f"📊 Sample rate: {porcupine.sample_rate}")
        print(f"📊 Número de keywords: {len(keyword_paths)}")
        print("🎯 Keywords configurados:")
        print("   - Índice 0: 'amanece' (despertar sistema)")
        print("   - Índice 1: 'botijo' (activar comandos)")
        
        # Limpiar recursos
        porcupine.delete()
        print("\n🧹 Recursos de Picovoice liberados")
        return True
        
    except Exception as e:
        print(f"❌ ERROR al inicializar Picovoice: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Prueba de configuración dual de Picovoice")
    print("=" * 50)
    
    success = test_picovoice_dual_keywords()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ÉXITO: Configuración dual de Picovoice funcional")
        print("🎤 Flujo de uso:")
        print("   1. Di 'Amanece' para despertar el sistema")
        print("   2. Di 'Botijo' seguido de tu comando")
        print("   3. El sistema procesará el comando con Google STT")
    else:
        print("❌ FALLO: Problemas en la configuración")
