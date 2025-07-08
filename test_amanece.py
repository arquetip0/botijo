#!/usr/bin/env python3
"""
Prueba rápida para verificar que el modelo personalizado 'amanece.ppn' funciona correctamente
"""
import os
from dotenv import load_dotenv
import pvporcupine

load_dotenv()

PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

def test_amanece_model():
    """Probar si el modelo personalizado se puede cargar"""
    if not PICOVOICE_ACCESS_KEY:
        print("❌ No se encontró PICOVOICE_ACCESS_KEY en .env")
        return False
    
    try:
        keyword_path = "/home/jack/botijo/amanece.ppn"
        model_path = "/home/jack/botijo/porcupine_params_es.pv"
        
        # Verificar que los archivos existen
        if not os.path.exists(keyword_path):
            print(f"❌ Archivo no encontrado: {keyword_path}")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Archivo no encontrado: {model_path}")
            return False
        
        print(f"✅ Archivo keyword encontrado: {keyword_path}")
        print(f"✅ Archivo modelo encontrado: {model_path}")
        
        # Intentar cargar el modelo
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=[keyword_path],
            model_path=model_path
        )
        
        print(f"✅ Modelo 'amanece' cargado exitosamente con español!")
        print(f"   Frame length: {porcupine.frame_length}")
        print(f"   Sample rate: {porcupine.sample_rate}")
        
        # Limpiar recursos
        porcupine.delete()
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Probando modelo personalizado 'amanece.ppn'...")
    success = test_amanece_model()
    
    if success:
        print("\n🎉 ¡El modelo está listo para usar!")
        print("   Puedes ejecutar integratorpico.py y decir 'amanece' para activarlo")
    else:
        print("\n💥 Hay un problema con el modelo")
        print("   Verifica tu clave de Picovoice y el archivo amanece.ppn")
