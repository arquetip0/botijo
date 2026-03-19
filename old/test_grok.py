#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración de Grok
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Cargar variables de entorno
load_dotenv()

# Configuración Grok
try:
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ [ERROR] No se encontró la clave XAI_API_KEY en el archivo .env")
        exit(1)
    
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    print("✅ [INFO] Cliente Grok inicializado correctamente")
    print(f"🔑 [INFO] API Key (primeros 10 chars): {xai_api_key[:10]}...")
except Exception as e:
    print(f"❌ [ERROR] Error inicializando cliente Grok: {e}")
    exit(1)

# Test con diferentes modelos
modelos_a_probar = ["grok-3-fast", "grok-1", "grok", "grok-3-fast-1212", "grok-3-fast-latest"]

for modelo in modelos_a_probar:
    try:
        print(f"\n🧪 [TEST] Probando modelo: {modelo}")
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres Botijo, un androide sarcástico."},
                {"role": "user", "content": "Hola, ¿funcionas?"}
            ],
            temperature=1,
            max_tokens=100
        )
        
        print(f"✅ [TEST] ¡Modelo {modelo} funciona!")
        print(f"🤖 Grok: {response.choices[0].message.content}")
        break
        
    except Exception as e:
        print(f"❌ [TEST] Modelo {modelo} falló: {str(e)[:100]}...")
        continue
