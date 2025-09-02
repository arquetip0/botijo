#!/usr/bin/env python3
"""
Test simple para verificar si GPT-5 funciona
"""

import openai
import json

# Leer configuración
with open('config.json', 'r') as f:
    config = json.load(f)

# Configurar cliente
client = openai.OpenAI(api_key=config['openai_api_key'])

def test_gpt5_simple():
    print("🧪 [TEST] Probando GPT-5 con mensaje simple...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": "Responde solo 'Hola mundo' - nada más"}
            ],
            max_completion_tokens=50,
            temperature=0.1
        )
        
        print(f"✅ [TEST] GPT-5 responde: '{response.choices[0].message.content}'")
        return True
        
    except Exception as e:
        print(f"❌ [TEST] Error con GPT-5: {e}")
        return False

def test_gpt5_streaming():
    print("🧪 [TEST] Probando GPT-5 con streaming...")
    
    try:
        stream = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": "Di 'streaming funciona'"}
            ],
            max_completion_tokens=50,
            temperature=0.1,
            stream=True
        )
        
        print("🔄 [TEST] Recibiendo chunks...")
        content = ""
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                print(f"📦 [CHUNK]: '{chunk.choices[0].delta.content}'")
        
        print(f"✅ [TEST] Streaming GPT-5 completo: '{content}'")
        return True
        
    except Exception as e:
        print(f"❌ [TEST] Error con streaming GPT-5: {e}")
        return False

if __name__ == "__main__":
    print("🚀 [TEST] Iniciando pruebas de GPT-5...")
    test_gpt5_simple()
    print("\n" + "="*50 + "\n")
    test_gpt5_streaming()
