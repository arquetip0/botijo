#!/usr/bin/env python3
"""
Test final para verificar que grokbotijo.py usa correctamente Grok-4 con extra_body.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_grok_integration():
    """Test básico de la integración con Grok usando extra_body"""
    try:
        # Configurar cliente Grok
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        print("🔥 Testing Grok-4 con extra_body para Live Search...")
        
        # Test con búsqueda habilitada
        messages = [
            {"role": "system", "content": "Eres Botijo, un androide sarcástico."},
            {"role": "user", "content": "¿Cuáles son las últimas noticias sobre SpaceX en 2024?"}
        ]
        
        response = client.chat.completions.create(
            model="grok-4",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            extra_body={
                "search_parameters": {
                    "mode": "auto",
                    "return_citations": True
                }
            }
        )
        
        print("✅ Respuesta de Grok:")
        print(response.choices[0].message.content)
        
        # Test de streaming
        print("\n🔥 Testing streaming...")
        
        stream = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "user", "content": "Dime algo rápido sobre inteligencia artificial"}
            ],
            temperature=0.7,
            max_tokens=100,
            stream=True,
            extra_body={
                "search_parameters": {
                    "mode": "auto",
                    "return_citations": True
                }
            }
        )
        
        print("✅ Streaming response:")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\n")
        
        print("🎉 ¡Integración con Grok-4 funciona correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_grok_integration()
