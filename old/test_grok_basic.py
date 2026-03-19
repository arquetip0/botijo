#!/usr/bin/env python3
"""
Test simple de Grok-4 sin parámetros extras
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_basic_grok():
    try:
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        print("🔥 Testing Grok-4 básico...")
        
        response = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": "Eres Botijo, un androide sarcástico."},
                {"role": "user", "content": "Hola, ¿cómo estás?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("✅ Respuesta:")
        print(response.choices[0].message.content)
        
        # Test de streaming
        print("\n🔥 Testing streaming...")
        
        stream = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "user", "content": "Di hola"}
            ],
            temperature=0.7,
            max_tokens=50,
            stream=True
        )
        
        print("✅ Streaming:")
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_response += content
        print(f"\n\nRespuesta completa: '{full_response}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_grok()
