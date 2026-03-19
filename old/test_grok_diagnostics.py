#!/usr/bin/env python3
"""
Test diagnóstico completo de xAI/Grok
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_grok_diagnostics():
    try:
        api_key = os.getenv("XAI_API_KEY")
        print(f"🔑 API Key encontrada: {api_key[:10]}..." if api_key else "❌ No se encontró XAI_API_KEY")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Test con diferentes modelos
        models_to_test = ["grok-4", "grok-beta", "grok-3-fast"]
        
        for model in models_to_test:
            print(f"\n🔥 Testing modelo: {model}")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Di solo 'hola'"}
                    ],
                    temperature=0.7,
                    max_tokens=10
                )
                
                content = response.choices[0].message.content
                print(f"✅ {model}: '{content}'")
                
                if content and content.strip():
                    print(f"🎉 ¡{model} funciona!")
                    return model
                    
            except Exception as e:
                print(f"❌ {model} error: {e}")
        
        print("\n❌ Ningún modelo funcionó")
        return None
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return None

if __name__ == "__main__":
    working_model = test_grok_diagnostics()
    if working_model:
        print(f"\n🔥 Usar modelo: {working_model}")
    else:
        print("\n💀 Revisar API key y suscripción de xAI")
