#!/usr/bin/env python3
"""
Test final de Grok con streaming
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Cargar variables de entorno
load_dotenv()

# Configuración Grok
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# Tools para búsquedas web
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Busca en internet y devuelve un resumen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta de búsqueda."}
                },
                "required": ["query"]
            }
        }
    }
]

conversation_history = [
    {
        "role": "system",
        "content": "Eres Botijo, un androide paranoide con voz agresiva y sarcástica. Cuando algo te parece cutre dices que es 'una zarria'."
    },
    {"role": "user", "content": "Hola Botijo, ¿cómo estás?"}
]

# Test de streaming
try:
    print("🧪 [TEST] Probando streaming con grok-3-fast...")
    
    # Primera llamada para chequear tools
    response = client.chat.completions.create(
        model="grok-3-fast",
        messages=conversation_history,
        tools=TOOLS,
        tool_choice="auto",
        temperature=1,
        max_tokens=200
    )
    
    msg = response.choices[0].message
    
    if msg.tool_calls:
        print("🔧 [INFO] Grok quiere usar tools")
    else:
        print("💬 [INFO] Respuesta directa, iniciando streaming...")
        
        response_stream = client.chat.completions.create(
            model="grok-3-fast",
            messages=conversation_history,
            temperature=1,
            max_tokens=200,
            stream=True
        )
        
        print("🤖 Botijo: ", end="", flush=True)
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # Nueva línea
        
    print("✅ [TEST] Streaming completado exitosamente!")
    
except Exception as e:
    print(f"❌ [TEST] Error: {e}")
