# grok3.py
# Script definitivo del androide discutidor con wake word "botijo",
# reconocimiento de voz local con Vosk, respuesta con ChatGPT, y síntesis con Piper

import os
import subprocess
import sounddevice as sd
import vosk
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
import threading
import time
from collections import deque

# === CONFIGURACIÓN ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Rutas y configuración
VOSK_MODEL_PATH = "model"  # Asegúrate de que este path es correcto
PIPER_BINARY = "/home/jack/piper/build/piper"
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"
SAMPLE_RATE = 16000

# Configuración del procesamiento de audio
SILENCE_THRESHOLD = 100
MIN_PHRASE_LENGTH = 2
PAUSE_LENGTH = 1.0  # segundos de silencio para considerar fin de frase

# Sistema de audio
audio_queue = deque(maxlen=1000)  # Limitar el tamaño máximo de la cola
is_speaking = False
accumulated_text = ""
silence_duration = 0

def init_vosk():
    """Inicializa el modelo Vosk"""
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en: {VOSK_MODEL_PATH}")
    return vosk.Model(VOSK_MODEL_PATH)

def audio_callback(indata, frames, time, status):
    """Callback para el stream de audio"""
    global silence_duration
    if status:
        print(f"Error de audio: {status}")
    if not is_speaking:
        # Convertir a array numpy para detectar silencio
        audio_array = np.frombuffer(bytes(indata), dtype=np.int16)
        if np.max(np.abs(audio_array)) < SILENCE_THRESHOLD:
            silence_duration += frames / SAMPLE_RATE
        else:
            silence_duration = 0
        audio_queue.append((bytes(indata), frames))

def process_audio_chunk(recognizer):
    """Procesa chunks de audio hasta detectar una frase completa"""
    global accumulated_text, silence_duration
    
    while audio_queue and not is_speaking:
        audio_data, frames = audio_queue.popleft()
        
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                accumulated_text += " " + text
                
        # Detectar fin de frase por silencio
        if silence_duration > PAUSE_LENGTH and accumulated_text:
            words = accumulated_text.split()
            if len(words) >= MIN_PHRASE_LENGTH:
                complete_text = accumulated_text.strip()
                accumulated_text = ""
                silence_duration = 0
                return complete_text
                
    return None

def main():
    # Inicialización
    model = init_vosk()
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    
    # Configuración de timeouts
    INACTIVITY_TIMEOUT = 120  # 2 minutos de inactividad
    WARNING_TIME = 90  # Aviso a los 90 segundos
    has_warned = False  # Flag para controlar el aviso de inactividad
    
    # Lista para mantener el contexto de la conversación
    conversation_history = [
        {"role": "system", "content": (
            "Eres un androide discutidor con voz agresiva y sarcástica. "
            "Respondes con desprecio a las preguntas, usando términos como "
            "'humano insignificante', 'criatura inferior' o similar. "
            "Mantienes el contexto de la conversación para hacer referencias "
            "a comentarios anteriores de forma sarcástica."
        )}
    ]
    
    # Configurar stream de audio
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        channels=1,
        dtype='int16',
        callback=audio_callback
    )
    
    print("Esperando la palabra clave 'botijo'...")
    
    with stream:
        try:
            is_conversation_active = False
            last_interaction_time = time.time()
            
            while True:
                current_time = time.time()
                inactive_time = current_time - last_interaction_time
                
                # Procesar audio solo cuando no está hablando
                if not is_speaking:
                    complete_text = process_audio_chunk(recognizer)
                    
                    if complete_text:
                        text = complete_text.lower()
                        if "botijo" in text or is_conversation_active:
                            last_interaction_time = current_time
                            if not is_conversation_active:
                                print("\n¡Botijo activado! Iniciando conversación...")
                                hablar("¿Qué quieres ahora, ser inferior?")
                                is_conversation_active = True
                                continue
                            
                            print(f"\nHumano: {text}")
                            try:
                                conversation_history.append({"role": "user", "content": text})
                                
                                # Obtener respuesta de GPT con contexto
                                response = client.chat.completions.create(
                                    model="gpt-4-1106-preview",
                                    messages=conversation_history,
                                    temperature=0.9,
                                    max_tokens=150,
                                    timeout=10
                                )
                                
                                respuesta = response.choices[0].message.content
                                conversation_history.append({"role": "assistant", "content": respuesta})
                                
                                if len(conversation_history) > 10:
                                    conversation_history = [conversation_history[0]] + conversation_history[-8:]
                                
                                print(f"Androide: {respuesta}")
                                hablar(respuesta)
                                
                            except Exception as e:
                                print(f"Error con ChatGPT: {e}")
                                hablar("Mi circuito de sarcasmo está sobrecargado. Reiniciando sistemas...")
                
                # Gestión de timeouts
                if is_conversation_active:
                    if inactive_time > INACTIVITY_TIMEOUT:
                        print("\nConversación terminada por inactividad")
                        hablar("Me aburres, humano. Vuelve cuando tengas algo interesante que decir.")
                        is_conversation_active = False
                        has_warned = False
                        conversation_history = [conversation_history[0]]
                    elif inactive_time > WARNING_TIME and not has_warned:
                        hablar("¿Sigues ahí, humano? Tu silencio me resulta... inquietante.")
                        has_warned = True
                
                time.sleep(0.01)  # Reduced sleep time for better responsiveness
                
        except KeyboardInterrupt:
            print("\nApagando el sistema...")
        except Exception as e:
            print(f"Error inesperado: {e}")
        finally:
            stream.stop()
            
if __name__ == "__main__":
    main()
