# grok3.py
# Script definitivo del androide discutidor con wake word "botijo",
# reconocimiento de voz local con Vosk, respuesta con ChatGPT, y síntesis con Piper

import os
import subprocess
import sounddevice as sd
import vosk
from openai import OpenAI
from dotenv import load_dotenv
import json
import threading
import time

# === CONFIGURACIÓN ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Rutas y configuración
VOSK_MODEL_PATH = "model"  # Asegúrate de que este path es correcto
PIPER_BINARY = "/home/jack/piper/build/piper"
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"
SAMPLE_RATE = 16000

# Sistema de audio
audio_queue = []
is_speaking = False

def init_vosk():
    """Inicializa el modelo Vosk"""
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en: {VOSK_MODEL_PATH}")
    return vosk.Model(VOSK_MODEL_PATH)

def hablar(texto):
    """Sintetiza y reproduce el texto usando Piper"""
    global is_speaking
    is_speaking = True
    
    try:
        # Generar audio con Piper
        cmd = [
            PIPER_BINARY,
            "--model", PIPER_MODEL_PATH,
            "--output_file", "respuesta.wav"
        ]
        
        process = subprocess.run(
            cmd,
            input=texto,
            text=True,
            capture_output=True,
            check=True
        )
        
        # Reproducir el audio
        if os.path.exists("respuesta.wav"):
            subprocess.run(["aplay", "respuesta.wav"], check=True)
            os.remove("respuesta.wav")
        
    except subprocess.CalledProcessError as e:
        print(f"Error en Piper: {e.stderr}")
    except Exception as e:
        print(f"Error en síntesis de voz: {e}")
    finally:
        is_speaking = False

def audio_callback(indata, frames, time, status):
    """Callback para el stream de audio"""
    if status:
        print(f"Error de audio: {status}")
    if not is_speaking:  # Solo procesa audio cuando no está hablando
        audio_queue.append(bytes(indata))

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
                if audio_queue and not is_speaking:
                    audio_data = audio_queue.pop(0)
                    if recognizer.AcceptWaveform(audio_data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower()
                        
                        if text:  # Si hay texto, resetear el tiempo de inactividad
                            has_warned = False
                            if "botijo" in text or is_conversation_active:
                                last_interaction_time = current_time
                                if not is_conversation_active:
                                    print("\n¡Botijo activado! Iniciando conversación...")
                                    hablar("¿Qué quieres ahora, ser inferior?")
                                    is_conversation_active = True
                                    last_interaction_time = current_time
                                    continue
                                
                                # Procesar entrada del usuario si hay texto
                                if text:
                                    print(f"\nHumano: {text}")
                                    last_interaction_time = current_time
                                    
                                    try:
                                        # Añadir input del usuario al historial
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
                                        
                                        # Mantener el historial en un tamaño manejable
                                        if len(conversation_history) > 10:
                                            # Mantener el sistema y los últimos 4 intercambios
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
                
                time.sleep(0.1)  # Evitar uso excesivo de CPU
                
        except KeyboardInterrupt:
            print("\nApagando el sistema...")
        except Exception as e:
            print(f"Error inesperado: {e}")
        finally:
            stream.stop()
            
if __name__ == "__main__":
    main()
