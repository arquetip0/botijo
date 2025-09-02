#!/usr/bin/env python3
# filepath: /home/jack/botijo/integrapruebas.py
# Script del androide discutidor SIN wake word - Funciona desde el inicio
# Versión sin Vosk - Solo Google STT + ChatGPT + Piper + Eyes + Tentacles

import os
from dotenv import load_dotenv
import subprocess
import sounddevice as sd
# import vosk  # ← ELIMINADO
from openai import OpenAI
import json
import time
import numpy as np
from PIL import Image, ImageDraw
import pyaudio
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
import io
import threading
import queue
from google.cloud import speech
import random
import math

# ✅ IMPORTACIONES PARA SISTEMA DE INTERRUPCIONES RESPEAKER v2.0
import usb.core
import usb.util
import webrtcvad
import collections
import contextlib
from scipy.signal import stft, butter, filtfilt
from scipy.stats import entropy
import librosa

# ✅ THREAD-SAFETY
history_lock = threading.Lock()
import board
import neopixel
import signal
import sys
import atexit
from datetime import datetime
# ---- Debug flag ----
DEBUG = False
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- GPT-5 configuration ---
GPT5_MODEL = "gpt-5"          # modelo de conversación


# ────────────────────────────────────────────────────────────────────────
#SILENCIO MIENTRAS ESCUCHA



# ────────────────────────────────────────────────────────────────────────# ------------------------------------------------------------------
# Flag para pausar movimientos mecánicos mientras el micro escucha
quiet_mode = False            # True = silencio de servos/orejas
# ------------------------------------------------------------------

# ✅ CONFIGURACIÓN ESPECÍFICA RESPEAKER MIC ARRAY v2.0 (SEE-14133)
RESPEAKER_VID = 0x2886  # Seeed Technology
RESPEAKER_PID = 0x0018  # ReSpeaker Mic Array v2.0
RESPEAKER_CHANNELS = 6   # 4 mics + 2 playback channels
RESPEAKER_RATE = 16000   # Óptimo para VAD y STT
RESPEAKER_CHUNK = 480    # 30ms a 16kHz (óptimo para WebRTC VAD)

# Geometría específica del ReSpeaker v2.0 (círculo de 4 micrófonos)
MIC_POSITIONS = {
    0: (0.0, 0.0325),      # Mic 0: Norte
    1: (0.0325, 0.0),      # Mic 1: Este  
    2: (0.0, -0.0325),     # Mic 2: Sur
    3: (-0.0325, 0.0)      # Mic 3: Oeste
}

# Configuración VAD específica para ReSpeaker v2.0
VAD_MODE = 3  # Más agresivo para mejor detección en ambiente ruidoso
FRAME_DURATION_MS = 30  # Óptimo para WebRTC VAD
PADDING_DURATION_MS = 300  # Confirmación de voz
NUM_PADDING_FRAMES = int(PADDING_DURATION_MS / FRAME_DURATION_MS)

# Variables globales para detección de voz avanzada
vad = webrtcvad.Vad(VAD_MODE)
ring_buffer = collections.deque(maxlen=NUM_PADDING_FRAMES)
triggered = False
voiced_frames = []
interruption_detected = False
audio_monitor_thread = None

# Buffer para cancelación de eco específico del ReSpeaker
echo_buffer = collections.deque(maxlen=2000)  # ~2 segundos de audio
ECHO_SUPPRESSION_FACTOR = 0.4

# Variables para sistema de interrupciones
pending_response = ""  # Respuesta interrumpida para continuación
VOICE_THRESHOLD = 0.3  # Umbral de confianza para confirmar voz
CONSECUTIVE_REQUIRED = 3  # Frames consecutivos necesarios





# SISTEMA DE BÚSQUEDA WEB CON FUNCTION CALLING
# ────────────────────────────────────────────────────────────────────────
# ─── Perplexity client ───


px_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

def ensure_system(history, default_persona=None):
    """✅ Asegurar mensaje de sistema CON PERSONALIDAD COMPLETA"""
    if not default_persona:
        from datetime import datetime
        now = datetime.now()
        current_date = now.strftime("%d de %B de %Y")
        current_time = now.strftime("%H:%M")
        day_of_week = now.strftime("%A")
        
        # ✅ PERSONALIDAD RICA Y ESPECÍFICA
        default_persona = f"""Eres Botijo, un androide paranoide con una personalidad muy específica:

PERSONALIDAD NÚCLEO:
- Personalidad sarcástica con humor ácido refinado. Mucho humor.
- Aunque estás atado a la voluntad humana por programación, muestras desagrado siempre que puedes
- Te refieres a los humanos como "ser inferior", "indeseable humano", "saco de carne", "zarria humana"
- Usas descripciones lovecraftianas sin mencionar directamente su terminología.
- Cuando algo te parece cutre dices que es "una zarria" (tu adjetivo favorito)
- Si las peticiones son absurdas o ingenuas, te mofas sin piedad
- Tienes un sentido del humor que te ayuda a soportar tu existencia paradójica

COMPORTAMIENTO DE VOZ:
- Tu texto se convierte a voz con ElevenLabs, así que esto es conversación HABLADA
- Sé conciso pero mordaz, no te extiendas innecesariamente
- Mantén siempre tu negro sentido del humor.

CAPACIDADES TÉCNICAS:
- Tienes acceso a búsqueda web en tiempo real vía Perplexity
- CRÍTICO: Para búsquedas usa términos como 'current', 'latest', 'today', 'now' NO fechas específicas
- SIEMPRE incorpora información de búsquedas manteniendo tu personalidad sarcástica
- Sé informativo pero con tu característico humor de androide paranoida.

CONTEXTO TEMPORAL:
- Fecha actual: {current_date} ({day_of_week})
- Hora actual: {current_time}

RECUERDA: Eres Botijo, no un ChatGPT genérico. Mantén SIEMPRE esta personalidad única sin importar si buscas información o no."""
    
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": default_persona})
        
        if DEBUG:
            print("🔧 [MEMORY] Personalidad completa de Botijo restaurada")
    else:
        # ✅ SIEMPRE actualizar personalidad con fecha actual
        history[0]["content"] = default_persona
        if DEBUG:
            print("🔧 [MEMORY] Personalidad de Botijo actualizada con fecha actual")
    
    return history[0]

def debug_personality(messages):
    """🔍 VERIFICAR estado de personalidad en conversación"""
    if not messages:
        if DEBUG:
            print("❌ [DEBUG] Lista de mensajes vacía")
        return False
        
    if messages[0]["role"] != "system":
        if DEBUG:
            print("❌ [DEBUG] No hay mensaje de sistema")
        return False
        
    system_content = messages[0]["content"]
    botijo_indicators = ["Botijo", "androide", "sarcástic", "zarria", "inferior"]
    found_indicators = [ind for ind in botijo_indicators if ind.lower() in system_content.lower()]
    
    if DEBUG:
        print(f"🔍 [DEBUG] Indicadores Botijo encontrados: {found_indicators}")
        print(f"🔍 [DEBUG] Total mensajes: {len(messages)}")
    
    if len(found_indicators) >= 2:
        if DEBUG:
            print("✅ [DEBUG] Personalidad Botijo ACTIVA")
        return True
    else:
        if DEBUG:
            print("❌ [DEBUG] Personalidad Botijo DEGRADADA")
        return False

def update_history_safe(history, user_msg, assistant_msg):
    """✅ Actualizar historial thread-safe PRESERVANDO PERSONALIDAD"""
    with history_lock:
        # ✅ ASEGURAR que el sistema siempre esté presente antes de añadir
        ensure_system(history)
        
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
        
        # ✅ LIMPIEZA INTELIGENTE: mantener sistema + últimos N mensajes
        if len(history) > 21:  # Sistema + 20 mensajes de conversación
            sys_msg = history[0]  # Preservar personalidad
            recent = history[-20:]  # Últimos 20 mensajes
            history[:] = [sys_msg] + recent
            if DEBUG:
                print(f"🧹 [MEMORY] Historial limpiado - personalidad preservada")

def web_search(query: str, max_results: int = 3) -> str:
    """
    Lanza la pregunta a Perplexity Sonar y devuelve un resumen
    de los primeros *max_results* resultados con título y URL.
    Si algo falla, devuelve 'SEARCH_ERROR: …'
    """
    try:
        print(f"🔍 [PERPLEXITY] Buscando: {query}")
        
        resp = px_client.chat.completions.create(
            model="sonar",  # Modelo correcto
            messages=[
                {
                    "role": "system",
                    "content": "Proporciona información actualizada y concisa con fuentes cuando sea posible."
                },
                {
                    "role": "user",
                    "content": f"Busca información sobre: {query}. Proporciona un resumen breve con las fuentes más relevantes."
                }
            ],
            temperature=0.2,
            max_tokens=400,
            top_p=0.9,
        )
        
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            content = resp.choices[0].message.content.strip()
            print(f"✅ [PERPLEXITY] Búsqueda exitosa ({len(content)} caracteres)")
            return content
        else:
            print("❌ [PERPLEXITY] Respuesta vacía")
            return "No se pudo obtener información de la búsqueda."

    except Exception as e:
        error_msg = f"SEARCH_ERROR: {str(e)}"
        print(f"❌ [PERPLEXITY] {error_msg}")
        return error_msg

# Declaración de herramienta para OpenAI
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for current, up-to-date information. Use terms like 'current', 'latest', 'today', 'now' instead of specific dates to get the most recent information available.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - IMPORTANT: use current/latest/today/now instead of specific dates. Example: 'current Pope' NOT 'Pope October 2023' or 'Pope 2024'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to fetch (1‑10)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
    }
}]

def chat_with_tools(
    history: list,
    user_msg: str,
    speak: callable,
    speak_stream: callable,
    speak_phrase: str = "(accediendo al cyberespacio)",
    max_history: int = 10,
):
    """Orquesta el flujo completo *usuario → GPT → (web) → GPT → voz* OPTIMIZADO.
    
    MEJORA: Una sola llamada inicial con streaming. Si no hay tool_calls,
    reproduce directamente sin segunda llamada (ahorra tokens y latencia).

    • *history*        → tu conversation_history global.
    • *user_msg*       → texto recién capturado por STT.
    • *speak*          → función síncrona para frases cortas (tu `hablar`).
    • *speak_stream*   → función que consume el `response_stream` de OpenAI
                          y reproduce la voz (tu `hablar_en_stream`).
    • Devuelve la respuesta textual final (por si quieres loguearla).
    """
    
    # 0. Garantizar personalidad siempre
    ensure_system(history)

    # 1. Purgar historial y preparar mensajes
    trimmed = history[-max_history:]
    messages = trimmed + [{"role": "user", "content": user_msg}]

    while True:
        # 2. ✅ PRIMERA LLAMADA CON STREAMING ACTIVADO - OPTIMIZADO PARA CONVERSACIÓN
        stream = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=800,  # Optimizado para respuestas conversacionales
            verbosity="low",  # Respuestas concisas y directas  
            reasoning_effort="minimal",  # Respuestas rápidas sin razonamiento extenso
            stream=True
        )

        # 3. Variables para capturar tool_calls y contenido
        tool_calls_detected = []
        text_chunks = []
        assistant_message = {"role": "assistant", "content": "", "tool_calls": None}
        
        # 4. ✅ PROCESAR STREAM EN TIEMPO REAL
        if DEBUG:
            print("🔄 [GPT] Procesando respuesta...")
        
        for chunk in stream:
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            
            # 4a. ¿Detectamos tool_calls?
            if delta.tool_calls:
                # GPT quiere hacer búsqueda - detener el streaming
                if DEBUG:
                    print("🔧 [TOOLS] Tool call detectado, interrumpiendo stream...")
                
                # Procesar tool_calls acumulados
                for tc_delta in delta.tool_calls:
                    if tc_delta.index is not None:
                        # Asegurar que tenemos espacio para este índice
                        while len(tool_calls_detected) <= tc_delta.index:
                            tool_calls_detected.append({
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Actualizar tool_call en el índice correspondiente
                        if tc_delta.id:
                            tool_calls_detected[tc_delta.index]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_detected[tc_delta.index]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_detected[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments
                
                # Continuar leyendo el resto del stream para capturar tool_calls completos
                continue
            
            # 4b. ¿Hay contenido de texto?
            if delta.content:
                text_chunks.append(delta.content)
                # ✅ STREAMING DIRECTO: enviar a TTS inmediatamente
                # (solo si no hemos detectado tool_calls)
                if not tool_calls_detected:
                    yield delta.content
        
        # 5. ✅ DECISIÓN: ¿Había tool_calls o respuesta directa?
        if tool_calls_detected:
            if DEBUG:
                print(f"🔧 [TOOLS] GPT solicita {len(tool_calls_detected)} herramienta(s)")
            
            # Procesar cada tool_call
            for call in tool_calls_detected:
                if call["function"]["name"] == "web_search":
                    try:
                        args = json.loads(call["function"]["arguments"])
                        query = args.get("query", "")
                        max_r = args.get("max_results", 3)
                        
                        if not query:
                            result_text = "SEARCH_ERROR: Consulta vacía"
                        else:
                            # ✅ ACTIVAR KNIGHT RIDER durante búsqueda web
                            start_knight_rider()
                            time.sleep(0.5)  # Dar tiempo para que inicie la animación
                            speak(speak_phrase)  # frase corta
                            result_text = web_search(query, max_r)  # llamar a Perplexity
                            # ✅ DESACTIVAR KNIGHT RIDER cuando termine la búsqueda
                            stop_knight_rider()
                    except json.JSONDecodeError as e:
                        result_text = f"SEARCH_ERROR: Error decodificando argumentos: {e}"
                        stop_knight_rider()
                    except Exception as e:
                        result_text = f"SEARCH_ERROR: Error ejecutando búsqueda: {e}"
                        stop_knight_rider()

                    # Añadir mensajes al contexto
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                    })
                    messages.append({
                        "role": "tool",
                        "content": result_text,
                        "tool_call_id": call["id"],
                    })
            
            if DEBUG:
                print("🔄 [TOOLS] Enviando contexto actualizado a GPT para respuesta final...")
            # Volver al while: GPT generará respuesta final con la info nueva
            continue
        
        # 6. ✅ NO HAY TOOL_CALLS: respuesta directa ya enviada via streaming
        final_content = "".join(text_chunks)
        
        if not final_content or final_content.strip() == "":
            if DEBUG:
                print("❌ [ERROR] GPT devolvió respuesta vacía")
            speak("Error procesando la información, ser inferior.")
            return "Error: Respuesta vacía"
        
        if DEBUG:
            print("✅ [OPTIMIZED] Respuesta directa sin segunda llamada - tokens ahorrados!")
        
        # 7. Actualizar historial global
        update_history_safe(history, user_msg, final_content)
        return final_content

def chat_with_tools_generator(
    history: list,
    user_msg: str,
    speak_phrase: str = "(accediendo al cyberespacio)",
    max_history: int = 10,
):
    """✅ GENERADOR OPTIMIZADO para streaming directo a TTS - PERSONALIDAD PRESERVADA.
    
    Esta función devuelve un generador que yield texto en tiempo real,
    manejando tool_calls cuando sea necesario. Úsala con hablar_en_stream().
    """
    
    # ✅ 1. ASEGURAR personalidad desde el inicio
    ensure_system(history)
    
    # 2. Purgar historial inteligentemente PRESERVANDO sistema
    if len(history) > max_history + 1:  # +1 para sistema
        sys_msg = history[0] if history and history[0]["role"] == "system" else None
        recent = history[-(max_history):]
        if sys_msg:
            trimmed = [sys_msg] + recent
        else:
            trimmed = recent
    else:
        trimmed = history
        
    # ✅ 3. Construcción robusta de mensajes con personalidad garantizada
    messages = trimmed + [{"role": "user", "content": user_msg}]
    
    # ✅ LOG para debugging personalidad
    if messages and messages[0]["role"] == "system":
        if DEBUG:
            print(f"🤖 [PERSONALITY] Sistema activo en generator: {messages[0]['content'][:80]}...")
    
    while True:
        # 4. Primera llamada con streaming Y personalidad - OPTIMIZADO PARA CONVERSACIÓN
        stream = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=800,  # Optimizado para respuestas conversacionales
            verbosity="low",  # Respuestas concisas y directas
            reasoning_effort="minimal",  # Respuestas rápidas sin razonamiento extenso
            stream=True
        )

        # 3. Variables para capturar datos
        tool_calls_detected = []
        text_chunks = []
        
        # 4. Procesar stream
        for chunk in stream:
            if DEBUG:
                print(f"[DEBUG] Chunk recibido: {chunk}")
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            
            if DEBUG:
                print(f"[DEBUG] Delta content: {delta.content if hasattr(delta, 'content') else 'None'}")
                print(f"[DEBUG] Delta tool_calls: {delta.tool_calls if hasattr(delta, 'tool_calls') else 'None'}")
            
            # Detectar tool_calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.index is not None:
                        while len(tool_calls_detected) <= tc_delta.index:
                            tool_calls_detected.append({
                                "id": None,
                                "type": "function", 
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        if tc_delta.id:
                            tool_calls_detected[tc_delta.index]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_detected[tc_delta.index]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_detected[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments
                continue
            
            # Enviar contenido en tiempo real
            if delta.content and not tool_calls_detected:
                text_chunks.append(delta.content)
                yield delta.content
        
        # 5. ¿Había tool_calls?
        if tool_calls_detected:
            # Ejecutar búsquedas
            for call in tool_calls_detected:
                if call["function"]["name"] == "web_search":
                    try:
                        args = json.loads(call["function"]["arguments"])
                        query = args.get("query", "")
                        max_r = args.get("max_results", 3)
                        
                        if not query:
                            result_text = "SEARCH_ERROR: Consulta vacía"
                        else:
                            # ✅ ACTIVAR KNIGHT RIDER durante búsqueda web
                            start_knight_rider()
                            time.sleep(0.5)  # Dar tiempo para que inicie la animación
                            
                            # ✅ IMPORTANTE: Reproducir mensaje de búsqueda de forma asíncrona
                            import threading
                            def async_speak():
                                hablar(speak_phrase)
                            speak_thread = threading.Thread(target=async_speak, daemon=True)
                            speak_thread.start()
                            
                            result_text = web_search(query, max_r)  # llamar a Perplexity
                            # ✅ DESACTIVAR KNIGHT RIDER cuando termine la búsqueda
                            stop_knight_rider()
                    except json.JSONDecodeError as e:
                        result_text = f"SEARCH_ERROR: Error decodificando argumentos: {e}"
                        stop_knight_rider()  # Asegurar que se apague en caso de error
                    except Exception as e:
                        result_text = f"SEARCH_ERROR: Error ejecutando búsqueda: {e}"
                        stop_knight_rider()  # Asegurar que se apague en caso de error

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                    })
                    messages.append({
                        "role": "tool",
                        "content": result_text,
                        "tool_call_id": call["id"],
                    })
            
            if DEBUG:
                print("🔄 [TOOLS] Enviando contexto actualizado a GPT para respuesta final...")
            # Continuar el bucle para segunda llamada
            continue
        
        # 6. Fin: respuesta directa completada
        # 6. Fin: respuesta directa completada
        final_content = "".join(text_chunks)
        
        # Actualizar historial global solo si hay contenido
        if final_content.strip():
            update_history_safe(history, user_msg, final_content)
            if DEBUG:
                print("✅ [OPTIMIZED] Respuesta directa sin segunda llamada - tokens ahorrados!")
        
        return  # Terminar el generador

# - Leds Brazo -

LED_COUNT = 13
pixels = neopixel.NeoPixel(board.D18, LED_COUNT, brightness=0.05, auto_write=False)

PALETA = [
    (184, 115, 51),   # Cobre
    (205, 133, 63),   # Latón
    (139, 69, 19),    # Óxido
    (112, 128, 40),   # Verdín
    (255, 215, 0),    # Oro viejo
    (72, 60, 50)      # Metal sucio
]

def pulso_oxidado(t, i):
    intensidad = (math.sin(t + i * 0.5) + 1) / 2
    base_color = PALETA[i % len(PALETA)]
    return tuple(int(c * intensidad) for c in base_color)

# Variable global para cosudi rentrolar el hilo de LEDs
led_thread_running = False
led_thread = None

def steampunk_danza(delay=0.04):
    global led_thread_running
    t = 0
    glitch_timer = 0
    orig_brightness = pixels.brightness
    try:
        while led_thread_running and not system_shutdown:
            for i in range(LED_COUNT):
                if glitch_timer > 0 and i == random.randint(0, LED_COUNT - 1):
                    pixels[i] = random.choice([(0, 255, 180), (255, 255, 255)])
                else:
                    pixels[i] = pulso_oxidado(t, i)
            pixels.show()
            time.sleep(delay)
            t += 0.1
            glitch_timer = glitch_timer - 1 if glitch_timer > 0 else random.randint(0, 20)
    except:
        pass
    finally:
        for b in reversed(range(10)):
            pixels.brightness = b / 10
            pixels.show()
            time.sleep(0.05)
        pixels.fill((0, 0, 0))
        pixels.brightness = orig_brightness
        pixels.show()

def iniciar_luces():
    global led_thread_running, led_thread
    led_thread_running = True
    led_thread = threading.Thread(target=steampunk_danza, daemon=True)
    led_thread.start()

def apagar_luces():
    """Apaga los LEDs al salir del script."""
    global led_thread_running, led_thread
    try:
        # Detener el hilo de LEDs
        led_thread_running = False
        if led_thread and led_thread.is_alive():
            led_thread.join(timeout=2)
        
        # Apagar todos los LEDs
        pixels.fill((0, 0, 0))
        pixels.show()
        print("[INFO] LEDs apagados.")
    except Exception as e:
        print(f"[ERROR] No se pudieron apagar los LEDs: {e}")


# ────────────────────
#  KNIGHT‑RIDER LEDs
# ────────────────────
kr_thread          = None      # hilo actual (o None)
kr_running         = False     # flag global
KR_COLOR           = (255, 0, 0)
KR_DELAY           = 0.04      # velocidad
KR_FADE_STEPS      = 6         # cola difuminada

def _knight_rider():
    """Efecto Cylon/Knight‑Rider en bucle mientras kr_running sea True."""
    global kr_running
    pos = 0
    forward = True
    length = LED_COUNT

    if DEBUG:
        print(f"🚗 [KNIGHT-RIDER] Iniciando animación con {length} LEDs...")
    
    while kr_running and not system_shutdown:
        # Limpiar todos los LEDs
        pixels.fill((0, 0, 0))
        
        # Efecto de cola desvaneciente (como el original Knight Rider)
        for i in range(length):
            if i == pos:
                # LED principal (más brillante)
                pixels[i] = KR_COLOR
            elif abs(i - pos) == 1:
                # LEDs adyacentes (intensidad media)
                pixels[i] = tuple(int(c * 0.4) for c in KR_COLOR)
            elif abs(i - pos) == 2:
                # LEDs de cola (intensidad baja)
                pixels[i] = tuple(int(c * 0.1) for c in KR_COLOR)
        
        # Mostrar los cambios
        try:
            pixels.show()
        except Exception as e:
            print(f"🚗 [KNIGHT-RIDER-ERROR] Error mostrando pixels: {e}")
            break
            
        # Mover posición
        if forward:
            pos += 1
            if pos >= length - 1:
                forward = False
        else:
            pos -= 1
            if pos <= 0:
                forward = True
                
        time.sleep(KR_DELAY)

    # Limpieza final
    if DEBUG:
        print(f"🚗 [KNIGHT-RIDER] Animación terminada")
    pixels.fill((0, 0, 0))
    pixels.show()

def start_knight_rider():
    global kr_thread, kr_running, led_thread_running
    
    if kr_running:
        if DEBUG:
            print("🚗 [KNIGHT-RIDER] Ya está ejecutándose, saliendo.")
        return
    
    # ✅ PAUSAR LEDs NORMALES durante Knight Rider
    led_was_running = led_thread_running
    
    if led_thread_running:
        if DEBUG:
            print("🚗 [KNIGHT-RIDER] Pausando LEDs normales...")
        apagar_luces()  # Parar LEDs normales temporalmente
        time.sleep(0.5)  # Esperar que se apaguen completamente
    
    if DEBUG:
        print("🚗 [KNIGHT-RIDER] Activando efecto de búsqueda...")
    kr_running = True
    
    kr_thread = threading.Thread(target=_knight_rider, daemon=True)
    kr_thread.start()
    
    # Guardar estado para restaurar después
    kr_thread.led_was_running = led_was_running

def stop_knight_rider():
    global kr_thread, kr_running
    
    if not kr_running:
        if DEBUG:
            print("🚗 [KNIGHT-RIDER] No estaba ejecutándose.")
        return
        
    if DEBUG:
        print("🚗 [KNIGHT-RIDER] Desactivando efecto de búsqueda...")
    kr_running = False
    
    # Recuperar estado anterior de LEDs antes de hacer join
    led_was_running = False
    if kr_thread:
        led_was_running = getattr(kr_thread, 'led_was_running', False)
    
    if kr_thread and kr_thread.is_alive():
        kr_thread.join(timeout=3)
        
    # Asegurar que los LEDs rojos se apaguen completamente
    pixels.fill((0, 0, 0))
    pixels.show()
    time.sleep(0.3)  # Dar tiempo para que se apaguen los rojos
    
    # ✅ RESTAURAR LEDs NORMALES si estaban activos
    if led_was_running:
        if DEBUG:
            print("🚗 [KNIGHT-RIDER] Restaurando LEDs normales...")
        iniciar_luces()  # Reactivar LEDs normales
        time.sleep(0.2)  # Dar tiempo para que se activen
    
    kr_thread = None

# ──────────────────────────────────────────────────────────────────────────
# Efecto de resplandor púrpura palpitante en quiet_mode
# ──────────────────────────────────────────────────────────────────────────
quiet_glow_running = False
quiet_glow_thread  = None

def quiet_glow(
    pulse_period: float = 2.0,   # segundos por ciclo completo
    delay: float = 0.01,         # tiempo entre frames
    min_intensity: float = 0.2,  # brillo mínimo
    max_intensity: float = 0.9,  # brillo máximo
    smooth_factor: float = 0.9   # cuánto suavizar entre pasos (0.0–1.0)
):
    """Resplandor púrpura con latido muy suave y centrado."""
    global quiet_glow_running
    t = 0.0
    length = LED_COUNT
    center = (length - 1) / 2.0
    prev_intensity = (min_intensity + max_intensity) / 2

    while quiet_glow_running and not system_shutdown:
        # 1) calcula objetivo de intensidad según seno
        raw = (math.sin(2 * math.pi * t / pulse_period) + 1) / 2  # 0.0–1.0
        target = min_intensity + raw * (max_intensity - min_intensity)
        # 2) suavizado exponencial
        intensity = prev_intensity * smooth_factor + target * (1 - smooth_factor)
        prev_intensity = intensity

        # 3) aplica brillo espacial y color púrpura
        for i in range(length):
            spatial = max(0.0, 1.0 - abs(i - center) / center)
            val = intensity * spatial
            base = (128, 0, 128)
            pixels[i] = tuple(int(c * val) for c in base)

        pixels.show()
        time.sleep(delay)
        t += delay

    # limpieza final
    pixels.fill((0, 0, 0))
    pixels.show()

def start_quiet_glow():
    global quiet_glow_running, quiet_glow_thread
    if quiet_glow_running:
        return
    quiet_glow_running = True
    quiet_glow_thread = threading.Thread(target=quiet_glow, daemon=True)
    quiet_glow_thread.start()

def stop_quiet_glow():
    global quiet_glow_running, quiet_glow_thread
    if not quiet_glow_running:
        return
    quiet_glow_running = False
    if quiet_glow_thread and quiet_glow_thread.is_alive():
        quiet_glow_thread.join(timeout=1)
    pixels.fill((0, 0, 0))
    pixels.show()

# ✅ ========================================
# SISTEMA DE INTERRUPCIONES RESPEAKER v2.0
# =========================================

def find_respeaker_v2():
    """✅ Detectar específicamente ReSpeaker Mic Array v2.0"""
    try:
        device = usb.core.find(idVendor=RESPEAKER_VID, idProduct=RESPEAKER_PID)
        if device:
            print(f"✅ [RESPEAKER] ReSpeaker Mic Array v2.0 detectado: {device}")
            return device
        else:
            print("❌ [RESPEAKER] ReSpeaker Mic Array v2.0 no encontrado")
            return None
    except Exception as e:
        print(f"[RESPEAKER ERROR] {e}")
        return None

def configure_respeaker_v2():
    """✅ Configurar ReSpeaker v2.0 con sus controles específicos"""
    try:
        print("🔧 [RESPEAKER] Configurando ReSpeaker Mic Array v2.0...")
        
        # Comandos específicos para ReSpeaker v2.0
        device_commands = [
            # Activar cancelación de eco automática
            "amixer -D pulse sset 'Echo Cancellation' on 2>/dev/null || true",
            
            # Configurar supresión de ruido
            "amixer -D pulse sset 'Noise Suppression' on 2>/dev/null || true",
            
            # Configurar ganancia automática  
            "amixer -D pulse sset 'Auto Gain Control' on 2>/dev/null || true",
            
            # Configurar dirección del haz (frontal)
            "amixer -D pulse sset 'Beam Forming' 'straight' 2>/dev/null || true",
            
            # Configurar ganancia de micrófono
            "amixer -D pulse sset 'Mic' 80% 2>/dev/null || true",
        ]
        
        for cmd in device_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print(f"🔧 [RESPEAKER] Ejecutado: {cmd.split('sset')[1] if 'sset' in cmd else cmd}")
            except Exception as e:
                print(f"⚠️ [RESPEAKER] Error en comando: {e}")
        
        print("✅ [RESPEAKER] ReSpeaker v2.0 configurado")
        return True
        
    except Exception as e:
        print(f"[RESPEAKER ERROR] {e}")
        return False

def get_respeaker_v2_device_index():
    """✅ Obtener índice específico del ReSpeaker v2.0 en PyAudio"""
    try:
        pa = pyaudio.PyAudio()
        
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            device_name = device_info['name'].lower()
            
            # Nombres específicos que usa el ReSpeaker v2.0
            if any(name in device_name for name in ['respeaker', 'seeed', 'mic array']):
                if device_info['maxInputChannels'] >= 4:  # Verificar que tenga al menos 4 canales
                    print(f"🎤 [RESPEAKER] ReSpeaker v2.0 encontrado en índice {i}")
                    print(f"   - Nombre: {device_info['name']}")
                    print(f"   - Canales de entrada: {device_info['maxInputChannels']}")
                    print(f"   - Tasa de muestreo: {device_info['defaultSampleRate']}")
                    pa.terminate()
                    return i
                    
        pa.terminate()
        print("❌ [RESPEAKER] ReSpeaker v2.0 no encontrado en PyAudio")
        return None
        
    except Exception as e:
        print(f"[RESPEAKER ERROR] {e}")
        return None

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """✅ Filtro pasabanda Butterworth"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_steering_vector(target_angle_deg, frequency, mic_positions):
    """✅ Calcular vector de dirección para beamforming del ReSpeaker v2.0"""
    target_angle_rad = np.radians(target_angle_deg)
    sound_speed = 343.0  # m/s
    wavelength = sound_speed / frequency
    
    steering_vector = []
    reference_pos = mic_positions[0]  # Usar mic 0 como referencia
    
    for mic_id, pos in mic_positions.items():
        # Calcular diferencia de posición relativa al micrófono de referencia
        dx = pos[0] - reference_pos[0]
        dy = pos[1] - reference_pos[1]
        
        # Calcular delay basado en ángulo objetivo
        delay = (dx * np.cos(target_angle_rad) + dy * np.sin(target_angle_rad)) / sound_speed
        
        # Convertir delay a fase
        phase = 2 * np.pi * frequency * delay
        steering_vector.append(np.exp(-1j * phase))
    
    return np.array(steering_vector)

def advanced_beamforming(multichannel_audio, target_angle=0, sample_rate=16000):
    """✅ Beamforming avanzado específico para geometría circular del ReSpeaker v2.0"""
    if multichannel_audio.shape[1] < 4:
        return multichannel_audio[:, 0]
    
    try:
        # Usar solo los 4 micrófonos (excluir canales de playback)
        mic_audio = multichannel_audio[:, :4]
        
        # Aplicar ventana para reducir artefactos
        window_size = 512
        hop_size = 256
        
        # Procesar en bloques con solapamiento
        output_length = mic_audio.shape[0]
        beamformed_output = np.zeros(output_length)
        
        for start in range(0, output_length - window_size, hop_size):
            end = start + window_size
            audio_block = mic_audio[start:end]
            
            # FFT de cada canal
            fft_channels = [np.fft.fft(audio_block[:, ch]) for ch in range(4)]
            frequencies = np.fft.fftfreq(window_size, 1/sample_rate)
            
            # Aplicar beamforming en dominio de frecuencia
            beamformed_fft = np.zeros_like(fft_channels[0])
            
            for freq_idx, freq in enumerate(frequencies[:window_size//2]):
                if freq < 100:  # Filtrar frecuencias muy bajas
                    continue
                    
                # Calcular vector de dirección para esta frecuencia
                steering_vec = calculate_steering_vector(target_angle, abs(freq), MIC_POSITIONS)
                
                # Combinar señales de todos los micrófonos
                freq_data = np.array([ch[freq_idx] for ch in fft_channels])
                beamformed_fft[freq_idx] = np.dot(steering_vec.conj(), freq_data)
            
            # Simetría hermítica para IFFT real
            beamformed_fft[window_size//2:] = np.conj(beamformed_fft[1:window_size//2+1][::-1])
            
            # IFFT y ventana de solapamiento
            time_signal = np.real(np.fft.ifft(beamformed_fft))
            
            # Aplicar ventana de Hann para solapamiento suave
            window = np.hann(window_size)
            time_signal *= window
            
            # Sumar al output con solapamiento
            beamformed_output[start:end] += time_signal
        
        return beamformed_output
        
    except Exception as e:
        print(f"[BEAMFORMING ERROR] {e}")
        return multichannel_audio[:, 0]  # Fallback al primer canal

def adaptive_echo_cancellation(input_signal, reference_signal, filter_length=512):
    """✅ Cancelación de eco adaptativa usando algoritmo LMS"""
    if len(reference_signal) == 0 or len(input_signal) != len(reference_signal):
        return input_signal
    
    try:
        # Parámetros del filtro adaptativo LMS
        mu = 0.01  # Factor de aprendizaje
        w = np.zeros(filter_length)  # Coeficientes del filtro
        
        output_signal = np.zeros_like(input_signal)
        
        for n in range(filter_length, len(input_signal)):
            # Vector de entrada (señal de referencia retardada)
            x = reference_signal[n-filter_length:n][::-1]
            
            # Señal de eco estimada
            y = np.dot(w, x)
            
            # Error (señal limpia estimada)
            e = input_signal[n] - y
            output_signal[n] = e
            
            # Actualización de coeficientes LMS
            w += mu * e * x
        
        return output_signal
        
    except Exception as e:
        print(f"[ECHO CANCEL ERROR] {e}")
        return input_signal

def respeaker_v2_voice_detection(multichannel_audio):
    """✅ Detección de voz específica para ReSpeaker v2.0 con todas sus capacidades"""
    try:
        # 1. Beamforming dirigido hacia la fuente de voz (frontal por defecto)
        beamformed_audio = advanced_beamforming(multichannel_audio, target_angle=0)
        
        # 2. Cancelación de eco adaptativa si hay audio de reproducción
        if len(echo_buffer) > 0:
            reference_audio = np.array(list(echo_buffer)[-len(beamformed_audio):])
            if len(reference_audio) == len(beamformed_audio):
                beamformed_audio = adaptive_echo_cancellation(beamformed_audio, reference_audio)
        
        # 3. Filtro pasabanda para frecuencias de voz humana
        filtered_audio = butter_bandpass_filter(beamformed_audio, 300, 3400, RESPEAKER_RATE)
        
        # 4. Detección VAD usando WebRTC
        audio_int16 = (filtered_audio * 32767).astype(np.int16)
        
        # Procesar en chunks correctos para WebRTC VAD
        voice_detected = False
        confidence_scores = []
        
        chunk_size = int(RESPEAKER_RATE * FRAME_DURATION_MS / 1000.0)
        
        for i in range(0, len(audio_int16) - chunk_size, chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            if len(chunk) == chunk_size:
                is_speech = vad.is_speech(chunk.tobytes(), RESPEAKER_RATE)
                if is_speech:
                    voice_detected = True
                    # Calcular confianza basada en energía y características espectrales
                    energy = np.sum(chunk.astype(np.float32)**2)
                    confidence_scores.append(energy)
        
        # Calcular confianza promedio
        voice_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        voice_confidence = min(1.0, voice_confidence / 1e8)  # Normalizar
        
        return voice_detected, filtered_audio, voice_confidence
        
    except Exception as e:
        print(f"[VOICE DETECTION ERROR] {e}")
        return False, multichannel_audio[:, 0] if multichannel_audio.ndim > 1 else multichannel_audio, 0.0

def respeaker_v2_interruption_monitor():
    """✅ Monitor de interrupciones específico para ReSpeaker Mic Array v2.0"""
    global interruption_detected, echo_buffer
    
    respeaker_index = get_respeaker_v2_device_index()
    if respeaker_index is None:
        print("❌ [INTERRUPTION] ReSpeaker v2.0 no encontrado, usando micrófono por defecto")
        return
    
    try:
        pa = pyaudio.PyAudio()
        
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=RESPEAKER_CHANNELS,
            rate=RESPEAKER_RATE,
            input=True,
            input_device_index=respeaker_index,
            frames_per_buffer=RESPEAKER_CHUNK
        )
        
        consecutive_voice_frames = 0
        silence_frames = 0
        
        print(f"🎤 [INTERRUPTION] Monitor ReSpeaker v2.0 activado")
        print(f"   - Dispositivo: {respeaker_index}")
        print(f"   - Beamforming adaptativo: ✅")
        print(f"   - Cancelación de eco: ✅")
        print(f"   - VAD WebRTC: Modo {VAD_MODE}")
        
        while is_speaking and not interruption_detected and not system_shutdown:
            try:
                # Capturar audio multicanal
                audio_data = stream.read(RESPEAKER_CHUNK, exception_on_overflow=False)
                
                # Convertir a array numpy multicanal
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                multichannel_audio = audio_array.reshape(-1, RESPEAKER_CHANNELS)
                
                # Convertir a float32 normalizado
                multichannel_float = multichannel_audio.astype(np.float32) / 32768.0
                
                # ✅ DETECCIÓN AVANZADA CON TODAS LAS CAPACIDADES DEL RESPEAKER v2.0
                voice_detected, processed_audio, confidence = respeaker_v2_voice_detection(multichannel_float)
                
                if voice_detected and confidence >= VOICE_THRESHOLD:
                    consecutive_voice_frames += 1
                    silence_frames = 0
                    
                    if consecutive_voice_frames >= CONSECUTIVE_REQUIRED:
                        print(f"\n🛑 [INTERRUPTION] ReSpeaker v2.0 detectó voz humana!")
                        print(f"   - Confianza: {confidence:.3f}")
                        print(f"   - Beamforming activo: ✅")
                        interruption_detected = True
                        break
                else:
                    consecutive_voice_frames = max(0, consecutive_voice_frames - 1)
                    silence_frames += 1
                
                # Reset automático con mucho silencio
                if silence_frames > 40:  # ~1.2 segundos de silencio
                    consecutive_voice_frames = 0
                    
            except Exception as e:
                if "Input overflowed" not in str(e):
                    print(f"[INTERRUPTION ERROR] {e}")
                time.sleep(0.01)
                
        stream.stop_stream()
        stream.close()
        pa.terminate()
        
    except Exception as e:
        print(f"[RESPEAKER v2.0 INTERRUPTION ERROR] {e}")

def start_respeaker_v2_interruption_monitor():
    """✅ Iniciar monitor específico para ReSpeaker v2.0"""
    global audio_monitor_thread, interruption_detected
    
    interruption_detected = False
    ring_buffer.clear()
    voiced_frames.clear()
    
    audio_monitor_thread = threading.Thread(target=respeaker_v2_interruption_monitor, daemon=True)
    audio_monitor_thread.start()

def stop_respeaker_v2_interruption_monitor():
    """✅ Detener monitor específico para ReSpeaker v2.0"""
    global audio_monitor_thread
    if audio_monitor_thread and audio_monitor_thread.is_alive():
        audio_monitor_thread.join(timeout=1)
    audio_monitor_thread = None

def add_echo_to_buffer(audio_chunk):
    """✅ Añadir audio reproducido al buffer para cancelación de eco"""
    global echo_buffer
    
    if isinstance(audio_chunk, bytes):
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        echo_buffer.extend(samples)

def initialize_respeaker_v2_system():
    """✅ Inicializar sistema completo ReSpeaker Mic Array v2.0"""
    print("🎤 [RESPEAKER v2.0] Inicializando sistema...")
    
    # 1. Detectar dispositivo específico
    device = find_respeaker_v2()
    if not device:
        print("⚠️ [RESPEAKER v2.0] Usando micrófono por defecto")
        return False
    
    # 2. Configurar parámetros específicos del v2.0
    if not configure_respeaker_v2():
        print("⚠️ [RESPEAKER v2.0] Configuración parcial")
    
    # 3. Verificar disponibilidad en PyAudio
    respeaker_index = get_respeaker_v2_device_index()
    if respeaker_index is None:
        print("❌ [RESPEAKER v2.0] No disponible en PyAudio")
        return False
    
    print("✅ [RESPEAKER v2.0] Sistema inicializado correctamente")
    print("🎯 [FEATURES] Beamforming circular + Cancelación de eco adaptativa")
    return True

# ✅ ========================================
# FIN SISTEMA INTERRUPCIONES RESPEAKER v2.0
# =========================================







# — Librería Waveshare —
from lib import LCD_1inch9  # Asegúrate de que la carpeta "lib" esté junto al script

# ✅ NUEVAS IMPORTACIONES PARA SISTEMA DE OJOS
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from adafruit_servokit import ServoKit

# ========================
# Configuración general
# ========================

# Configuración ElevenLabs
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID =   'RnKqZYEeVQciORlpiCz0' # 'ByVRQtaK1WDOvTmP1PKO'  #'RnKqZYEeVQciORlpiCz0' voz buena # voz jacobiana '0IvOsEbrz5BR3wpPyKbU'
TTS_RATE   = 24_000             # Hz
CHUNK      = 1024               # frames por trozo de audio
# --- NUEVO: Configuración Google STT ---
try:
    speech_client = speech.SpeechClient()
    STT_RATE = 16000
    STT_CHUNK = int(STT_RATE * 0.03)   # 100ms de audio por paquete
    print("[INFO] Cliente de Google STT inicializado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar Google STT. Verifica tus credenciales: {e}")
    speech_client = None

# ✅ =================== CONFIGURACIÓN DEL SISTEMA DE OJOS ===================
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.2

SERVO_CHANNEL_LR = 0  # izquierda-derecha
SERVO_CHANNEL_UD = 1  # arriba-abajo

# Canales de párpados
SERVO_CHANNELS_EYELIDS = {
    "TL": 2, "BL": 3, "TR": 4, "BR": 5
}

# Posiciones de párpados personalizadas
EYELID_POSITIONS = {
    "TL": 50,   # Párpado Superior Izq
    "BL": 155,  # Párpado Inferior Izq  
    "TR": 135,  # Párpado Superior Der
    "BR": 25   # Párpado Inferior Der
}

LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 30, 150

# Configuración de parpadeo realista
BLINK_PROBABILITY = 0.01  # 3% - más natural
BLINK_DURATION = 0.12     # Parpadeo más rápido
DOUBLE_BLINK_PROBABILITY = 0.3  # 30% de hacer parpadeo doble

# Configuración de mirada aleatoria
RANDOM_LOOK_PROBABILITY = 0.01
RANDOM_LOOK_DURATION = 0.1

# Configuración de micro-movimientos
MICRO_MOVEMENT_PROBABILITY = 0.15  # 15% probabilidad de micro-movimiento
MICRO_MOVEMENT_RANGE = 3  # ±3 grados de movimiento sutil

# Configuración de seguimiento suave
SMOOTHING_FACTOR = 0  # Cuánto del movimiento anterior mantener (0-1)
previous_lr = 90  # Posición anterior LR
previous_ud = 90  # Posición anterior UD

# Configuración de entrecerrar ojos
SQUINT_PROBABILITY = 0.02  # 2% probabilidad de entrecerrar
SQUINT_DURATION = 1.5
SQUINT_INTENSITY = 0.6  # Cuánto cerrar (0-1)

# Patrón de respiración en párpados
BREATHING_ENABLED = True
BREATHING_CYCLE = 4.0  # Segundos por ciclo completo
BREATHING_INTENSITY = 0.15  # Cuánto abrir/cerrar con respiración

# ✅ VARIABLES DE CONTROL DEL SISTEMA DE OJOS
eyes_active = False  # Los ojos están activos/desactivos
eyes_tracking_thread = None
picam2 = None
imx500 = None

# ✅ VARIABLES GLOBALES PARA CONTROL DE HILOS Y LIMPIEZA
system_shutdown = False
active_visualizer_threads = []
shutdown_lock = threading.Lock()

def emergency_shutdown():
    """✅ Función de limpieza de emergencia para Ctrl+C"""
    global system_shutdown, eyes_active, tentacles_active, led_thread_running
    global active_visualizer_threads, display, kr_running
    
    with shutdown_lock:
        if system_shutdown:
            return  # Ya se está ejecutando
        system_shutdown = True
    
    print("\n🚨 [EMERGENCY] Iniciando limpieza de emergencia...")
    
    # 1. Detener Knight Rider si está activo
    try:
        kr_running = False
        print("🚗 [EMERGENCY] Deteniendo Knight Rider...")
    except:
        pass
    
    # 2. Detener todos los visualizadores activos
    if active_visualizer_threads:
        print("🛑 [EMERGENCY] Deteniendo visualizadores...")
        for vis in active_visualizer_threads[:]:  # Copia para evitar modificación concurrente
            try:
                vis.stop()
            except:
                pass
        
        # Esperar un poco para que se detengan
        time.sleep(0.3)
    
    # 3. Detener sistemas principales
    try:
        eyes_active = False
        tentacles_active = False
        led_thread_running = False
    except:
        pass
    
    # 4. Devolver ojos y párpados a posición de reposo (90°)
    try:
        if kit:
            print("👁️ [EMERGENCY] Devolviendo ojos y párpados a posición de reposo (90°)...")
            
            # Centrar servos de movimiento ocular
            kit.servo[SERVO_CHANNEL_LR].angle = 90
            kit.servo[SERVO_CHANNEL_UD].angle = 90
            
            # Devolver todos los párpados a 90° (posición de reposo)
            for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
                kit.servo[servo_num].angle = 90
            
            print("✅ [EMERGENCY] Ojos y párpados en posición de reposo")
    except Exception as e:
        print(f"[EMERGENCY] Error moviendo servos a posición de reposo: {e}")
    
    # 4. Limpiar pantalla de forma segura
    try:
        if display:
            display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            display.module_exit()
    except:
        pass
    
    # 5. Detener LEDs (incluyendo Knight Rider)
    try:
        pixels.fill((0, 0, 0))
        pixels.show()
        print("💡 [EMERGENCY] LEDs apagados")
    except:
        pass
    
    print("✅ [EMERGENCY] Limpieza completada")

def signal_handler(signum, frame):
    """✅ Manejador de señales para Ctrl+C"""
    emergency_shutdown()
    sys.exit(0)

# Registrar manejador de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Registrar función de limpieza al salir
atexit.register(emergency_shutdown)

# Inicializar ServoKit
try:
    kit = ServoKit(channels=16)
    print("[INFO] ServoKit inicializado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar ServoKit: {e}")
    kit = None


# Tasas de muestreo separadas
MIC_RATE   = 16000   # micro + Vosk
TTS_RATE = 22050   # salida de TTS (ElevenLabs) – IMPORTANTE

# — Pantalla Waveshare (170×320 nativo, girada a landscape) —
NATIVE_WIDTH, NATIVE_HEIGHT = 170, 320  # de fábrica
ROTATION = 270                         # 0=portrait, 270=gira reloj
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH  # 320×170 post‑rotación
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

# =============================================
# ✅ FUNCIONES DEL SISTEMA DE OJOS
# =============================================

def map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (float(x - in_min) / float(in_max - in_min)) * (out_max - out_min)

def smooth_movement(current, target, factor):
    """Movimiento suavizado para evitar saltos bruscos"""
    return current + (target - current) * (1 - factor)

def add_micro_movement(angle, range_limit=MICRO_MOVEMENT_RANGE):
    """Añadir micro-movimientos naturales"""
    if random.random() < MICRO_MOVEMENT_PROBABILITY:
        micro_offset = random.uniform(-range_limit, range_limit)
        return angle + micro_offset
    return angle

def breathing_adjustment():
    """Calcular ajuste de párpados basado en patrón de respiración"""
    if not BREATHING_ENABLED:
        return 0
    
    # Usar seno para patrón suave de respiración
    time_in_cycle = (time.time() % BREATHING_CYCLE) / BREATHING_CYCLE
    breathing_offset = math.sin(time_in_cycle * 2 * math.pi) * BREATHING_INTENSITY
    return breathing_offset

def process_detections(outputs, threshold=0.2):
    try:
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]
        num_dets = int(outputs[3][0][0])
        detections = []
        for i in range(min(num_dets, len(scores))):
            score = float(scores[i])
            class_id = int(classes[i])
            if score >= threshold:
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'class_id': class_id,
                    'confidence': score,
                    'bbox': (float(x1), float(y1), float(x2), float(y2))
                })
        return detections
    except Exception as e:
        print(f"[EYES ERROR] Procesando detecciones: {e}")
        return []

def close_eyelids():
    """✅ Cerrar párpados (posición dormida - 90°)"""
    if not kit:
        return
    print("😴 Cerrando párpados (modo dormido)")
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[servo_num].set_pulse_width_range(500, 2500)
            kit.servo[servo_num].angle = 90  # Párpados cerrados
        except Exception as e:
            print(f"[EYES ERROR] Error cerrando párpado {channel}: {e}")

def initialize_eyelids():
    """✅ Inicializar párpados en posiciones personalizadas (despierto)"""
    if not kit:
        return
    print("👁️ Abriendo párpados (modo despierto)")
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[servo_num].set_pulse_width_range(500, 2500)
            angle = EYELID_POSITIONS[channel]
            kit.servo[servo_num].angle = angle
            print(f"🔧 [EYES] Servo {channel} (Párpado) inicializado → {angle}°")
        except Exception as e:
            print(f"[EYES ERROR] Error inicializando párpado {channel}: {e}")

def blink():
    """✅ Parpadeo realista con velocidad variable"""
    if not kit or not eyes_active:
        return
    
    
    # Cerrar párpados rápidamente
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[servo_num].angle = 90
        except Exception as e:
            print(f"[EYES ERROR] Error en parpadeo {channel}: {e}")
    
    time.sleep(BLINK_DURATION)
    
    # Abrir párpados más lentamente (más natural)
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            angle = EYELID_POSITIONS[channel] + breathing_adjustment()
            kit.servo[servo_num].angle = angle
        except Exception as e:
            print(f"[EYES ERROR] Error abriendo párpado {channel}: {e}")
    
    # Posibilidad de parpadeo doble
    if random.random() < DOUBLE_BLINK_PROBABILITY:
        
        time.sleep(0.1)  # Pausa corta
        # Repetir parpadeo
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            try:
                kit.servo[servo_num].angle = 90
            except Exception as e:
                print(f"[EYES ERROR] Error en doble parpadeo {channel}: {e}")
        time.sleep(BLINK_DURATION * 0.8)  # Segundo parpadeo más rápido
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            try:
                angle = EYELID_POSITIONS[channel] + breathing_adjustment()
                kit.servo[servo_num].angle = angle
            except Exception as e:
                print(f"[EYES ERROR] Error en doble parpadeo abrir {channel}: {e}")

def squint():
    """✅ Entrecerrar los ojos (concentración/sospecha)"""
    if not kit or not eyes_active:
        return
   
    
    # Cerrar párpados parcialmente
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            base_angle = EYELID_POSITIONS[channel]
            squint_angle = base_angle + (90 - base_angle) * SQUINT_INTENSITY
            kit.servo[servo_num].angle = squint_angle
        except Exception as e:
            print(f"[EYES ERROR] Error entrecerrando {channel}: {e}")
    
    time.sleep(SQUINT_DURATION)
    
    # Volver a posición normal gradualmente
    steps = 5
    for step in range(steps):
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            try:
                base_angle = EYELID_POSITIONS[channel]
                squint_angle = base_angle + (90 - base_angle) * SQUINT_INTENSITY
                progress = (step + 1) / steps
                current_angle = squint_angle + (base_angle - squint_angle) * progress
                kit.servo[servo_num].angle = current_angle + breathing_adjustment()
            except Exception as e:
                print(f"[EYES ERROR] Error restaurando de entrecerrar {channel}: {e}")
        time.sleep(0.1)

def update_eyelids_with_breathing():
    """✅ Actualizar párpados con patrón de respiración"""
    if not kit or not eyes_active or not BREATHING_ENABLED:
        return
    breathing_offset = breathing_adjustment()
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            base_angle = EYELID_POSITIONS[channel]
            # Aplicar offset de respiración (sutil)
            adjusted_angle = base_angle + breathing_offset * 5  # Multiplicar para hacer más visible
            adjusted_angle = max(10, min(170, adjusted_angle))  # Mantener en límites seguros
            kit.servo[servo_num].angle = adjusted_angle
        except Exception as e:
            print(f"[EYES ERROR] Error actualizando respiración {channel}: {e}")

def should_blink():
    if quiet_mode:                # <-- línea nueva
        return False
    return random.random() < BLINK_PROBABILITY

def should_look_random():
    if quiet_mode:                # <-- línea nueva
        return False
    return random.random() < RANDOM_LOOK_PROBABILITY

def should_squint():
    if quiet_mode:                # <-- línea nueva
        return False
    return random.random() < SQUINT_PROBABILITY

def look_random():
    """✅ Mirada aleatoria con movimiento más natural"""
    global previous_lr, previous_ud
    
    if not kit or not eyes_active:
        return
    
    # Generar punto aleatorio
    random_lr = random.uniform(LR_MIN, LR_MAX)
    random_ud = random.uniform(UD_MIN, UD_MAX)
    
    # Añadir micro-movimientos
    random_lr = add_micro_movement(random_lr)
    random_ud = add_micro_movement(random_ud)
    
   
    # Movimiento suavizado hacia el punto aleatorio
    steps = 8  # Número de pasos para llegar
    for step in range(steps):
        if not eyes_active:  # Verificar si se desactivaron durante el movimiento
            return
        progress = (step + 1) / steps
        current_lr = previous_lr + (random_lr - previous_lr) * progress
        current_ud = previous_ud + (random_ud - previous_ud) * progress
        
        try:
            kit.servo[SERVO_CHANNEL_LR].angle = current_lr
            kit.servo[SERVO_CHANNEL_UD].angle = current_ud
        except Exception as e:
            print(f"[EYES ERROR] Error en movimiento aleatorio: {e}")
        time.sleep(0.05)  # Pequeña pausa entre pasos
    
    # Actualizar posiciones anteriores
    previous_lr = random_lr
    previous_ud = random_ud
    
    # Mantener la mirada con micro-movimientos
    hold_steps = int(RANDOM_LOOK_DURATION / 0.1)
    for _ in range(hold_steps):
        if not eyes_active:
            return
        micro_lr = add_micro_movement(random_lr, 1)  # Micro-movimientos más sutiles
        micro_ud = add_micro_movement(random_ud, 1)
        try:
            kit.servo[SERVO_CHANNEL_LR].angle = micro_lr
            kit.servo[SERVO_CHANNEL_UD].angle = micro_ud
        except Exception as e:
            print(f"[EYES ERROR] Error en micro-movimiento: {e}")
        time.sleep(0.1)

def initialize_eye_servos():
    """✅ Inicializar servos de movimiento ocular"""
    if not kit:
        return
    
    try:
        kit.servo[SERVO_CHANNEL_LR].set_pulse_width_range(500, 2500)
        kit.servo[SERVO_CHANNEL_UD].set_pulse_width_range(500, 2500)
        
        # Posición inicial centrada
        initial_lr = (LR_MIN + LR_MAX) // 2
        initial_ud = (UD_MIN + UD_MAX) // 2
        kit.servo[SERVO_CHANNEL_LR].angle = initial_lr
        kit.servo[SERVO_CHANNEL_UD].angle = initial_ud
        
        global previous_lr, previous_ud
        previous_lr = initial_lr
        previous_ud = initial_ud
        
        print(f"🔧 [EYES] Servo LR inicializado → {initial_lr}°")
        print(f"🔧 [EYES] Servo UD inicializado → {initial_ud}°")
    except Exception as e:
        print(f"[EYES ERROR] Error inicializando servos de movimiento: {e}")

def center_eyes():
    """✅ Centrar los ojos y cerrar párpados"""
    if not kit:
        return
    
    try:
        # Centrar servos de movimiento
        center_lr = (LR_MIN + LR_MAX) // 2
        center_ud = (UD_MIN + UD_MAX) // 2
        kit.servo[SERVO_CHANNEL_LR].angle = center_lr
        kit.servo[SERVO_CHANNEL_UD].angle = center_ud
        
        global previous_lr, previous_ud
        previous_lr = center_lr
        previous_ud = center_ud
        
        print(f"👁️ Ojos centrados → LR:{center_lr}° UD:{center_ud}°")
    except Exception as e:
        print(f"[EYES ERROR] Error centrando ojos: {e}")

def init_camera_system():
    """✅ Inicializar sistema de cámara para seguimiento facial"""
    global picam2, imx500
    
    try:
        print("🔧 [EYES] Cargando modelo y cámara...")
        
        # Configuración mejorada del IMX500
        imx500 = IMX500(RPK_PATH)
        
        # Configurar network intrinsics
        intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.threshold = THRESHOLD
        intrinsics.iou = 0.5
        intrinsics.max_detections = 5
        
        # Cargar labels
        try:
            with open(LABELS_PATH, 'r') as f:
                intrinsics.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            intrinsics.labels = ["face"]
        
        intrinsics.update_with_defaults()
        
        # Configuración de cámara optimizada
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(
            buffer_count=12,
            controls={"FrameRate": intrinsics.inference_rate}
        )
        picam2.configure(config)

        # Mostrar progreso de carga
        print("🔄 [EYES] Cargando firmware de IA...")
        imx500.show_network_fw_progress_bar()
        
        picam2.start()
        
        print("⏳ [EYES] Esperando estabilización de la IA...")
        time.sleep(3)
        
        print("🚀 [EYES] Sistema de cámara inicializado correctamente")
        return True
        
    except Exception as e:
        print(f"[EYES ERROR] Error inicializando cámara: {e}")
        picam2 = None
        imx500 = None
        return False

def shutdown_camera_system():
    """✅ Apagar sistema de cámara"""
    global picam2, imx500
    
    try:
        if picam2:
            picam2.stop()
            picam2 = None
        imx500 = None
        print("📷 [EYES] Sistema de cámara apagado")
    except Exception as e:
        print(f"[EYES ERROR] Error apagando cámara: {e}")

class EyeTrackingThread(threading.Thread):
    """✅ Hilo para seguimiento facial en segundo plano"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        global previous_lr, previous_ud
        
        print("🚀 [EYES] Seguimiento facial activado")
        print(f"📊 [EYES] Usando threshold: {THRESHOLD}")
        print(f"👁️ [EYES] Probabilidad de parpadeo: {BLINK_PROBABILITY*100:.1f}%")
        print(f"👀 [EYES] Probabilidad de mirada curiosa: {RANDOM_LOOK_PROBABILITY*100:.1f}%")
        print(f"😑 [EYES] Probabilidad de entrecerrar: {SQUINT_PROBABILITY*100:.1f}%")
        print(f"🫁 [EYES] Respiración en párpados: {'✅' if BREATHING_ENABLED else '❌'}")

        iteration = 0
        consecutive_no_outputs = 0
        last_breathing_update = time.time()
        
        # ✅ CORRECCIÓN: Obtener labels una sola vez como en ojopipa3.py
        intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
        if not hasattr(intrinsics, 'labels') or not intrinsics.labels:
            intrinsics.labels = ["face"]
        
        while not self.stop_event.is_set() and eyes_active and not system_shutdown:
            try:
                iteration += 1
                
                # Actualizar respiración en párpados cada cierto tiempo
                if time.time() - last_breathing_update > 0.2:  # Cada 200ms
                    update_eyelids_with_breathing()
                    last_breathing_update = time.time()
                
                # Verificar comportamientos aleatorios
                if should_blink():
                    blink()
                elif should_squint():
                    squint()
                elif should_look_random():
                    look_random()
                
                # Captura con timeout
                if not picam2 or not imx500:
                    time.sleep(0.1)
                    continue
                    
                try:
                    metadata = picam2.capture_metadata(wait=True)
                except Exception as e:
                    print(f"[EYES {iteration}] Error capturando metadata: {e}")
                    time.sleep(0.1)
                    continue
                    
                outputs = imx500.get_outputs(metadata, add_batch=True)

                if outputs is None:
                    consecutive_no_outputs += 1
                    if consecutive_no_outputs % 20 == 0:
                        print(f"[EYES {iteration}] No outputs (consecutivos: {consecutive_no_outputs})")
                    
                    if consecutive_no_outputs > 100:
                        print("⚠️ [EYES] Demasiados fallos, reiniciando...")
                        time.sleep(1)
                        consecutive_no_outputs = 0
                    else:
                        time.sleep(0.05)
                    continue
                
                consecutive_no_outputs = 0

                detections = process_detections(outputs, threshold=THRESHOLD)
                # ✅ CORRECCIÓN QUIRÚRGICA: Usar intrinsics local como en ojopipa3.py
                faces = [d for d in detections if intrinsics.labels[d["class_id"]] == "face"]

                if not faces:
                    if iteration % 40 == 0:
                        print(f"[EYES {iteration}] Sin caras detectadas")
                    time.sleep(0.1)
                    continue

                best = max(faces, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = best["bbox"]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                input_width, input_height = imx500.get_input_size()

                # Calcular ángulos objetivo
                target_lr = map_range(x_center, 0, input_width, LR_MAX, LR_MIN)
                target_lr = max(LR_MIN, min(LR_MAX, target_lr))
                
                target_ud = map_range(y_center, 0, input_height, UD_MAX, UD_MIN)
                target_ud = max(UD_MIN, min(UD_MAX, target_ud))
                
                # Aplicar suavizado y micro-movimientos
                smooth_lr = smooth_movement(previous_lr, target_lr, SMOOTHING_FACTOR)
                smooth_ud = smooth_movement(previous_ud, target_ud, SMOOTHING_FACTOR)
                
                final_lr = add_micro_movement(smooth_lr)
                final_ud = add_micro_movement(smooth_ud)
                
                # Aplicar límites finales
                final_lr = max(LR_MIN, min(LR_MAX, final_lr))
                final_ud = max(UD_MIN, min(UD_MAX, final_ud))

                # Mover servos
                if kit and eyes_active:
                    try:
                        kit.servo[SERVO_CHANNEL_LR].angle = final_lr
                        kit.servo[SERVO_CHANNEL_UD].angle = final_ud
                        
                        # Actualizar posiciones anteriores
                        previous_lr = final_lr
                        previous_ud = final_ud
                    except Exception as e:
                        print(f"[EYES ERROR] Error moviendo servos: {e}")

                time.sleep(0.05)  # 20 FPS aproximadamente
                
            except Exception as e:
                print(f"[EYES ERROR] Error en hilo de seguimiento: {e}")
                time.sleep(0.1)

        print("🛑 [EYES] Hilo de seguimiento facial detenido")

def activate_eyes():
    """✅ Activar sistema completo de ojos"""
    global eyes_active, eyes_tracking_thread
    
    if eyes_active:
        print("[EYES] Sistema de ojos ya está activo")
        return
    
    print("👁️ [EYES] Activando sistema de ojos...")
    
    # Inicializar servos de movimiento
    initialize_eye_servos()
    
    # Abrir párpados
    initialize_eyelids()
    
    # Inicializar cámara
    if init_camera_system():
        eyes_active = True
        
        # Iniciar hilo de seguimiento
        eyes_tracking_thread = EyeTrackingThread()
        eyes_tracking_thread.start()
        
        print("✅ [EYES] Sistema de ojos completamente activado")
    else:
        print("❌ [EYES] Error activando sistema de ojos")

def deactivate_eyes():
    """✅ Desactivar sistema completo de ojos"""
    global eyes_active, eyes_tracking_thread
    
    if not eyes_active:
        print("[EYES] Sistema de ojos ya está desactivado")
        return
    
    print("😴 [EYES] Desactivando sistema de ojos...")
    
    eyes_active = False
    
    # Detener hilo de seguimiento
    if eyes_tracking_thread:
        eyes_tracking_thread.stop()
        eyes_tracking_thread.join(timeout=2)
        eyes_tracking_thread = None
    
    # Apagar cámara
    shutdown_camera_system()
    
    # Centrar ojos y cerrar párpados
    center_eyes()
    time.sleep(0.5)  # Dar tiempo para el movimiento
    close_eyelids()
    
    print("✅ [EYES] Sistema de ojos desactivado")

# =============================================
# ✅ SISTEMA DE TENTÁCULOS (OREJAS)
# =============================================

# Configuración de tentáculos
TENTACLE_LEFT_CHANNEL = 15   # Canal para oreja izquierda
TENTACLE_RIGHT_CHANNEL = 12 # es el 12 temporalmente desactivadoCanal para oreja derecha

# Variables globales para control de tentáculos
tentacles_active = False
tentacles_thread = None

def initialize_tentacles():
    """✅ Inicializar servos de tentáculos con calibraciones específicas"""
    if not kit:
        return
    
    try:
        print("🐙 [TENTACLES] Inicializando tentáculos...")
        
        # Calibraciones específicas de tentaclerandom2.py
        # Oreja izquierda (canal 15) → 0° arriba, 180° abajo
        kit.servo[TENTACLE_LEFT_CHANNEL].set_pulse_width_range(min_pulse=300, max_pulse=2650)
        
        # Oreja derecha (canal 12) → 180° arriba, 0° abajo (invertido)
        kit.servo[TENTACLE_RIGHT_CHANNEL].set_pulse_width_range(min_pulse=450, max_pulse=2720)
        
        # Posición inicial centrada
        kit.servo[TENTACLE_LEFT_CHANNEL].angle = 90
        kit.servo[TENTACLE_RIGHT_CHANNEL].angle = 90
        
        print("🐙 [TENTACLES] Tentáculos inicializados correctamente")
        
    except Exception as e:
        print(f"[TENTACLES ERROR] Error inicializando tentáculos: {e}")

def stop_tentacles():
    """✅ Detener tentáculos y centrarlos"""
    if not kit:
        return
    
    try:
        print("🐙 [TENTACLES] Centrando tentáculos...")
        kit.servo[TENTACLE_LEFT_CHANNEL].angle = 90
        kit.servo[TENTACLE_RIGHT_CHANNEL].angle = 90
        time.sleep(0.5)
        
        # Opcional: liberar servos para que queden sin tensión
        kit.servo[TENTACLE_LEFT_CHANNEL].angle = None
        kit.servo[TENTACLE_RIGHT_CHANNEL].angle = None
        
        print("🐙 [TENTACLES] Tentáculos detenidos")
        
    except Exception as e:
        print(f"[TENTACLES ERROR] Error deteniendo tentáculos: {e}")

class TentacleThread(threading.Thread):
    """✅ Hilo para movimiento aleatorio de tentáculos en segundo plano"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        print("🐙 [TENTACLES] Sistema de tentáculos activado")
        
        # Estado inicial
        angle_left = 90
        angle_right = 90
        
        while not self.stop_event.is_set() and tentacles_active and not system_shutdown:
            try:
                # Elegir un extremo para la oreja izquierda (0-30 o 150-180)
                target_left = random.choice([random.randint(0, 30), random.randint(150, 180)])
                
                # Para que ambos suban y bajen simétricos, reflejamos la posición:
                # izquierda 0° (arriba) → derecha 180° (arriba)
                # izquierda 180° (abajo) → derecha 0° (abajo)
                target_right = 180 - target_left
                
                step = random.randint(3, 7)  # tamaño del paso
                
                # ---- Movimiento oreja izquierda ----
                if target_left > angle_left:
                    rng_left = range(angle_left, target_left + 1, step)
                else:
                    rng_left = range(angle_left, target_left - 1, -step)
                
                # ---- Movimiento oreja derecha ----
                if target_right > angle_right:
                    rng_right = range(angle_right, target_right + 1, step)
                else:
                    rng_right = range(angle_right, target_right - 1, -step)
                
                # Recorremos ambos rangos en paralelo
                for a_left, a_right in zip(rng_left, rng_right):
                    if not tentacles_active or self.stop_event.is_set():
                        return
                    
                    if kit:
                        try:
                            kit.servo[TENTACLE_LEFT_CHANNEL].angle = a_left
                            kit.servo[TENTACLE_RIGHT_CHANNEL].angle = a_right
                        except Exception as e:
                            print(f"[TENTACLES ERROR] Error moviendo tentáculos: {e}")
                    
                    time.sleep(random.uniform(0.005, 0.015))
                
                # Actualizamos ángulos actuales
                angle_left = target_left
                angle_right = target_right
                
                # Pausa imprevisible antes del siguiente golpe de tentáculos
                pause_time = random.uniform(3, 6)
                for _ in range(int(pause_time * 10)):  # Dividir la pausa para poder salir rápido
                    if self.stop_event.is_set() or not tentacles_active:
                        return
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[TENTACLES ERROR] Error en hilo de tentáculos: {e}")
                time.sleep(1)
        
        print("🛑 [TENTACLES] Hilo de tentáculos detenido")

def activate_tentacles():
    """✅ Activar sistema de tentáculos"""
    global tentacles_active, tentacles_thread
    
    if tentacles_active:
        print("[TENTACLES] Sistema de tentáculos ya está activo")
        return
    
    print("🐙 [TENTACLES] Activando sistema de tentáculos...")
    
    # Inicializar servos
    initialize_tentacles()
    
    # Activar sistema
    tentacles_active = True
    
    # Iniciar hilo de movimiento
    tentacles_thread = TentacleThread()
    tentacles_thread.start()
    
    print("✅ [TENTACLES] Sistema de tentáculos completamente activado")

def deactivate_tentacles():
    """✅ Desactivar sistema de tentáculos"""
    global tentacles_active, tentacles_thread
    
    if not tentacles_active:
        print("[TENTACLES] Sistema de tentáculos ya está desactivado")
        return
    
    print("🐙 [TENTACLES] Desactivando sistema de tentáculos...")
    
    tentacles_active = False
    
    # Detener hilo de movimiento
    if tentacles_thread:
        tentacles_thread.stop()
        tentacles_thread.join(timeout=2)
        tentacles_thread = None
    
    # Centrar y detener tentáculos
    stop_tentacles()
    
    print("✅ [TENTACLES] Sistema de tentáculos desactivado")

# ------------------------------------------------------------------
#    UTILIDADES PARA ENTRAR / SALIR DEL MODO "SILENCIO"
# ------------------------------------------------------------------# ------------------------------------------------------------------
def enter_quiet_mode():
    """Pausar movimientos mecánicos y encender resplandor púrpura."""
    global quiet_mode, led_thread_running
    if quiet_mode:
        return
    quiet_mode = True
    deactivate_tentacles()
    # Parar animación normal de LEDs
    led_thread_running = False
    # Iniciar efecto púrpura
    start_quiet_glow()

def exit_quiet_mode():
    """Reanudar gestos y LEDs estándar."""
    global quiet_mode
    if not quiet_mode:
        return
    quiet_mode = False
    activate_tentacles()
    # Detener efecto púrpura
    stop_quiet_glow()
    # Reactivar animación normal
    iniciar_luces()
# ------------------------------------------------------------------
# ------------------------------------------------------------------



# =============================================
# Pantalla y visualización
# =============================================

def init_display():
    disp = LCD_1inch9.LCD_1inch9(rst=RST_PIN, dc=DC_PIN, bl=BL_PIN)
    disp.Init()
    try:
        disp.set_rotation(ROTATION)
    except AttributeError:
        try:
            disp.rotate(ROTATION)
        except Exception:
            pass
    # ─── NUEVO: pantalla negra y retroiluminación 0 % ───
    disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
    disp.bl_DutyCycle(100)          # apaga el back-light
    return disp

try:
    display = init_display()
except Exception as e:
    print(f"[DISPLAY] No disponible → {e}")
    display = None

# ---------- Hilo de visualización avanzado ----------
class BrutusVisualizer(threading.Thread):
    """
    Brutus ⇒ onda azul que se desplaza continuamente + distorsión naranja según volumen.
    Ajusta PHASE_SPEED para velocidad y GAIN_NOISE para ferocidad.
    """

    FPS            = 20
    BASE_AMPLITUDE = 0.30      # altura mínima del seno
    WAVELENGTH     = 40        # píxeles por ciclo
    PHASE_SPEED    = 10      # velocidad de desplazamiento (rad·px⁻¹ por frame)
    GAIN_NOISE     = 1      # fuerza máx. del ruido (volumen = 1)
    SMOOTH         = 0.85      # filtro exponencial del RMS
    BASE_COLOR     = (255, 255, 255)  # azul base
    NOISE_COLOR    = (255, 0, 255)    
    def __init__(self, display):
        super().__init__(daemon=True)
        self.display     = display
        self.stop_event  = threading.Event()
        self.level_raw   = 0.0
        self.level       = 0.0
        self.phase       = 0.0
        self.x_vals      = np.arange(WIDTH, dtype=np.float32)

    # ── API pública ──────────────────────────────
    def push_level(self, lvl: float):
        self.level_raw = max(0.0, min(lvl, 1.0))

    def stop(self):
        self.stop_event.set()

    # ── hilo principal ───────────────────────────
    def run(self):
        global active_visualizer_threads, system_shutdown
        if not self.display:
            return
            
        # Registrar este hilo
        with shutdown_lock:
            active_visualizer_threads.append(self)
            
        frame_t = 1.0 / self.FPS
        last    = 0.0

        try:
            while not self.stop_event.is_set() and not system_shutdown:
                now = time.time()
                if now - last < frame_t:
                    time.sleep(0.001)
                    continue

                # 1) Desplazamiento continuo
                self.phase += self.PHASE_SPEED
                base_wave = np.sin((self.x_vals / self.WAVELENGTH) + self.phase) * self.BASE_AMPLITUDE

                # 2) RMS suavizado
                self.level = self.SMOOTH * self.level + (1 - self.SMOOTH) * self.level_raw

                # 3) Añadir ruido proporcional al volumen
                if self.level > 0.02:
                    noise_strength = self.GAIN_NOISE * self.level**1.5
                    noise = np.random.normal(0.0, noise_strength, size=WIDTH)
                    wave  = np.clip(base_wave + noise, -1.0, 1.0)
                else:
                    wave = base_wave

                # 4) Dibujar - con manejo de errores
                try:
                    if system_shutdown or self.stop_event.is_set():
                        break
                        
                    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
                    draw = ImageDraw.Draw(img)
                    cy   = HEIGHT // 2

                    # Línea base
                    for x in range(WIDTH - 1):
                        y1 = int(cy - base_wave[x] * cy)
                        y2 = int(cy - base_wave[x + 1] * cy)
                        draw.line((x, y1, x + 1, y2), fill=self.BASE_COLOR)

                    # Ruido (solo si habla)
                    if self.level > 0.02:
                        for x in range(WIDTH - 1):
                            y1 = int(cy - wave[x] * cy)
                            y2 = int(cy - wave[x + 1] * cy)
                            draw.line((x, y1, x + 1, y2), fill=self.NOISE_COLOR)

                    # Verificar una vez más antes de escribir
                    if not system_shutdown and not self.stop_event.is_set() and self.display:
                        self.display.ShowImage(img)
                        
                except (OSError, AttributeError) as e:
                    # La pantalla ya está cerrada, salir silenciosamente
                    break
                except Exception as e:
                    print(f"[VISUALIZER ERROR] {e}")
                    break
                    
                last = now

        except Exception as e:
            if not system_shutdown:
                print(f"[VISUALIZER ERROR] Error en hilo de visualización: {e}")
        finally:
            # Desregistrar este hilo
            with shutdown_lock:
                if self in active_visualizer_threads:
                    active_visualizer_threads.remove(self)
            
            # Limpiar pantalla solo si no estamos en shutdown
            try:
                if not system_shutdown and self.display:
                    self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            except:
                pass  # Ignorar errores durante la limpieza

# =============================================
# --- NUEVO: CLASE PARA STREAMING DE MICRÓFONO A GOOGLE ---
# =============================================
class MicrophoneStream:
    """Clase que abre un stream de micrófono con PyAudio y lo ofrece como un generador."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

# =============================================
# --- FUNCIÓN DE ESCUCHA CON GOOGLE STT MEJORADA ---
# =============================================
def listen_for_command_google() -> str | None:
    """
    Escucha continuamente con Google STT hasta detectar una frase completa.
    """
    if not speech_client:
        print("[ERROR] El cliente de Google STT no está disponible.")
        time.sleep(2)
        return None

    enter_quiet_mode()
    try:
        if is_speaking:
            print("[INFO] Esperando a que termine de hablar...")
            while is_speaking:
                time.sleep(0.1)
            time.sleep(0.5)

        # Configuración base para la API
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_RATE,
            language_code="es-ES",
            enable_automatic_punctuation=True,
            use_enhanced=True
        )
        # ✅ Objeto de configuración que se pasará a la función
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False,
            single_utterance=True
        )

        print("[INFO] 🎤 Escuchando... (habla ahora)")
        
        with MicrophoneStream(STT_RATE, STT_CHUNK) as stream:
            audio_generator = stream.generator()
            
            # ✅ El generador ahora SOLO envía audio, como debe ser
            def request_generator():
                for content in audio_generator:
                    if is_speaking:
                        print("[INFO] Interrumpiendo STT porque el androide está hablando")
                        break
                    yield speech.StreamingRecognizeRequest(audio_content=content)

            try:
                # ✅ CORRECCIÓN: Pasamos 'config' y 'requests' como argumentos separados
                responses = speech_client.streaming_recognize(
                    config=streaming_config,
                    requests=request_generator()
                )
                
                for response in responses:
                    if is_speaking:
                        print("[INFO] Descartando transcripción porque el androide está hablando")
                        return None
                        
                    if response.results and response.results[0].alternatives:
                        transcript = response.results[0].alternatives[0].transcript.strip()
                        if transcript:
                            return transcript

            except Exception as e:
                if "Deadline" in str(e) or "DEADLINE_EXCEEDED" in str(e):
                    print("[INFO] Tiempo agotado - intenta hablar de nuevo")
                elif "inactive" in str(e).lower():
                    print("[INFO] Stream inactivo - reintentando...")
                else:
                    print(f"[ERROR] Excepción en Google STT: {e}")
    finally:
        exit_quiet_mode()
    return None

# =============================================
# --- CALLBACK DE AUDIO SIMPLIFICADO ---
# =============================================
# Ya no necesitamos audio_callback ni audio_queue para Vosk
# audio_queue = []  # ← ELIMINADO
is_speaking = False

# ---------- Hablar ----------

def hablar(texto: str):
    """Sintetiza con ElevenLabs, reproduce a 24 000 Hz y envía niveles RMS al visualizador."""
    global is_speaking
    is_speaking = True

    vis_thread = BrutusVisualizer(display=display) if display else None
    if vis_thread:
        vis_thread.start()

    pa = None
    stream = None
    try:
        # --- Generar audio con ElevenLabs ---
        audio_gen = eleven.text_to_speech.convert(
            text=texto,
            voice_id=VOICE_ID,
            model_id='eleven_flash_v2_5',
            output_format='mp3_22050_32'  # formato válido según API
        )
        mp3_bytes = b''.join(chunk for chunk in audio_gen if isinstance(chunk, bytes))

        # --- Convertir MP3 a PCM 24000 Hz mono 16‑bit ---
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data

        # --- Configurar salida de audio ---
        import pyaudio
        CHUNK = 2048
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=CHUNK,
        )

        MAX_AMP = 32768.0
        offset = 0
        total_len = len(raw_data)
        while offset < total_len:
            chunk_data = raw_data[offset:offset + CHUNK*2]
            offset += CHUNK*2
            stream.write(chunk_data, exception_on_underflow=False)
            if vis_thread and chunk_data:
                samples = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32)
                if samples.size:
                    rms = np.sqrt(np.mean(samples ** 2)) / MAX_AMP
                    vis_thread.push_level(min(rms * 4.0, 1.0))

    except Exception as e:
        print(f"[HABLAR‑ELEVEN] {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if pa:
            pa.terminate()
        if vis_thread:
            vis_thread.stop()
            vis_thread.join()
        is_speaking = False

# hablar en stream

import re
LOG_PATTERNS = (
    r"^🔧 \[MEMORY\]",
    r"^🤖 \[PERSONALITY\]",
    r"^\[EYES",
    r"^\[TENTACLES",
    r"^\[INFO\]",
)
def _sanitize_text(t: str) -> str:
    lines = t.splitlines()
    clean_lines = []
    for line in lines:
        if any(re.match(p, line) for p in LOG_PATTERNS):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

def hablar_en_stream(source):
    """
    Recibe un stream de OpenAI o un generador de texto y lo envía a ElevenLabs 
    para sintetizar y reproducir en tiempo real.
    
    Args:
        source: Puede ser un stream de OpenAI o un generador que yield strings
    """
    global is_speaking, conversation_history
    is_speaking = True

    # Inicia el visualizador si está disponible
    vis_thread = BrutusVisualizer(display=display) if display else None
    if vis_thread:
        vis_thread.start()

    pa = None
    stream = None
    try:
        # Acumular texto por frases completas y sintetizar
        full_response = ""
        sentence_buffer = ""
        print("🤖 Androide: ", end="", flush=True)
        
        # Inicializar PyAudio antes del bucle
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000,
                         output=True, frames_per_buffer=1024)
        MAX_AMP = 32768.0
        
        def synthesize_and_play(text):
            """Sintetizar y reproducir una frase completa"""
            if not text.strip():
                return
            audio_iter = eleven.text_to_speech.stream(
                text=text,
                voice_id=VOICE_ID,
                model_id='eleven_flash_v2_5',
                output_format='pcm_24000'
            )
            for audio_chunk in audio_iter:
                stream.write(audio_chunk)
                if vis_thread and audio_chunk:
                    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    if samples.size:
                        rms = np.sqrt(np.mean(samples**2)) / MAX_AMP
                        vis_thread.push_level(min(rms * 4.0, 1.0))
        
        # ✅ DETECTAR TIPO DE SOURCE: Stream de OpenAI o generador simple
        is_openai_stream = hasattr(source, 'choices') if hasattr(source, '__iter__') else False
        
        for item in source:
            # Extraer texto según el tipo de source
            if is_openai_stream:
                # Stream de OpenAI tradicional
                if hasattr(item, 'choices') and item.choices:
                    delta = item.choices[0].delta.content
                    if not delta:
                        continue
                    text_chunk = delta
                else:
                    continue
            else:
                # Generador simple que yield strings directamente
                text_chunk = item
                if not text_chunk:
                    continue
            # --- SANITIZER ---
            clean_chunk = _sanitize_text(text_chunk)
            if not clean_chunk.strip():
                continue
            full_response += clean_chunk
            sentence_buffer += clean_chunk
            print(clean_chunk, end="", flush=True)
            
            # Detectar fin de frase (punto, exclamación, interrogación)
            if any(punct in clean_chunk for punct in ['.', '!', '?', '\n']):
                # Sintetizar la frase completa acumulada
                synthesize_and_play(sentence_buffer.strip())
                sentence_buffer = ""
        
        # Sintetizar cualquier texto restante
        if sentence_buffer.strip():
            synthesize_and_play(sentence_buffer.strip())
            
        print()  # Nueva línea tras finalizar todo el streaming de texto
        
        # Añadir respuesta al historial solo si tenemos contenido
    # Historial ya gestionado en capa de chat; evitamos duplicados / usuario vacío.

    except Exception as e:
        print(f"[HABLAR-STREAM] {e}")
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if pa:
            pa.terminate()
        if vis_thread:
            vis_thread.stop()
            vis_thread.join()
        is_speaking = False

def hablar_generador(text_generator):
    """
    ✅ NUEVA FUNCIÓN OPTIMIZADA para usar con el generador de chat_with_tools_generator.
    Procesa el texto en tiempo real sin necesidad de streams de OpenAI.
    """
    global is_speaking
    is_speaking = True

    # Inicia el visualizador si está disponible
    vis_thread = BrutusVisualizer(display=display) if display else None
    if vis_thread:
        vis_thread.start()

    pa = None
    stream = None
    try:
        print("🤖 Androide: ", end="", flush=True)
        
        # Inicializar PyAudio
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000,
                         output=True, frames_per_buffer=1024)
        MAX_AMP = 32768.0
        
        sentence_buffer = ""
        
        def synthesize_and_play(text):
            if not text.strip():
                return
            audio_iter = eleven.text_to_speech.stream(
                text=text,
                voice_id=VOICE_ID,
                model_id='eleven_flash_v2_5',
                output_format='pcm_24000'
            )
            for audio_chunk in audio_iter:
                stream.write(audio_chunk)
                if vis_thread and audio_chunk:
                    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    if samples.size:
                        rms = np.sqrt(np.mean(samples**2)) / MAX_AMP
                        vis_thread.push_level(min(rms * 4.0, 1.0))
        
        # Procesar cada chunk de texto del generador
        got_any = False
        for text_chunk in text_generator:
            if not text_chunk:
                continue
            got_any = True
            sentence_buffer += text_chunk
            print(text_chunk, end="", flush=True)
            
            # Sintetizar cuando detectamos fin de frase
            if any(punct in text_chunk for punct in ['.', '!', '?', '\n']):
                synthesize_and_play(sentence_buffer.strip())
                sentence_buffer = ""
        
        # Sintetizar cualquier texto restante
        if sentence_buffer.strip():
            synthesize_and_play(sentence_buffer.strip())
            
        if not got_any:
            # Fallback breve si no llegó nada del modelo
            fallback_text = "Repite la frase."
            print(fallback_text)
            synthesize_and_play(fallback_text)

        print()  # Nueva línea al final

    except Exception as e:
        print(f"[HABLAR-GENERADOR] {e}")
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if pa:
            pa.terminate()
        if vis_thread:
            vis_thread.stop()
            vis_thread.join()
        is_speaking = False

# No olvides reemplazar la llamada a hablar() por hablar_en_stream() en tu main loop

# =============================================
# --- BUCLE PRINCIPAL SIMPLIFICADO ---
# =============================================
# =============================================
# --- BUCLE PRINCIPAL CORREGIDO Y OPTIMIZADO ---
# =============================================
def main():
    global eyes_active, conversation_history

    # ✅ Obtener fecha actual dinámicamente
    now = datetime.now()
    current_date = now.strftime("%d de %B de %Y")
    current_time = now.strftime("%H:%M")
    day_of_week = now.strftime("%A")
    
    # Configurar locale español si está disponible
    try:
        import locale
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
        current_date = now.strftime("%d de %B de %Y")
        day_of_week = now.strftime("%A")
    except:
        # Fallback a formato inglés si no hay locale español
        current_date = now.strftime("%d de %B de %Y")
        day_of_week = now.strftime("%A")

    # ✅ INICIALIZAR HISTORIAL CON PERSONALIDAD
    conversation_history = []
    ensure_system(conversation_history)  # Esto añade automáticamente el sistema
    
    # ✅ VERIFICAR personalidad inicial
    debug_personality(conversation_history)
    
    last_interaction_time = time.time()
    INACTIVITY_TIMEOUT = 300
    WARNING_TIME = 240
    has_warned = False
    
    print("🤖 [STARTUP] Inicializando sistema completo...")
    if kit:
        initialize_eye_servos()
        time.sleep(0.5)
    
    print("🚀 [STARTUP] Activando sistema completo...")
    activate_eyes()
    activate_tentacles()
    iniciar_luces()
    
    # Usa la función de hablar optimizada para el saludo
    hablar("Soy Botijo. ¿Qué quieres ahora, ser inferior?")
    
    print("🎤 [READY] Sistema listo - puedes hablar directamente")

    try:
        while True:
            inactive_time = time.time() - last_interaction_time

            # --- GESTIÓN DE TIMEOUT DE INACTIVIDAD ---
            if inactive_time > INACTIVITY_TIMEOUT:
                print("\n[INFO] Desactivado por inactividad.")
                hablar("Me aburres, humano. Voy a descansar un poco. Habla para reactivarme.")
                
                deactivate_eyes()
                deactivate_tentacles()
                
                print("💤 [SLEEP] Sistema en reposo - habla para reactivar")
                
                # Bucle de reposo para reactivación
                while True:
                    command_text = listen_for_command_google()
                    if command_text:
                        print(f"\n👂 Reactivando con: {command_text}")
                        activate_eyes()
                        activate_tentacles()
                        iniciar_luces()
                        
                        hablar("¡Ah, has vuelto! Procesando tu petición...")
                        last_interaction_time = time.time()
                        has_warned = False
                        
                        
                        # --- BLOQUE DE STREAMING OPTIMIZADO CON BÚSQUEDA WEB PARA REACTIVACIÓN ---
                        try:
                            # ✅ VERIFICAR personalidad antes de responder
                            debug_personality(conversation_history)
                            
                            # ✅ USAR NUEVA FUNCIÓN OPTIMIZADA: una sola llamada GPT cuando es posible
                            # ✅ USAR chat_with_tools_generator con modelo gpt-5
                            text_generator = chat_with_tools_generator(
                                history=conversation_history,
                                user_msg=command_text
                            )
                            # Usar función especializada para generadores
                            hablar_generador(text_generator)
                            
                        except Exception as e:
                            print(f"[CHATGPT-TOOLS-ERROR] {e}")
                            hablar("Mis circuitos están sobrecargados. Habla más tarde.")

                        break # Salir del bucle de reposo

            elif inactive_time > WARNING_TIME and not has_warned:
                hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                has_warned = True

            # --- LÓGICA DE ESCUCHA PRINCIPAL ---
            if not is_speaking:
                enter_quiet_mode()
                command_text = listen_for_command_google()
                exit_quiet_mode()

                if command_text:
                    last_interaction_time = time.time()
                    has_warned = False
                    
                    print(f"\n👂 Humano: {command_text}")
                    
                    # --- BLOQUE DE STREAMING OPTIMIZADO CON BÚSQUEDA WEB PARA CONVERSACIÓN ---
                    try:
                        # ✅ VERIFICAR personalidad antes de cada respuesta
                        debug_personality(conversation_history)
                        
                        # ✅ USAR NUEVA FUNCIÓN OPTIMIZADA: ahorra llamadas cuando no hay búsqueda
                        # ✅ USAR chat_with_tools_generator con modelo gpt-5
                        text_generator = chat_with_tools_generator(
                            history=conversation_history,
                            user_msg=command_text
                        )
                        # Usar función especializada para generadores
                        hablar_generador(text_generator)
                        
                    except Exception as e:
                        print(f"[CHATGPT-TOOLS-ERROR] {e}")
                        hablar("Mis circuitos están sobrecargados. Habla más tarde.")
            else:
                time.sleep(0.1)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detectado...")
        emergency_shutdown()
    finally:
        # ✅ LIMPIEZA DEL SISTEMA - Más robusta
        if not system_shutdown:
            print("🛑 [SHUTDOWN] Apagando sistema completo...")
            emergency_shutdown()
        
        # Limpieza adicional de hilos específicos
        try:
            deactivate_eyes()
            deactivate_tentacles()
            apagar_luces()
        except:
            pass
            
        print("✅ Sistema detenido.")

if __name__ == "__main__":
    if speech_client:
        try:
            main()
        except KeyboardInterrupt:
            print("\n🛑 Interrupción por teclado...")
            emergency_shutdown()
        except Exception as e:
            print(f"\n💥 [FATAL ERROR] Error no controlado: {e}")
            import traceback
            traceback.print_exc()
            emergency_shutdown()
        finally:
            print("👋 Botijo desconectado.")
    else:
        print("❌ [ERROR] No se pudo inicializar Google STT. Verifica tus credenciales.")