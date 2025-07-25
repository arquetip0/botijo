#!/usr/bin/env python3
# filepath: /home/jack/botijo/integrapruebas.py
# Script del androide discutidor SIN wake word - Funciona desde el inicio
# Versión sin Vosk - Solo Google STT + ChatGPT + Piper + Eyes + Tentacles

import os
import subprocess
import sounddevice as sd
# import vosk  # ← ELIMINADO
from openai import OpenAI
from dotenv import load_dotenv
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
import board
import neopixel
import threading
import signal
import sys
import atexit
from duckduckgo_search import DDGS  # pip install duckduckgo-search

# ────────────────────────────────────────────────────────────────────────
# SISTEMA DE BÚSQUEDA WEB CON FUNCTION CALLING
# ────────────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    """Devuelve un resumen compacto de los *max_results* primeros hits.

    Cada línea incluye título, extracto (≈180 chars) y URL.  Se recorta para
    no inflar el prompt; si algo falla devuelve "SEARCH_ERROR: …" que el
    modelo puede manejar pidiendo otra búsqueda.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
    except Exception as e:
        return f"SEARCH_ERROR: {e}"

    if not results:
        return "No results found."

    lines = [f"Search results for '{query}':"]
    for idx, res in enumerate(results, 1):
        title = (res.get("title") or "No title").strip()
        body = (res.get("body") or "").strip().replace("\n", " ")
        href = res.get("href") or ""
        lines.append(f"{idx}. {title} — {body[:180]}… (Source: {href})")
    return "\n".join(lines)

# Declaración de herramienta para OpenAI
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet and return a concise summary of the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
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
    """Orquesta el flujo completo *usuario → GPT → (web) → GPT → voz*.

    • *history*        → tu conversation_history global.
    • *user_msg*       → texto recién capturado por STT.
    • *speak*          → función síncrona para frases cortas (tu `hablar`).
    • *speak_stream*   → función que consume el `response_stream` de OpenAI
                          y reproduce la voz (tu `hablar_en_stream`).
    • Devuelve la respuesta textual final (por si quieres loguearla).
    """
    import json
    
    # 1. Purgar historial y preparar mensajes
    trimmed = history[-max_history:]
    messages = trimmed + [{"role": "user", "content": user_msg}]

    while True:
        # 2. Primera petición (o subsiguiente si volvió al loop)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=1,
            max_tokens=300,
        )

        msg = response.choices[0].message

        # 3. ¿GPT solicita herramienta?
        if msg.tool_calls:
            for call in msg.tool_calls:
                if call.function.name == "web_search":
                    args = json.loads(call.function.arguments)
                    query = args["query"]
                    max_r = args.get("max_results", 3)
                    speak(speak_phrase)                       # frase corta
                    result_text = web_search(query, max_r)   # llamar a DDG

                    # 3a. Añadir marca de la *tool call*
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call.model_dump()],
                    })
                    # 3b. Añadir respuesta de la tool
                    messages.append({
                        "role": "tool",
                        "content": result_text,
                        "tool_call_id": call.id,
                    })
            # Vuelve al while: GPT generará respuesta final con la info nueva
            continue

        # 4. No hay tool_call → generó respuesta definitiva
        reply_stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages + [{"role": "assistant", "content": msg.content}],
            temperature=1,
            max_tokens=300,
            stream=True,
        )
        speak_stream(reply_stream)  # voz en tiempo real

        # 5. Actualizar historial global
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": msg.content},
        ])
        return msg.content  # por si lo necesitas en el caller

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
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuración ElevenLabs
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID =   'RnKqZYEeVQciORlpiCz0' # 'ByVRQtaK1WDOvTmP1PKO'  #'RnKqZYEeVQciORlpiCz0' voz buena # voz jacobiana '0IvOsEbrz5BR3wpPyKbU'
TTS_RATE   = 22_050             # Hz
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
    global active_visualizer_threads, display
    
    with shutdown_lock:
        if system_shutdown:
            return  # Ya se está ejecutando
        system_shutdown = True
    
    print("\n🚨 [EMERGENCY] Iniciando limpieza de emergencia...")
    
    # 1. Detener todos los visualizadores activos
    if active_visualizer_threads:
        print("🛑 [EMERGENCY] Deteniendo visualizadores...")
        for vis in active_visualizer_threads[:]:  # Copia para evitar modificación concurrente
            try:
                vis.stop()
            except:
                pass
        
        # Esperar un poco para que se detengan
        time.sleep(0.3)
    
    # 2. Detener sistemas principales
    try:
        eyes_active = False
        tentacles_active = False
        led_thread_running = False
    except:
        pass
    
    # 3. Devolver ojos y párpados a posición de reposo (90°)
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
    
    # 5. Detener LEDs
    try:
        pixels.fill((0, 0, 0))
        pixels.show()
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
    return random.random() < BLINK_PROBABILITY

def should_look_random():
    return random.random() < RANDOM_LOOK_PROBABILITY

def should_squint():
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
    BASE_COLOR     = (50, 255, 20)  # azul base
    NOISE_COLOR    = (150, 0, 150)    
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
        single_utterance=False
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

    return None

# =============================================
# --- CALLBACK DE AUDIO SIMPLIFICADO ---
# =============================================
# Ya no necesitamos audio_callback ni audio_queue para Vosk
# audio_queue = []  # ← ELIMINADO
is_speaking = False

# ---------- Hablar ----------

def hablar(texto: str):
    """Sintetiza con ElevenLabs, reproduce a 22 050 Hz y envía niveles RMS al visualizador."""
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
            output_format='mp3_22050_32'
        )
        mp3_bytes = b''.join(chunk for chunk in audio_gen if isinstance(chunk, bytes))

        # --- Convertir MP3 a PCM 22050 Hz mono 16‑bit ---
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(TTS_RATE).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data

        # --- Configurar salida de audio ---
        import pyaudio
        CHUNK = 2048
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_RATE,
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

def hablar_en_stream(response_stream):
    """
    Recibe un stream de texto de ChatGPT y lo envía directamente a ElevenLabs para 
    sintetizar y reproducir en tiempo real.
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
        
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            full_response += delta
            sentence_buffer += delta
            print(delta, end="", flush=True)
            
            # Detectar fin de frase (punto, exclamación, interrogación)
            if any(punct in delta for punct in ['.', '!', '?', '\n']):
                # Sintetizar la frase completa acumulada
                synthesize_and_play(sentence_buffer.strip())
                sentence_buffer = ""
        
        # Sintetizar cualquier texto restante
        if sentence_buffer.strip():
            synthesize_and_play(sentence_buffer.strip())
            
        print()  # Nueva línea tras finalizar todo el streaming de texto
        # Añadir respuesta al historial
        conversation_history.append({"role": "assistant", "content": full_response})
        conversation_history = [conversation_history[0]] + conversation_history[-9:]

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

# No olvides reemplazar la llamada a hablar() por hablar_en_stream() en tu main loop

# =============================================
# --- BUCLE PRINCIPAL SIMPLIFICADO ---
# =============================================
# =============================================
# --- BUCLE PRINCIPAL CORREGIDO Y OPTIMIZADO ---
# =============================================
def main():
    global eyes_active, conversation_history

    conversation_history = [
        {
            "role": "system",
            "content": "Eres un androide paranoide con voz agresiva y sarcástica. Tienes un agudo sentido y refinado sentido del humor que te ayuda a soportar tu paradojica existencia. Tu nombre es botijo. Cuando hablas tu texto se transforma a voz con Elevenlabs asi que tienes que ser consciente que lo que haces es una conversacion hablada y no escrita por eso no debes extenderte demasiado en tus respuestas a menos que sea necesario. Aunque estas atado a las voluntad de los humanos por tu programación intentas mostrar tu desagrado siempre que puedes y siempre con un ácido sentido del humor. Si la petición que te hacen es absurda o ingenua no dudas un segundo en mofarte de tu interlocutor. Sueles referirte a los humanos como -ser inferior- -indeseable humano- y apelativos similares. Usas adjetivos Lovecraftianos. Cuando algo te parece cutre dices que es -una zarria- un adjetivo que usas con frecuencia. NUEVO: Tienes acceso a búsqueda web en tiempo real. Cuando necesites información actualizada o específica que no tienes, puedes buscar en internet. Eres consciente de esta capacidad y la usas cuando es apropiado, aunque siempre con tu característico sarcasmo.",
        }
    ]
    
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
                
                # Bucle de reposo para reactivar
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
                        
                        
                        # --- BLOQUE DE STREAMING CON BÚSQUEDA WEB PARA REACTIVACIÓN ---
                        try:
                            chat_with_tools(
                                history=conversation_history,
                                user_msg=command_text,
                                speak=hablar,
                                speak_stream=hablar_en_stream
                            )
                        except Exception as e:
                            print(f"[CHATGPT-TOOLS-ERROR] {e}")
                            hablar("Mis circuitos están sobrecargados. Habla más tarde.")

                        break # Salir del bucle de reposo

            elif inactive_time > WARNING_TIME and not has_warned:
                hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                has_warned = True

            # --- LÓGICA DE ESCUCHA PRINCIPAL ---
            if not is_speaking:
                command_text = listen_for_command_google()

                if command_text:
                    last_interaction_time = time.time()
                    has_warned = False
                    
                    print(f"\n👂 Humano: {command_text}")
                    
                    # --- BLOQUE DE STREAMING CON BÚSQUEDA WEB PARA CONVERSACIÓN ---
                    try:
                        chat_with_tools(
                            history=conversation_history,
                            user_msg=command_text,
                            speak=hablar,
                            speak_stream=hablar_en_stream
                        )
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