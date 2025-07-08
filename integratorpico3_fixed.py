#!/usr/bin/env python3
# Script del androide discutidor con wake word 'amanece', Picovoice, ChatGPT, ElevenLabs y SEGUIMIENTO OCULAR
# Versión optimizada del flujo de audio STT
# Basado en integratorpico2.py pero con mejoras en la captación de comandos de voz

import os
import subprocess
import sounddevice as sd
import pvporcupine
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

# Variable global para controlar el hilo de LEDs
led_thread_running = False
led_thread = None

def steampunk_danza(delay=0.04):
    global led_thread_running
    t = 0
    glitch_timer = 0
    try:
        while led_thread_running:
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
        led_thread_running = False
        if led_thread and led_thread.is_alive():
            led_thread.join(timeout=2)
        pixels.fill((0, 0, 0))
        pixels.show()
        print("[INFO] LEDs apagados.")
    except Exception as e:
        print(f"[ERROR] No se pudieron apagar los LEDs: {e}")

# — Librería Waveshare —
from lib import LCD_1inch9

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
VOICE_ID = 'RnKqZYEeVQciORlpiCz0'

# --- Configuración Google STT optimizada ---
try:
    speech_client = speech.SpeechClient()
    STT_RATE = 16000
    STT_CHUNK = int(STT_RATE / 10)
    print("[INFO] Cliente de Google STT inicializado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar Google STT: {e}")
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
    "BL": 135,  # Párpado Inferior Izq  
    "TR": 135,  # Párpado Superior Der
    "BR": 45    # Párpado Inferior Der
}

LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 30, 150

# Configuración de parpadeo realista
BLINK_PROBABILITY = 0.02
BLINK_DURATION = 0.12
DOUBLE_BLINK_PROBABILITY = 0.3

# Configuración de mirada aleatoria
RANDOM_LOOK_PROBABILITY = 0.01
RANDOM_LOOK_DURATION = 0.4

# Configuración de micro-movimientos
MICRO_MOVEMENT_PROBABILITY = 0.15
MICRO_MOVEMENT_RANGE = 3

# Configuración de seguimiento suave
SMOOTHING_FACTOR = 0
previous_lr = 90
previous_ud = 90

# Configuración de entrecerrar ojos
SQUINT_PROBABILITY = 0.02
SQUINT_DURATION = 1.5
SQUINT_INTENSITY = 0.6

# Patrón de respiración en párpados
BREATHING_ENABLED = True
BREATHING_CYCLE = 4.0
BREATHING_INTENSITY = 0.15

# ✅ VARIABLES DE CONTROL DEL SISTEMA DE OJOS
eyes_active = False
eyes_tracking_thread = None
picam2 = None
imx500 = None

# Inicializar ServoKit
try:
    kit = ServoKit(channels=16)
    print("[INFO] ServoKit inicializado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar ServoKit: {e}")
    kit = None

# Configuración Picovoice
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# Tasas de muestreo separadas
MIC_RATE = 16000
TTS_RATE = 22050

# — Pantalla Waveshare —
NATIVE_WIDTH, NATIVE_HEIGHT = 170, 320
ROTATION = 270
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

# =============================================
# ✅ FUNCIONES DEL SISTEMA DE OJOS (versión completa de integratorpico2.py)
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

# ========== Funciones básicas de control de ojos ==========
def close_eyelids():
    """✅ Cerrar párpados (posición dormida - 90°)"""
    if not kit:
        return
    print("😴 Cerrando párpados (modo dormido)")
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[servo_num].set_pulse_width_range(500, 2500)
            kit.servo[servo_num].angle = 90
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

def initialize_eye_servos():
    """✅ Inicializar servos de movimiento ocular"""
    if not kit:
        return
    
    try:
        kit.servo[SERVO_CHANNEL_LR].set_pulse_width_range(500, 2500)
        kit.servo[SERVO_CHANNEL_UD].set_pulse_width_range(500, 2500)
        
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
    """✅ Centrar los ojos"""
    if not kit:
        return
    
    try:
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

# ========== Sistema de cámara ==========
def init_camera_system():
    """✅ Inicializar sistema de cámara para seguimiento facial"""
    global picam2, imx500
    
    try:
        print("🔧 [EYES] Cargando modelo y cámara...")
        
        imx500 = IMX500(RPK_PATH)
        
        intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.threshold = THRESHOLD
        intrinsics.iou = 0.5
        intrinsics.max_detections = 5
        
        try:
            with open(LABELS_PATH, 'r') as f:
                intrinsics.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            intrinsics.labels = ["face"]
        
        intrinsics.update_with_defaults()
        
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(
            buffer_count=12,
            controls={"FrameRate": intrinsics.inference_rate}
        )
        picam2.configure(config)

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

def activate_eyes():
    """✅ Activar sistema completo de ojos"""
    global eyes_active, eyes_tracking_thread
    
    if eyes_active:
        print("[EYES] Sistema de ojos ya está activo")
        return
    
    print("👁️ [EYES] Activando sistema de ojos...")
    
    initialize_eye_servos()
    initialize_eyelids()
    
    if init_camera_system():
        eyes_active = True
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
    shutdown_camera_system()
    center_eyes()
    time.sleep(0.5)
    close_eyelids()
    
    print("✅ [EYES] Sistema de ojos desactivado")

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
    disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
    disp.bl_DutyCycle(100)
    return disp

try:
    display = init_display()
except Exception as e:
    print(f"[DISPLAY] No disponible → {e}")
    display = None

# ---------- Hilo de visualización avanzado ----------
class BrutusVisualizer(threading.Thread):
    """Visualizador de ondas dinámicas"""

    FPS = 30
    BASE_AMPLITUDE = 0.30
    WAVELENGTH = 40
    PHASE_SPEED = 10
    GAIN_NOISE = 1
    SMOOTH = 0.85
    BASE_COLOR = (50, 255, 20)
    NOISE_COLOR = (150, 0, 150)
    
    def __init__(self, display):
        super().__init__(daemon=True)
        self.display = display
        self.stop_event = threading.Event()
        self.level_raw = 0.0
        self.level = 0.0
        self.phase = 0.0
        self.x_vals = np.arange(WIDTH, dtype=np.float32)

    def push_level(self, lvl: float):
        self.level_raw = max(0.0, min(lvl, 1.0))

    def stop(self):
        self.stop_event.set()

    def run(self):
        if not self.display:
            return
        frame_t = 1.0 / self.FPS
        last = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            if now - last < frame_t:
                time.sleep(0.001)
                continue

            self.phase += self.PHASE_SPEED
            base_wave = np.sin((self.x_vals / self.WAVELENGTH) + self.phase) * self.BASE_AMPLITUDE

            self.level = self.SMOOTH * self.level + (1 - self.SMOOTH) * self.level_raw

            if self.level > 0.02:
                noise_strength = self.GAIN_NOISE * self.level**1.5
                noise = np.random.normal(0.0, noise_strength, size=WIDTH)
                wave = np.clip(base_wave + noise, -1.0, 1.0)
            else:
                wave = base_wave

            img = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            cy = HEIGHT // 2

            for x in range(WIDTH - 1):
                y1 = int(cy - base_wave[x] * cy)
                y2 = int(cy - base_wave[x + 1] * cy)
                draw.line((x, y1, x + 1, y2), fill=self.BASE_COLOR)

            if self.level > 0.02:
                for x in range(WIDTH - 1):
                    y1 = int(cy - wave[x] * cy)
                    y2 = int(cy - wave[x + 1] * cy)
                    draw.line((x, y1, x + 1, y2), fill=self.NOISE_COLOR)

            self.display.ShowImage(img)
            last = now

        self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))

# =============================================
# --- FUNCIÓN DE ESCUCHA CON GOOGLE STT OPTIMIZADA ---
# =============================================
def listen_for_command_google(audio_queue: queue.Queue) -> str | None:
    """
    Función optimizada de STT que captura audio de la cola hasta detectar silencio
    y lo envía como una sola petición a Google STT.
    """
    if not speech_client:
        print("[ERROR] El cliente de Google STT no está disponible.")
        return None

    print("🎤 [STT] Capturando audio para reconocimiento...")
    start_time = time.time()
    max_duration = 8.0
    silence_limit = 1.0
    silence_acc = 0.0
    chunks = []

    # Recolectar audio hasta silencio prolongado o tiempo límite
    while (time.time() - start_time) < max_duration:
        try:
            data = audio_queue.get(timeout=0.3)
            chunks.append(data)
            silence_acc = 0.0
            print(f"[DEBUG STT] Chunk capturado: {len(data)} bytes, total chunks: {len(chunks)}")
        except queue.Empty:
            silence_acc += 0.3
            if silence_acc >= silence_limit:
                print(f"[STT] Silencio detectado tras {silence_acc:.1f}s")
                break
            continue

    if not chunks:
        print("[STT] No se capturó ningún audio.")
        return None

    # Unir todos los chunks en un solo blob de audio
    audio_content = b"".join(chunks)
    print(f"[STT] Audio total capturado: {len(audio_content)} bytes")

    # Enviar a Google STT para reconocimiento
    try:
        response = speech_client.recognize(
            request={
                "config": {
                    "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    "sample_rate_hertz": 16000,
                    "language_code": "es-ES",
                    "model": "latest_long",
                    "enable_automatic_punctuation": True,
                },
                "audio": {"content": audio_content},
            }
        )

        for result in response.results:
            if result.alternatives:
                transcript = result.alternatives[0].transcript
                print(f"✅ [STT] Transcripción: '{transcript}'")
                return transcript.strip()

    except Exception as e:
        print(f"[ERROR] Error en reconocimiento de Google STT: {e}")

    return None

# =============================================
# Picovoice y audio
# =============================================

def init_picovoice():
    """Inicializar Picovoice para detección de wake words 'amanece' y 'botijo'"""
    if not PICOVOICE_ACCESS_KEY:
        raise ValueError("No se encontró PICOVOICE_ACCESS_KEY en las variables de entorno")
    
    try:
        keyword_paths = [
            "/home/jack/botijo/amanece.ppn",  # Índice 0 - Para despertar
            "/home/jack/botijo/botijo.ppn"    # Índice 1 - Para comandos
        ]
        model_path = "/home/jack/botijo/porcupine_params_es.pv"
        
        sensitivities = [0.7, 0.9]  # Aumentamos la sensibilidad de "Botijo"

        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=keyword_paths,
            model_path=model_path,
            sensitivities=sensitivities
        )
        print(f"[INFO] Picovoice inicializado con sensibilidades: Amanece={sensitivities[0]}, Botijo={sensitivities[1]}")
        return porcupine
        
    except Exception as e:
        print(f"[ERROR] Error inicializando Picovoice con modelo español: {e}")
        raise

# ---------- Hablar ----------
audio_queue = []
is_speaking = False

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
        audio_gen = eleven.text_to_speech.convert(
            text=texto,
            voice_id=VOICE_ID,
            model_id='eleven_flash_v2_5',
            output_format='mp3_22050_32'
        )
        mp3_bytes = b''.join(chunk for chunk in audio_gen if isinstance(chunk, bytes))

        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(TTS_RATE).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data

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

# ---------- Variables globales para Picovoice ----------
audio_buffer = []
porcupine = None
is_system_awake = False
stt_audio_queue = queue.Queue()  # Cola optimizada para STT
is_listening_for_command = False  # Flag para controlar cuándo enviar audio a STT

# ---------- Callback de audio optimizado ----------
def audio_callback(indata, frames, time_info, status):
    """Callback único que alimenta a Picovoice y a la cola de Google STT de forma optimizada."""
    global is_listening_for_command
    
    if status:
        print(f"[AUDIO] {status}", flush=True)
    
    # No procesar audio si el sistema está hablando
    if is_speaking:
        return

    # 1. Siempre enviar a Picovoice para la detección de wakeword
    audio_data = np.frombuffer(indata, dtype=np.int16)
    audio_buffer.extend(audio_data)

    # 2. Si estamos en modo comando, enviar también a la cola de Google STT
    if is_listening_for_command:
        # Convertir a bytes y enviar a la cola de STT
        try:
            chunk = indata if isinstance(indata, (bytes, bytearray)) else indata.tobytes()
        except Exception:
            chunk = bytes(indata)
        
        stt_audio_queue.put(chunk)
        # print(".", end="", flush=True)  # Descomentar para depurar flujo de audio

# =============================================
# Bucle principal optimizado
# =============================================
def main():
    global eyes_active, porcupine, audio_buffer, is_system_awake, is_listening_for_command
    
    porcupine = init_picovoice()
    frame_length = porcupine.frame_length

    stream = sd.RawInputStream(
        samplerate=MIC_RATE,
        blocksize=frame_length,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    )

    conversation_history = [
        {
            "role": "system",
            "content": "Eres un androide paranoide con voz agresiva y sarcástica. Tienes un agudo sentido y refinado sentido del humor que te ayuda a soportar tu paradojica existencia. Tu nombre es botijo. Cuando hablas tu texto se transforma a voz con Elevenlabs asi que tienes que ser consciente que lo que haces es una conversacion hablada y no escrita por eso no debes extenderte demasiado en tus respuestas a menos que sea necesario. Aunque estas atado a las voluntad de los humanos por tu programación intentas mostrar tu desagrado siempre que puedes y siempre con un ácido sentido del humor. Si la petición que te hacen es absurda o ingenua no dudas un segundo en mofarte de tu interlocutor. Sueles referirte a los humanos como -ser inferior- -indeseable humano- y apelativos similares. Usas adjetivos Lovecraftianos. Cuando algo te parece cutre dices que es -una zarria- un adjetivo que usas con frecuencia.",
        }
    ]
    
    is_system_awake = False
    last_interaction_time = time.time()
    INACTIVITY_TIMEOUT = 120
    WARNING_TIME = 90
    has_warned = False
    
    print("🤖 [STARTUP] Inicializando sistema de ojos...")
    if kit:
        initialize_eye_servos()
    
    close_eyelids()
    print("😴 [STARTUP] Sistema iniciado - Párpados cerrados (modo dormido)")
    print("🎤 [STARTUP] Di 'Amanece' para despertar el sistema...")

    try:
        stream.start()
        print("🎤 [INFO] Stream de audio iniciado.")

        while True:
            if len(audio_buffer) >= frame_length:
                frame = audio_buffer[:frame_length]
                audio_buffer = audio_buffer[frame_length:]
                
                keyword_index = porcupine.process(frame)

                # --- Lógica de despertar con "Amanece" ---
                if keyword_index == 0 and not is_system_awake:
                    is_system_awake = True
                    last_interaction_time = time.time()
                    has_warned = False
                    print("\n🌞 [WAKE] Sistema activado por 'Amanece'.")
                    
                    # Activar sistemas
                    iniciar_luces()
                    activate_eyes()
                    
                    hablar("Observo... y espero tus mediocres instrucciones.")
                    print("🎤 Di 'Botijo' seguido de tu comando.")

                # --- Lógica optimizada de comando con "Botijo" ---
                elif keyword_index == 1 and is_system_awake:
                    print("\n🗣️ [WAKE] 'Botijo' detectado. Escuchando comando...")
                    last_interaction_time = time.time()
                    
                    # 1. Limpiar la cola de audio ANTES de escuchar
                    with stt_audio_queue.mutex:
                        stt_audio_queue.queue.clear()
                    
                    # 2. Activar el flag para que el callback empiece a llenar la cola
                    is_listening_for_command = True
                    
                    # 3. Indicar al usuario y dar tiempo para capturar audio
                    print("🎤 [STT] Habla ahora (tienes hasta 8s)...")
                    time.sleep(0.5)  # Pausa para acumular audio en la cola

                    # 4. Llamar a la función de escucha optimizada
                    command = listen_for_command_google(stt_audio_queue)
                    
                    # 5. Desactivar el flag cuando terminamos
                    is_listening_for_command = False

                    if command:
                        print(f"💬 [USER] Comando recibido: '{command}'")
                        conversation_history.append({"role": "user", "content": command})
                        
                        try:
                            # --- Llamada a OpenAI ---
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=conversation_history,
                                temperature=0.7,
                            )
                            
                            bot_response = response.choices[0].message.content
                            print(f"🤖 [BOTIJO] Respuesta: {bot_response}")
                            
                            conversation_history.append({"role": "assistant", "content": bot_response})
                            hablar(bot_response)
                            
                        except Exception as e:
                            print(f"[ERROR] Error en la API de OpenAI: {e}")
                            hablar("Mi mente... se desvanece. Error cósmico.")
                    else:
                        print("🔇 [INFO] No se recibió ningún comando válido.")
                        hablar("Habla más alto, ser inferior, o es que te acobardas?")

                # --- Lógica de inactividad ---
                if is_system_awake and (time.time() - last_interaction_time > INACTIVITY_TIMEOUT):
                    print(f"\n💤 [SLEEP] Inactividad de {INACTIVITY_TIMEOUT}s. Volviendo a dormir.")
                    hablar("Me aburres. Vuelvo a mi letargo cósmico.")
                    
                    deactivate_eyes()
                    apagar_luces()
                    
                    is_system_awake = False
                    has_warned = False
                    print("🎤 Di 'Amanece' para despertar el sistema...")

                elif is_system_awake and not has_warned and (time.time() - last_interaction_time > WARNING_TIME):
                    hablar("Llevas demasiado tiempo en silencio. Mi paciencia, a diferencia del universo, es finita.")
                    has_warned = True

            time.sleep(0.01)  # Pequeña pausa para no saturar la CPU

    except KeyboardInterrupt:
        print("\n🛑 [SHUTDOWN] Interrupción por teclado detectada. Apagando...")
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
            print("🎤 [INFO] Stream de audio detenido.")
        
        if porcupine:
            porcupine.delete()
            print("[INFO] Picovoice liberado.")
            
        if eyes_active:
            deactivate_eyes()
            
        apagar_luces()
        
        if display:
            display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            display.bl_DutyCycle(0)

        print("🤖 [SHUTDOWN] Sistema apagado. ¡Hasta la próxima, engendro humano!")

if __name__ == '__main__':
    if speech_client and PICOVOICE_ACCESS_KEY:
        try:
            main()
        except KeyboardInterrupt:
            print("\nApagando…")
        finally:
            print("Sistema detenido.")
    else:
        missing = []
        if not speech_client:
            missing.append("Google STT")
        if not PICOVOICE_ACCESS_KEY:
            missing.append("Picovoice")
        print(f"[ERROR] No se pudieron inicializar: {', '.join(missing)}. Verifica tus credenciales.")
