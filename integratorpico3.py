# Script del androide discutidor con wake word 'amanece', Picovoice, ChatGPT, Picovoice y SEGUIMIENTO OCULAR
# Visualización avanzada: multi‑onda estilo "Proyecto‑M retro" (ecos, color dinámico)
# Pantalla Waveshare 1.9" en landscape (320×170)
# Versión 2025‑04‑26‑d + EYES INTEGRATION
# ‑ Corrección de muestreo (22 050 Hz)                ✅
# ‑ Hilo de visualización separado                   ✅
# ‑ Nueva visualización multi‑onda con color volumen ✅
# ‑ SISTEMA DE OJOS CON SEGUIMIENTO FACIAL           ✅

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
import threading

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
VOICE_ID = 'RnKqZYEeVQciORlpiCz0'

# --- NUEVO: Configuración Google STT ---
try:
    speech_client = speech.SpeechClient()
    STT_RATE = 16000
    STT_CHUNK = int(STT_RATE / 10)  # 100ms de audio por paquete
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
    "BL": 135,  # Párpado Inferior Izq  
    "TR": 135,  # Párpado Superior Der
    "BR": 45    # Párpado Inferior Der
}

LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 30, 150

# Configuración de parpadeo realista
BLINK_PROBABILITY = 0.02  # 3% - más natural
BLINK_DURATION = 0.12     # Parpadeo más rápido
DOUBLE_BLINK_PROBABILITY = 0.3  # 30% de hacer parpadeo doble

# Configuración de mirada aleatoria
RANDOM_LOOK_PROBABILITY = 0.01
RANDOM_LOOK_DURATION = 0.4

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

# Inicializar ServoKit
try:
    kit = ServoKit(channels=16)
    print("[INFO] ServoKit inicializado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar ServoKit: {e}")
    kit = None

# Rutas y configuración Picovoice ‑ Piper
# Configuración Picovoice
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# Tasas de muestreo separadas
MIC_RATE   = 16000   # micro + Picovoice
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
        
        while not self.stop_event.is_set() and eyes_active:
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

    FPS            = 30
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
        if not self.display:
            return
        frame_t = 1.0 / self.FPS
        last    = 0.0

        while not self.stop_event.is_set():
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

            # 4) Dibujar
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

            self.display.ShowImage(img)
            last = now

        # limpiar al salir
        self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))

# =============================================
# --- FUNCIÓN DE ESCUCHA CON GOOGLE STT (MODIFICADA) ---
# =============================================
def listen_for_command_google(audio_queue: queue.Queue) -> str | None:
    """
    Captura audio de la cola hasta detectar silencio o un máximo de segundos,
    y lo envía como una sola petición sin streaming a Google STT.
    """
    if not speech_client:
        print("[ERROR] El cliente de Google STT no está disponible.")
        return None

    print("🎤 [STT] Capturando audio para reconocimiento...")
    start_time = time.time()
    max_duration = 8.0        # segundos máximos de grabación
    silence_limit = 1.0       # segundos de silencio para terminar antes
    silence_acc = 0.0
    chunks = []

    # Recolectar audio hasta silencio prolongado o tiempo límite
    while (time.time() - start_time) < max_duration:
        try:
            data = audio_queue.get(timeout=0.3)
            chunks.append(data)
            silence_acc = 0.0
        except queue.Empty:
            silence_acc += 0.3
            if silence_acc >= silence_limit:
                break
            continue

    if not chunks:
        print("[STT] No se capturó ningún audio.")
        return None

    audio_content = b"".join(chunks)

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
                return transcript

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
        # Usar modelos personalizados para "amanece" y "botijo" con modelo en español
        keyword_paths = [
            "/home/jack/botijo/amanece.ppn",  # Índice 0 - Para despertar
            "/home/jack/botijo/botijo.ppn"    # Índice 1 - Para comandos
        ]
        model_path = "/home/jack/botijo/porcupine_params_es.pv"
        
        # ✅ NUEVO: Ajustar la sensibilidad para cada palabra clave
        # El orden debe coincidir con 'keyword_paths'.
        # [0] = "amanece", [1] = "botijo"
        sensitivities = [0.7, 0.9] # Aumentamos la sensibilidad de "Botijo" a 0.7

        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=keyword_paths,
            model_path=model_path,  # Usar modelo en español
            sensitivities=sensitivities # ✅ Aplicar las sensibilidades personalizadas
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

# ---------- Variables globales para Picovoice ----------
audio_buffer = []
porcupine = None
is_system_awake = False  # Estado global del sistema
stt_audio_queue = queue.Queue() # Cola para comunicación entre callback y STT
is_listening_for_command = False # Flag para controlar cuándo enviar audio a STT

# ---------- Callback de audio Unificado ----------
def audio_callback(indata, frames, time_info, status):
    """Callback único que alimenta a Picovoice y a la cola de Google STT."""
    global is_listening_for_command
    if status:
        print(f"[AUDIO] {status}", flush=True)
    
    # El audio no se procesa si el sistema está hablando para evitar acoples.
    if is_speaking:
        return

    # 1. Siempre enviar a Picovoice para la detección de wakeword
    audio_data = np.frombuffer(indata, dtype=np.int16)
    audio_buffer.extend(audio_data)

    # 2. Si estamos en modo comando, enviar también a la cola de Google STT
    if is_listening_for_command:
        # Enviar bytes brutos para STT
        try:
            chunk = indata if isinstance(indata, (bytes, bytearray)) else indata.tobytes()
        except Exception:
            chunk = bytes(indata)
        stt_audio_queue.put(chunk)
        # Depuración: mostrar tamaño de fragmento y tamaño de cola
        print(f"[DEBUG STT] Chunk recibido: {len(chunk)} bytes, cola={stt_audio_queue.qsize()}")