# ritmo8_elevenlabs.py
# Script del androide discutidor con wake word, Vosk, ChatGPT y ElevenLabs
# Visualización avanzada: multi‑onda estilo "Proyecto‑M retro" (ecos, color dinámico)
# Pantalla Waveshare 1.9" en landscape (320×170)
# Versión 2025‑06‑09‑elevenlabs

import os
import subprocess
import sounddevice as sd
import vosk
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
import numpy as np
from PIL import Image, ImageDraw
import pyaudio
import threading
import queue
import pvporcupine

# ElevenLabs imports (simple API)
from elevenlabs import generate, play, set_api_key, voices

# — Librería Waveshare —
from lib import LCD_1inch9

# ========================
# Configuración general
# ========================

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Configuración OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuración ElevenLabs
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if ELEVENLABS_API_KEY:
    set_api_key(ELEVENLABS_API_KEY)
    try:
        available_voices = voices()
        print(f"[INFO] ElevenLabs inicializado - {len(available_voices)} voces disponibles")
    except Exception as e:
        print(f"[WARN] ElevenLabs no disponible: {e}")
else:
    print("[WARN] ELEVENLABS_API_KEY no encontrada")

# Configuración ElevenLabs
ELEVENLABS_VOICE_ID = "ErXwobaYiN019PkySvjV"  # Rachel
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# Configuración Vosk
VOSK_MODEL_PATH = "model"
MIC_RATE = 16000

# Configuración Picovoice
KEYWORD_PATH = "/home/jack/picovoice/Botijo_es_raspberry-pi_v3_0_0.ppn"

# Configuración pantalla
NATIVE_WIDTH, NATIVE_HEIGHT = 170, 320
ROTATION = 270
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH  # 320×170
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

# Variables globales
audio_queue = []
is_speaking = False

# =============================================
# Pantalla y visualización
# =============================================

def init_display():
    """Inicializa la pantalla Waveshare"""
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

# Inicializar pantalla
try:
    display = init_display()
    print("[INFO] Pantalla inicializada correctamente")
except Exception as e:
    print(f"[DISPLAY] No disponible → {e}")
    display = None

# ---------- Hilo de visualización avanzado ----------
class VisualizerThread(threading.Thread):
    """Multi‑onda con ecos y color según volumen (estilo Proyecto‑M retro)."""

    WAVE_COUNT = 4     # nº de ecos (0 = último nivel)
    DECAY      = 0.75  # cuánto se atenúan los ecos sucesivos
    FPS        = 30

    def __init__(self, display):
        super().__init__(daemon=True)
        self.display = display
        self.level_q = queue.Queue(maxsize=300)
        self.stop_event = threading.Event()
        # buffer para cada onda (320 columnas)
        self.buffers = [np.zeros(WIDTH, dtype=np.float32) for _ in range(self.WAVE_COUNT)]
        self.last_draw = 0.0

    def push_level(self, level):
        try:
            self.level_q.put_nowait(level)
        except queue.Full:
            pass

    def stop(self):
        self.stop_event.set()

    @staticmethod
    def _volume_to_color(level):
        """De verde (bajo) a rojo (alto) pasando por amarillo."""
        r = int(255 * level)
        g = int(255 * (1.0 - 0.5 * level))
        return (r, g, 0)

    def _shift_buffers(self, new_level):
        for buf in self.buffers:
            buf[:-1] = buf[1:]
        self.buffers[0][-1] = new_level
        for i in range(1, self.WAVE_COUNT):
            self.buffers[i][-1] = self.buffers[i-1][-2] * self.DECAY

    def run(self):
        if not self.display:
            return
        frametime = 1 / self.FPS
        while not self.stop_event.is_set():
            now = time.time()
            if now - self.last_draw < frametime:
                time.sleep(0.002)
                continue
            try:
                level = self.level_q.get_nowait()
            except queue.Empty:
                continue
            self._shift_buffers(level)
            img = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            cy = HEIGHT // 2
            for idx, buf in enumerate(self.buffers):
                fade = (self.DECAY ** idx)
                color = tuple(int(c * fade) for c in self._volume_to_color(buf[-1]))
                for x in range(WIDTH - 1):
                    y1 = int(cy - buf[x] * cy)
                    y2 = int(cy - buf[x + 1] * cy)
                    draw.line((x, y1, x + 1, y2), fill=color)
            self.display.ShowImage(img)
            self.last_draw = now
        self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))

class BrutusVisualizer(threading.Thread):
    """Brutus ⇒ onda azul + distorsión naranja según volumen."""

    FPS            = 30
    BASE_AMPLITUDE = 0.30
    WAVELENGTH     = 40
    PHASE_SPEED    = 10
    GAIN_NOISE     = 1
    SMOOTH         = 0.85
    BASE_COLOR     = (60, 140, 255)
    NOISE_COLOR    = (255, 70, 0)

    def __init__(self, display):
        super().__init__(daemon=True)
        self.display     = display
        self.stop_event  = threading.Event()
        self.level_raw   = 0.0
        self.level       = 0.0
        self.phase       = 0.0
        self.x_vals      = np.arange(WIDTH, dtype=np.float32)

    def push_level(self, lvl):
        self.level_raw = max(0.0, min(lvl, 1.0))

    def stop(self):
        self.stop_event.set()

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

            self.phase += self.PHASE_SPEED
            base_wave = np.sin((self.x_vals / self.WAVELENGTH) + self.phase) * self.BASE_AMPLITUDE

            self.level = self.SMOOTH * self.level + (1 - self.SMOOTH) * self.level_raw

            if self.level > 0.02:
                noise_strength = self.GAIN_NOISE * self.level**1.5
                noise = np.random.normal(0.0, noise_strength, size=WIDTH)
                wave  = np.clip(base_wave + noise, -1.0, 1.0)
            else:
                wave = base_wave

            img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            cy   = HEIGHT // 2

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
# Audio y reconocimiento de voz
# =============================================

def init_vosk():
    """Inicializa el modelo Vosk"""
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en {VOSK_MODEL_PATH}")
    return vosk.Model(VOSK_MODEL_PATH)

def hablar_con_elevenlabs(texto):
    """Genera audio con ElevenLabs y visualización"""
    global is_speaking
    is_speaking = True
    vis_thread = BrutusVisualizer(display=display) if display else None
    
    try:
        if vis_thread:
            vis_thread.start()
            
        print(f"[ELEVENLABS] Generando: {texto}")
        
        # Generar audio con ElevenLabs
        audio = generate(
            text=texto,
            voice=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL_ID
        )
        
        # Reproducir audio
        play(audio)
        
        # Simular visualización durante reproducción
        if vis_thread:
            duration = len(texto) * 0.08  # Duración aproximada
            steps = int(duration * 30)  # 30 FPS
            for i in range(steps):
                level = 0.5 + 0.3 * np.sin(i * 0.5)
                vis_thread.push_level(level)
                time.sleep(1/30)
                
    except Exception as e:
        print(f"[ERROR] ElevenLabs falló: {e}")
        # Fallback a espeak
        try:
            subprocess.run(['espeak', '-v', 'es', texto], check=True)
        except:
            print(f"[FALLBACK] {texto}")
    finally:
        if vis_thread:
            vis_thread.stop()
            vis_thread.join()
        is_speaking = False

def hablar_fallback(texto):
    """Fallback usando espeak"""
    global is_speaking
    is_speaking = True
    try:
        subprocess.run(['espeak', '-v', 'es', texto], check=True)
    except:
        print(f"[FALLBACK] {texto}")
    finally:
        is_speaking = False

# Seleccionar función de habla
if ELEVENLABS_API_KEY:
    hablar = hablar_con_elevenlabs
else:
    hablar = hablar_fallback

def generar_respuesta(texto_usuario):
    """Genera respuesta usando OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un androide discutidor sarcástico pero útil. Responde de forma breve y con humor."},
                {"role": "user", "content": texto_usuario}
            ],
            max_tokens=100,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI falló: {e}")
        return "Lo siento, mi cerebro androide está procesando otras quejas."

# =============================================
# Función principal
# =============================================

def main():
    """Función principal del script"""
    print("🤖 Iniciando androide discutidor...")
    
    # Verificar API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY no encontrada")
        return
    
    if not os.getenv("PICOVOICE_ACCESS_KEY"):
        print("[ERROR] PICOVOICE_ACCESS_KEY no encontrada") 
        return
    
    try:
        # Inicializar Vosk
        print("[INFO] Cargando modelo Vosk...")
        model = init_vosk()
        recognizer = vosk.KaldiRecognizer(model, MIC_RATE)
        
        # Inicializar Picovoice
        print("[INFO] Inicializando Picovoice...")
        porcupine = pvporcupine.create(
            access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            model_path="/home/jack/picovoice/porcupine_params_es.pv",
            keyword_paths=[KEYWORD_PATH]
        )
        
        # Inicializar PyAudio
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )
        
        print("🎤 Escuchando wake word 'Botijo'...")
        hablar("Sistema iniciado. Di 'Botijo' para activarme.")
        
        # Loop principal
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            
            # Detectar wake word
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("🔔 Wake word detectado!")
                hablar("¿Qué quieres ahora?")
                
                # Escuchar comando
                print("🎤 Escuchando comando...")
                frames = []
                silence_count = 0
                max_silence = 50  # ~3 segundos
                
                while silence_count < max_silence:
                    pcm = audio_stream.read(porcupine.frame_length)
                    frames.append(pcm)
                    
                    # Detectar silencio
                    audio_data = np.frombuffer(pcm, dtype=np.int16)
                    if np.max(np.abs(audio_data)) < 500:
                        silence_count += 1
                    else:
                        silence_count = 0
                
                # Procesar audio grabado
                audio_data = b''.join(frames)
                
                # Reconocimiento con Vosk
                if recognizer.AcceptWaveform(audio_data):
                    result = json.loads(recognizer.Result())
                    texto = result.get('text', '').strip()
                    
                    if texto:
                        print(f"👤 Usuario: {texto}")
                        
                        # Generar y decir respuesta
                        respuesta = generar_respuesta(texto)
                        print(f"🤖 Botijo: {respuesta}")
                        hablar(respuesta)
                    else:
                        hablar("No te he entendido. ¿Puedes repetir?")
                
                # Reset recognizer
                recognizer = vosk.KaldiRecognizer(model, MIC_RATE)
    
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
    except Exception as e:
        print(f"[ERROR] Error en main: {e}")
    finally:
        try:
            if 'audio_stream' in locals():
                audio_stream.close()
            if 'pa' in locals():
                pa.terminate()
            if 'porcupine' in locals():
                porcupine.delete()
            if display:
                display.module_exit()
        except Exception:
            pass
        print("Sistema detenido.")

if __name__ == "__main__":
    main()
