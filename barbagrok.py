#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barbacoa.py — Botijo Modo Barbacoa

✔ Sin wake word: escucha continuo con Google Cloud STT
✔ LLM: Grok (xAI) vía openai.OpenAI con base_url de x.ai
✔ TTS: ElevenLabs (streaming y no‑streaming) — NADA de Piper
✔ Animación: LEDs, pantalla Waveshare con visualizador Brutus
✔ Animatrónica: ojos (IMX500 + servos) y tentáculos (servos)
✔ Limpieza robusta: Ctrl+C, SIGTERM, atexit

Config por .env:
  XAI_API_KEY=...
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=RnKqZYEeVQciORlpiCz0   # opcional (default)
  STT_LANG=es-ES                              # opcional

Requisitos clave:
  pip install openai python-dotenv elevenlabs pyaudio numpy pillow pydub
  pip install adafruit-blinka adafruit-circuitpython-servokit adafruit-circuitpython-pca9685 rpi-lgpio
  pip install google-cloud-speech picamera2
  (y librería de la pantalla: lib/LCD_1inch9 incluida en el proyecto)
"""

# --- Imports estándar ---
import os
import sys
import io
import time
import math
import json
import atexit
import queue
import random
import signal
import threading

# --- Terceros ---
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw
import pyaudio
from pydub import AudioSegment

# Audio/HW visual
import board
import neopixel

# STT
from google.cloud import speech

# LLM (Grok vía x.ai compatible OpenAI)
from openai import OpenAI

# TTS ElevenLabs
from elevenlabs.client import ElevenLabs

# Pantalla Waveshare (ruta local del proyecto)
from lib import LCD_1inch9  # Asegúrate de tener lib/LCD_1inch9

# Ojos / cámara / servos
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from adafruit_servokit import ServoKit

# (Opcional) Setup externo si existe
try:
    from botijo_barbacoa import make_setup  # tu helper opcional
except Exception:
    def make_setup():
        return None

# ============================================================
# CARGA .env Y CLIENTES
# ============================================================
load_dotenv()

# --- xAI / Grok ---
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
if not XAI_API_KEY:
    print("❌ Falta XAI_API_KEY en .env")
    sys.exit(1)

try:
    grok = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    print("✅ Grok listo")
except Exception as e:
    print(f"❌ Error iniciando Grok: {e}")
    sys.exit(1)

# --- ElevenLabs ---
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "RnKqZYEeVQciORlpiCz0").strip()
if not ELEVEN_API_KEY:
    print("❌ Falta ELEVENLABS_API_KEY en .env")
    sys.exit(1)

eleven = ElevenLabs(api_key=ELEVEN_API_KEY)

# --- Google STT ---
STT_LANG = os.getenv("STT_LANG", "es-ES")
try:
    speech_client = speech.SpeechClient()
    print("✅ Google STT listo")
except Exception as e:
    print(f"❌ Error iniciando Google STT: {e}")
    speech_client = None

# ============================================================
# CONSTANTES Y ESTADO GLOBAL
# ============================================================
# LEDs (brazo)
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

# Pantalla Waveshare (170×320 nativo, rotada a landscape)
NATIVE_W, NATIVE_H = 170, 320
ROTATION = 270
WIDTH, HEIGHT = NATIVE_H, NATIVE_W  # 320×170
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

# Audio
TTS_RATE = 22_050
CHUNK_PLAY = 2048
STT_RATE = 16_000
STT_CHUNK = int(STT_RATE * 0.03)  # ~30 ms

# Ojos / servos
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.2
SERVO_CHANNEL_LR = 0
SERVO_CHANNEL_UD = 1
SERVO_CHANNELS_EYELIDS = {"TL": 2, "BL": 3, "TR": 4, "BR": 5}
EYELID_POS = {"TL": 50, "BL": 155, "TR": 135, "BR": 25}
LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 30, 150
BLINK_P = 0.01
BLINK_DUR = 0.12
DBL_BLINK_P = 0.3
RAND_LOOK_P = 0.01
RAND_LOOK_DUR = 0.1
MICRO_MOVE_P = 0.15
MICRO_RANGE = 3
SMOOTH = 0.0
SQUINT_P = 0.02
SQUINT_DUR = 1.5
SQUINT_INT = 0.6
BREATH = True
BREATH_CYCLE = 4.0
BREATH_INT = 0.15

# Tentáculos (orejas)
TENTACLE_LEFT_CH = 15
TENTACLE_RIGHT_CH = 12

# Timers
INACTIVITY_TIMEOUT = 300
WARNING_TIME = 240

# Estado global
system_shutdown = False
is_speaking = False
led_thread_running = False
active_visualizer_threads = []
previous_lr = 90
previous_ud = 90

tentacles_active = False
eyes_active = False
led_thread = None
eyes_tracking_thread = None
picam2 = None
imx500 = None
kit = None

# ============================================================
# Utilidades
# ============================================================

def pulso_oxidado(t, i):
    intensidad = (math.sin(t + i * 0.5) + 1) / 2
    base = PALETA[i % len(PALETA)]
    return tuple(int(c * intensidad) for c in base)

# ============================================================
# LEDs
# ============================================================

def _led_loop(delay=0.04):
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
    except Exception:
        pass
    finally:
        for b in reversed(range(10)):
            pixels.brightness = b / 10.0
            pixels.show()
            time.sleep(0.05)
        pixels.fill((0, 0, 0))
        pixels.show()

def iniciar_luces():
    global led_thread_running, led_thread
    if led_thread_running:
        return
    led_thread_running = True
    led_thread = threading.Thread(target=_led_loop, daemon=True)
    led_thread.start()

def apagar_luces():
    global led_thread_running, led_thread
    try:
        led_thread_running = False
        if led_thread and led_thread.is_alive():
            led_thread.join(timeout=2)
        pixels.fill((0, 0, 0))
        pixels.show()
    except Exception as e:
        print(f"[LED] No se pudieron apagar: {e}")

# ============================================================
# Pantalla + Visualizador Brutus
# ============================================================

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
    try:
        disp.bl_DutyCycle(100)  # 100 = off en tu panel (ajusta si es al revés)
    except Exception:
        pass
    return disp

try:
    display = init_display()
except Exception as e:
    print(f"[DISPLAY] No disponible → {e}")
    display = None

class BrutusVisualizer(threading.Thread):
    FPS = 20
    BASE_AMP = 0.30
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
        global active_visualizer_threads, system_shutdown
        if not self.display:
            return
        active_visualizer_threads.append(self)
        frame_t = 1.0 / self.FPS
        last = 0.0
        try:
            while not self.stop_event.is_set() and not system_shutdown:
                now = time.time()
                if now - last < frame_t:
                    time.sleep(0.001)
                    continue
                self.phase += self.PHASE_SPEED
                base_wave = np.sin((self.x_vals / self.WAVELENGTH) + self.phase) * self.BASE_AMP
                self.level = self.SMOOTH * self.level + (1 - self.SMOOTH) * self.level_raw
                if self.level > 0.02:
                    noise_strength = self.GAIN_NOISE * (self.level ** 1.5)
                    noise = np.random.normal(0.0, noise_strength, size=WIDTH)
                    wave = np.clip(base_wave + noise, -1.0, 1.0)
                else:
                    wave = base_wave
                try:
                    if system_shutdown or self.stop_event.is_set():
                        break
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
                    if not system_shutdown and not self.stop_event.is_set() and self.display:
                        self.display.ShowImage(img)
                except Exception as e:
                    print(f"[VIS] {e}")
                    break
                last = now
        finally:
            try:
                active_visualizer_threads.remove(self)
            except ValueError:
                pass
            try:
                if not system_shutdown and self.display:
                    self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            except Exception:
                pass

# ============================================================
# Google STT streaming
# ============================================================

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
        finally:
            self.closed = True
            self._buff.put(None)
            if self._audio_interface:
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

def listen_for_command_google() -> str | None:
    if not speech_client:
        time.sleep(1)
        return None
    global is_speaking
    if is_speaking:
        while is_speaking:
            time.sleep(0.05)
        time.sleep(0.2)

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=STT_RATE,
        language_code=STT_LANG,
        enable_automatic_punctuation=True,
        use_enhanced=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=False,
        single_utterance=False,
    )

    print("[INFO] 🎤 Escuchando…")
    with MicrophoneStream(STT_RATE, STT_CHUNK) as stream:
        audio_gen = stream.generator()

        def request_gen():
            for content in audio_gen:
                if is_speaking:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=content)

        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=request_gen())
            for response in responses:
                if is_speaking:
                    return None
                if response.results and response.results[0].alternatives:
                    text = response.results[0].alternatives[0].transcript.strip()
                    if text:
                        return text
        except Exception as e:
            print(f"[STT] {e}")
            return None
    return None

# ============================================================
# TTS ElevenLabs
# ============================================================

def hablar(texto: str):
    global is_speaking
    is_speaking = True
    vis = BrutusVisualizer(display) if display else None
    if vis:
        vis.start()
    pa = None
    stream = None
    try:
        # Convert (no streaming), 22.05 kHz MP3 -> PCM
        audio_gen = eleven.text_to_speech.convert(
            text=texto,
            voice_id=VOICE_ID,
            model_id='eleven_flash_v2_5',
            output_format='mp3_22050_32'
        )
        mp3_bytes = b''.join(chunk for chunk in audio_gen if isinstance(chunk, bytes))
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(TTS_RATE).set_channels(1).set_sample_width(2)
        raw = audio.raw_data

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=TTS_RATE, output=True, frames_per_buffer=CHUNK_PLAY)
        MAX_AMP = 32768.0
        off = 0
        total = len(raw)
        while off < total:
            chunk = raw[off:off + CHUNK_PLAY * 2]
            off += CHUNK_PLAY * 2
            stream.write(chunk, exception_on_underflow=False)
            if vis and chunk:
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples ** 2)) / MAX_AMP)
                    vis.push_level(min(rms * 4.0, 1.0))
    except Exception as e:
        print(f"[TTS] {e}")
    finally:
        try:
            if stream:
                stream.stop_stream(); stream.close()
        finally:
            if pa:
                pa.terminate()
        if vis:
            vis.stop(); vis.join()
        is_speaking = False

def hablar_en_stream(response_stream):
    global is_speaking
    is_speaking = True
    vis = BrutusVisualizer(display) if display else None
    if vis:
        vis.start()
    pa = None
    stream = None
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=1024)
        MAX_AMP = 32768.0

        def synth_and_play(text):
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
                if vis and audio_chunk:
                    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    if samples.size:
                        rms = float(np.sqrt(np.mean(samples ** 2)) / MAX_AMP)
                        vis.push_level(min(rms * 4.0, 1.0))

        full, buf = "", ""
        print("🤖 Androide: ", end="", flush=True)
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            full += delta
            buf += delta
            print(delta, end="", flush=True)
            if any(p in delta for p in [".", "!", "?", "\n"]):
                synth_and_play(buf.strip())
                buf = ""
        if buf.strip():
            synth_and_play(buf.strip())
        print()
    except Exception as e:
        print(f"[TTS-STREAM] {e}")
    finally:
        try:
            if stream and stream.is_active():
                stream.stop_stream(); stream.close()
        finally:
            if pa:
                pa.terminate()
        if vis:
            vis.stop(); vis.join()
        is_speaking = False

# ============================================================
# OJOS (servos + IMX500)
# ============================================================

def map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (float(x - in_min) / float(in_max - in_min)) * (out_max - out_min)

def _smooth(cur, tgt, factor):
    return cur + (tgt - cur) * (1 - factor)

def _micro(angle, rng=MICRO_RANGE):
    if random.random() < MICRO_MOVE_P:
        return angle + random.uniform(-rng, rng)
    return angle

def _breath_offset():
    if not BREATH:
        return 0
    t = (time.time() % BREATH_CYCLE) / BREATH_CYCLE
    return math.sin(t * 2 * math.pi) * BREATH_INT

def process_detections(outputs, threshold=0.2):
    try:
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]
        num = int(outputs[3][0][0])
        dets = []
        for i in range(min(num, len(scores))):
            s = float(scores[i]); cid = int(classes[i])
            if s >= threshold:
                x1, y1, x2, y2 = boxes[i]
                dets.append({"class_id": cid, "confidence": s, "bbox": (float(x1), float(y1), float(x2), float(y2))})
        return dets
    except Exception as e:
        print(f"[EYES] parse: {e}")
        return []

def close_eyelids():
    if not kit: return
    for _, ch in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[ch].set_pulse_width_range(500, 2500)
            kit.servo[ch].angle = 90
        except Exception as e:
            print(f"[EYES] close lid: {e}")

def init_eyelids():
    if not kit: return
    for key, ch in SERVO_CHANNELS_EYELIDS.items():
        try:
            kit.servo[ch].set_pulse_width_range(500, 2500)
            kit.servo[ch].angle = EYELID_POS[key]
        except Exception as e:
            print(f"[EYES] init lid {key}: {e}")

def blink():
    if not kit or not eyes_active: return
    for _, ch in SERVO_CHANNELS_EYELIDS.items():
        try: kit.servo[ch].angle = 90
        except Exception: pass
    time.sleep(BLINK_DUR)
    for key, ch in SERVO_CHANNELS_EYELIDS.items():
        try: kit.servo[ch].angle = EYELID_POS[key] + _breath_offset()
        except Exception: pass
    if random.random() < DBL_BLINK_P:
        time.sleep(0.1)
        for _, ch in SERVO_CHANNELS_EYELIDS.items():
            try: kit.servo[ch].angle = 90
            except Exception: pass
        time.sleep(BLINK_DUR * 0.8)
        for key, ch in SERVO_CHANNELS_EYELIDS.items():
            try: kit.servo[ch].angle = EYELID_POS[key] + _breath_offset()
            except Exception: pass

def squint():
    if not kit or not eyes_active: return
    for key, ch in SERVO_CHANNELS_EYELIDS.items():
        try:
            base = EYELID_POS[key]
            sq = base + (90 - base) * SQUINT_INT
            kit.servo[ch].angle = sq
        except Exception: pass
    time.sleep(SQUINT_DUR)
    steps = 5
    for step in range(steps):
        for key, ch in SERVO_CHANNELS_EYELIDS.items():
            try:
                base = EYELID_POS[key]
                sq = base + (90 - base) * SQUINT_INT
                prog = (step + 1) / steps
                cur = sq + (base - sq) * prog
                kit.servo[ch].angle = cur + _breath_offset()
            except Exception: pass
        time.sleep(0.1)

def update_eyelids_with_breathing():
    if not kit or not eyes_active or not BREATH: return
    off = _breath_offset() * 5
    for key, ch in SERVO_CHANNELS_EYELIDS.items():
        try:
            base = EYELID_POS[key]
            ang = max(10, min(170, base + off))
            kit.servo[ch].angle = ang
        except Exception: pass

def should_blink():
    return random.random() < BLINK_P

def should_look_random():
    return random.random() < RAND_LOOK_P

def should_squint():
    return random.random() < SQUINT_P

def initialize_eye_servos():
    global previous_lr, previous_ud
    if not kit: return
    try:
        kit.servo[SERVO_CHANNEL_LR].set_pulse_width_range(500, 2500)
        kit.servo[SERVO_CHANNEL_UD].set_pulse_width_range(500, 2500)
        lr0 = (LR_MIN + LR_MAX) // 2
        ud0 = (UD_MIN + UD_MAX) // 2
        kit.servo[SERVO_CHANNEL_LR].angle = lr0
        kit.servo[SERVO_CHANNEL_UD].angle = ud0
        previous_lr, previous_ud = lr0, ud0
    except Exception as e:
        print(f"[EYES] init servos: {e}")

def center_eyes():
    global previous_lr, previous_ud
    if not kit: return
    try:
        lr = (LR_MIN + LR_MAX) // 2
        ud = (UD_MIN + UD_MAX) // 2
        kit.servo[SERVO_CHANNEL_LR].angle = lr
        kit.servo[SERVO_CHANNEL_UD].angle = ud
        previous_lr, previous_ud = lr, ud
    except Exception as e:
        print(f"[EYES] center: {e}")

def look_random():
    global previous_lr, previous_ud
    if not kit or not eyes_active: return
    rand_lr = max(LR_MIN, min(LR_MAX, _micro(random.uniform(LR_MIN, LR_MAX))))
    rand_ud = max(UD_MIN, min(UD_MAX, _micro(random.uniform(UD_MIN, UD_MAX))))
    steps = 8
    for s in range(steps):
        if not eyes_active: return
        prog = (s + 1) / steps
        cur_lr = previous_lr + (rand_lr - previous_lr) * prog
        cur_ud = previous_ud + (rand_ud - previous_ud) * prog
        try:
            kit.servo[SERVO_CHANNEL_LR].angle = cur_lr
            kit.servo[SERVO_CHANNEL_UD].angle = cur_ud
        except Exception: pass
        time.sleep(0.05)
    previous_lr, previous_ud = rand_lr, rand_ud
    hold = int(RAND_LOOK_DUR / 0.1)
    for _ in range(hold):
        if not eyes_active: return
        try:
            kit.servo[SERVO_CHANNEL_LR].angle = _micro(rand_lr, 1)
            kit.servo[SERVO_CHANNEL_UD].angle = _micro(rand_ud, 1)
        except Exception: pass
        time.sleep(0.1)

def init_camera_system():
    global picam2, imx500
    try:
        imx500 = IMX500(RPK_PATH)
        intr = imx500.network_intrinsics or NetworkIntrinsics()
        intr.task = "object detection"
        intr.threshold = THRESHOLD
        intr.iou = 0.5
        intr.max_detections = 5
        try:
            with open(LABELS_PATH, 'r') as f:
                intr.labels = [ln.strip() for ln in f.readlines()]
        except FileNotFoundError:
            intr.labels = ["face"]
        intr.update_with_defaults()
        picam2 = Picamera2(imx500.camera_num)
        cfg = picam2.create_preview_configuration(buffer_count=12, controls={"FrameRate": intr.inference_rate})
        picam2.configure(cfg)
        imx500.show_network_fw_progress_bar()
        picam2.start()
        time.sleep(3)
        print("✅ Cámara/IMX500 OK")
        return True
    except Exception as e:
        print(f"[EYES] cam init: {e}")
        picam2 = None; imx500 = None
        return False

def shutdown_camera_system():
    global picam2, imx500
    try:
        if picam2:
            picam2.stop(); picam2 = None
        imx500 = None
        print("📷 Cámara apagada")
    except Exception as e:
        print(f"[EYES] cam shutdown: {e}")

class EyeTrackingThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        global previous_lr, previous_ud
        print("🚀 Seguimiento facial ON")
        iteration = 0
        consecutive_no = 0
        last_breath = time.time()
        intr = imx500.network_intrinsics or NetworkIntrinsics()
        if not getattr(intr, 'labels', None):
            intr.labels = ["face"]
        while not self.stop_event.is_set() and eyes_active and not system_shutdown:
            try:
                iteration += 1
                if time.time() - last_breath > 0.2:
                    update_eyelids_with_breathing()
                    last_breath = time.time()
                if should_blink():
                    blink()
                elif should_squint():
                    squint()
                elif should_look_random():
                    look_random()
                if not picam2 or not imx500:
                    time.sleep(0.1); continue
                try:
                    metadata = picam2.capture_metadata(wait=True)
                except Exception as e:
                    time.sleep(0.1); continue
                outputs = imx500.get_outputs(metadata, add_batch=True)
                if outputs is None:
                    consecutive_no += 1
                    if consecutive_no > 100:
                        time.sleep(1); consecutive_no = 0
                    else:
                        time.sleep(0.05)
                    continue
                consecutive_no = 0
                dets = process_detections(outputs, THRESHOLD)
                faces = [d for d in dets if intr.labels[d["class_id"]] == "face"]
                if not faces:
                    time.sleep(0.1); continue
                best = max(faces, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = best["bbox"]
                xc = (x1 + x2) / 2; yc = (y1 + y2) / 2
                iw, ih = imx500.get_input_size()
                tgt_lr = max(LR_MIN, min(LR_MAX, map_range(xc, 0, iw, LR_MAX, LR_MIN)))
                tgt_ud = max(UD_MIN, min(UD_MAX, map_range(yc, 0, ih, UD_MAX, UD_MIN)))
                sm_lr = _smooth(previous_lr, tgt_lr, SMOOTH)
                sm_ud = _smooth(previous_ud, tgt_ud, SMOOTH)
                fin_lr = max(LR_MIN, min(LR_MAX, _micro(sm_lr)))
                fin_ud = max(UD_MIN, min(UD_MAX, _micro(sm_ud)))
                if kit and eyes_active:
                    try:
                        kit.servo[SERVO_CHANNEL_LR].angle = fin_lr
                        kit.servo[SERVO_CHANNEL_UD].angle = fin_ud
                        previous_lr, previous_ud = fin_lr, fin_ud
                    except Exception:
                        pass
                time.sleep(0.05)
            except Exception as e:
                time.sleep(0.1)
        print("🛑 Seguimiento facial OFF")


def activate_eyes():
    global eyes_active, eyes_tracking_thread
    if eyes_active:
        return
    initialize_eye_servos()
    init_eyelids()
    if init_camera_system():
        eyes_active = True
        eyes_tracking_thread = EyeTrackingThread()
        eyes_tracking_thread.start()
        print("✅ Ojos activos")
    else:
        print("❌ Error activando ojos")


def deactivate_eyes():
    global eyes_active, eyes_tracking_thread
    if not eyes_active:
        return
    eyes_active = False
    if eyes_tracking_thread:
        eyes_tracking_thread.stop()
        eyes_tracking_thread.join(timeout=2)
        eyes_tracking_thread = None
    shutdown_camera_system()
    center_eyes(); time.sleep(0.3); close_eyelids()
    print("✅ Ojos desactivados")

# ============================================================
# TENTÁCULOS (orejas)
# ============================================================

def initialize_tentacles():
    if not kit: return
    try:
        kit.servo[TENTACLE_LEFT_CH].set_pulse_width_range(min_pulse=300, max_pulse=2650)
        kit.servo[TENTACLE_RIGHT_CH].set_pulse_width_range(min_pulse=450, max_pulse=2720)
        kit.servo[TENTACLE_LEFT_CH].angle = 90
        kit.servo[TENTACLE_RIGHT_CH].angle = 90
        print("🐙 Tentáculos inicializados")
    except Exception as e:
        print(f"[TENT] init: {e}")

def stop_tentacles():
    if not kit: return
    try:
        kit.servo[TENTACLE_LEFT_CH].angle = 90
        kit.servo[TENTACLE_RIGHT_CH].angle = 90
        time.sleep(0.3)
        kit.servo[TENTACLE_LEFT_CH].angle = None
        kit.servo[TENTACLE_RIGHT_CH].angle = None
    except Exception as e:
        print(f"[TENT] stop: {e}")

class TentacleThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        global tentacles_active
        angle_left, angle_right = 90, 90
        print("🐙 Tentáculos ON")
        while not self.stop_event.is_set() and tentacles_active and not system_shutdown:
            try:
                target_left = random.choice([random.randint(0, 30), random.randint(150, 180)])
                target_right = 180 - target_left
                step = random.randint(3, 7)
                rng_left = range(angle_left, target_left + 1, step) if target_left > angle_left else range(angle_left, target_left - 1, -step)
                rng_right = range(angle_right, target_right + 1, step) if target_right > angle_right else range(angle_right, target_right - 1, -step)
                for a_left, a_right in zip(rng_left, rng_right):
                    if not tentacles_active or self.stop_event.is_set():
                        return
                    if kit:
                        try:
                            kit.servo[TENTACLE_LEFT_CH].angle = a_left
                            kit.servo[TENTACLE_RIGHT_CH].angle = a_right
                        except Exception:
                            pass
                    time.sleep(random.uniform(0.005, 0.015))
                angle_left, angle_right = target_left, target_right
                pause = random.uniform(3, 6)
                t0 = time.time()
                while time.time() - t0 < pause:
                    if self.stop_event.is_set() or not tentacles_active:
                        return
                    time.sleep(0.1)
            except Exception as e:
                time.sleep(1)
        print("🛑 Tentáculos OFF")


tentacles_thread = None

def activate_tentacles():
    global tentacles_active, tentacles_thread
    if tentacles_active:
        return
    initialize_tentacles()
    tentacles_active = True
    tentacles_thread = TentacleThread()
    tentacles_thread.start()


def deactivate_tentacles():
    global tentacles_active, tentacles_thread
    if not tentacles_active:
        return
    tentacles_active = False
    if tentacles_thread:
        tentacles_thread.stop()
        tentacles_thread.join(timeout=2)
        tentacles_thread = None
    stop_tentacles()

# ============================================================
# EMERGENCY SHUTDOWN y señales
# ============================================================

def emergency_shutdown():
    global system_shutdown
    if system_shutdown:
        return
    system_shutdown = True
    print("\n🚨 EMERGENCY SHUTDOWN…")
    try:
        for vis in active_visualizer_threads[:]:
            try: vis.stop()
            except Exception: pass
        time.sleep(0.3)
    except Exception: pass
    try:
        deactivate_eyes()
    except Exception: pass
    try:
        deactivate_tentacles()
    except Exception: pass
    try:
        apagar_luces()
    except Exception: pass
    try:
        if display:
            display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            display.module_exit()
    except Exception: pass
    print("✅ Limpieza completada")


def _signal_handler(signum, frame):
    emergency_shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(emergency_shutdown)

# ============================================================
# ServoKit init
# ============================================================
try:
    kit = ServoKit(channels=16)
    print("✅ ServoKit listo")
except Exception as e:
    print(f"[SERVOS] {e}")
    kit = None

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global eyes_active
    conversation_history = [
        {
            "role": "system",
            "content": (
                "Eres un androide paranoide con voz irónica y sarcástica. "
                "Tu nombre es botijo. Hablas en voz (ElevenLabs), así que sé conciso. "
                "Mofas a humanos con humor ácido cuando procede. Dices 'zarria' cuando algo es cutre."
            ),
        }
    ]

    last_interaction = time.time()
    has_warned = False

    if kit:
        initialize_eye_servos(); time.sleep(0.3)

    iniciar_luces()
    activate_eyes()
    activate_tentacles()

    hablar("Soy Botijo. ¿Qué quieres ahora, ser inferior?")
    print("🎤 READY — habla directo")

    try:
        while True:
            inactive = time.time() - last_interaction
            if inactive > INACTIVITY_TIMEOUT:
                print("\n[SLEEP] Inactividad")
                hablar("Me aburres, humano. Me piro a dormir. Habla para reactivarme.")
                deactivate_eyes(); deactivate_tentacles()
                while True:
                    cmd = listen_for_command_google()
                    if cmd:
                        print(f"\n👂 Reactivado con: {cmd}")
                        activate_eyes(); activate_tentacles(); iniciar_luces()
                        hablar("Ah, has vuelto. Procesando…")
                        last_interaction = time.time(); has_warned = False
                        conversation_history.append({"role": "user", "content": cmd})
                        try:
                            stream = grok.chat.completions.create(
                                model="grok-3-fast",
                                messages=conversation_history,
                                temperature=1,
                                max_tokens=300,
                                stream=True,
                            )
                            hablar_en_stream(stream)
                        except Exception as e:
                            print(f"[GROK] {e}")
                            hablar("Error en el vacío cósmico. Revisa tu API key, mortal.")
                        break
            elif inactive > WARNING_TIME and not has_warned:
                hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                has_warned = True

            if not is_speaking:
                cmd = listen_for_command_google()
                if cmd:
                    last_interaction = time.time(); has_warned = False
                    print(f"\n👂 Humano: {cmd}")
                    conversation_history.append({"role": "user", "content": cmd})
                    try:
                        stream = grok.chat.completions.create(
                            model="grok-3-fast",
                            messages=conversation_history,
                            temperature=1,
                            max_tokens=300,
                            stream=True,
                        )
                        hablar_en_stream(stream)
                    except Exception as e:
                        print(f"[GROK] {e}")
                        hablar("Pantalla azul existencial. Revisa tu suscripción a x.ai.")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C")
        emergency_shutdown()
    finally:
        if not system_shutdown:
            emergency_shutdown()
        print("✅ Sistema detenido.")


if __name__ == "__main__":
    if speech_client:
        try:
            main()
        except Exception as e:
            print(f"\n💥 Fatal: {e}")
            try:
                import traceback; traceback.print_exc()
            except Exception:
                pass
            emergency_shutdown()
        finally:
            print("👋 Botijo desconectado.")
    else:
        print("❌ Google STT no está inicializado. Verifica credenciales y red.")
