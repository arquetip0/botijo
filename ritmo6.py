# grok3_waveshare_lib_fixed.py
# Script del androide discutidor con wake word, Vosk, ChatGPT y Piper
# Visualización avanzada: multi‑onda estilo “Proyecto‑M retro” (ecos, color dinámico)
# Pantalla Waveshare 1.9" en landscape (320×170)
# Versión 2025‑04‑26‑d
# ‑ Corrección de muestreo (22 050 Hz)                ✅
# ‑ Hilo de visualización separado                   ✅
# ‑ Nueva visualización multi‑onda con color volumen ✅

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

# — Librería Waveshare —
from lib import LCD_1inch9  # Asegúrate de que la carpeta "lib" esté junto al script

# ========================
# Configuración general
# ========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Rutas y configuración Vosk ‑ Piper
VOSK_MODEL_PATH = "model"
PIPER_BINARY     = "/home/jack/piper/build/piper"
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"

# Tasas de muestreo separadas
MIC_RATE   = 16000   # micro + Vosk
PIPER_RATE = 22050   # salida de Piper – IMPORTANTE

# — Pantalla Waveshare (170×320 nativo, girada a landscape) —
NATIVE_WIDTH, NATIVE_HEIGHT = 170, 320  # de fábrica
ROTATION = 270                         # 0=portrait, 270=gira reloj
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH  # 320×170 post‑rotación
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

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
    disp.clear()
    disp.bl_DutyCycle(50)
    return disp

try:
    display = init_display()
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
        self.level_q: queue.Queue[float] = queue.Queue(maxsize=300)
        self.stop_event = threading.Event()
        # buffer para cada onda (320 columnas)
        self.buffers = [np.zeros(WIDTH, dtype=np.float32) for _ in range(self.WAVE_COUNT)]
        self.last_draw = 0.0

    # — API pública —
    def push_level(self, level: float):
        try:
            self.level_q.put_nowait(level)
        except queue.Full:
            pass  # descartamos

    def stop(self):
        self.stop_event.set()

    # — Funciones internas —
    @staticmethod
    def _volume_to_color(level: float) -> tuple[int, int, int]:
        """De verde (bajo) a rojo (alto) pasando por amarillo."""
        r = int(255 * level)
        g = int(255 * (1.0 - 0.5 * level))  # se apaga hasta ~127
        return (r, g, 0)

    def _shift_buffers(self, new_level: float):
        # Desplaza derecha→izq buffers y añade nivel nuevo en cola
        for buf in self.buffers:
            buf[:-1] = buf[1:]
        self.buffers[0][-1] = new_level
        # Los ecos copian el valor previo atenuado
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
            # — dibujo —
            img = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            cy = HEIGHT // 2
            for idx, buf in enumerate(self.buffers):
                fade = (self.DECAY ** idx)
                # color base segun volumen del trazo actual (última muestra del buf)
                color = tuple(int(c * fade) for c in self._volume_to_color(buf[-1]))
                for x in range(WIDTH - 1):
                    y1 = int(cy - buf[x] * cy)
                    y2 = int(cy - buf[x + 1] * cy)
                    draw.line((x, y1, x + 1, y2), fill=color)
            self.display.ShowImage(img)
            self.last_draw = now
        # limpiar al salir
        self.display.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))

# =============================================
# Vosk y audio
# =============================================

def init_vosk():
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en {VOSK_MODEL_PATH}")
    return vosk.Model(VOSK_MODEL_PATH)

# ---------- Hablar ----------

audio_queue = []
is_speaking = False


def hablar(texto: str):
    """Sintetiza con Piper, reproduce a 22 050 Hz y envía niveles RMS al visualizador."""
    global is_speaking
    is_speaking = True
    vis_thread = VisualizerThread(display) if display else None
    if vis_thread:
        vis_thread.start()

    pa = None; stream = None; proc = None
    try:
        proc = subprocess.Popen(
            [PIPER_BINARY, "--model", PIPER_MODEL_PATH, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(texto.encode("utf‑8")); proc.stdin.close()

        pa = pyaudio.PyAudio(); CHUNK = 2048
        stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=PIPER_RATE,
            output=True, frames_per_buffer=CHUNK,
        )
        MAX_AMP = 32768.0
        while True:
            data = proc.stdout.read(CHUNK * 2)
            if not data: break
            stream.write(data, exception_on_underflow=False)
            if vis_thread:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if samples.size:
                    rms = np.sqrt(np.mean(samples ** 2)) / MAX_AMP
                    vis_thread.push_level(min(rms * 4.0, 1.0))

    except Exception as e:
        print(f"[HABLAR] {e}")
    finally:
        if stream:
            stream.stop_stream(); stream.close()
        if pa:
            pa.terminate()
        if proc:
            proc.terminate(); proc.wait()
        if vis_thread:
            vis_thread.stop(); vis_thread.join()
        is_speaking = False


# ---------- Hablar sin pantalla ----------

def hablar_sin_visualizacion(texto: str):
    global is_speaking
    is_speaking = True
    try:
        subprocess.run([
            PIPER_BINARY, "--model", PIPER_MODEL_PATH, "--output_file", "respuesta.wav"
        ], input=texto.encode("utf‑8"), check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        subprocess.run(["aplay", "-q", "-r", str(PIPER_RATE), "respuesta.wav"], check=True)
        os.remove("respuesta.wav")
    except Exception as e:
        print(f"[HABLAR‑NO‑VIS] {e}")
    finally:
        is_speaking = False


# ---------- Callback de audio ----------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO] {status}")
    if not is_speaking:
        audio_queue.append(bytes(indata))


# =============================================
# Bucle principal
# =============================================

def main():
    model = init_vosk()
    recognizer = vosk.KaldiRecognizer(model, MIC_RATE)

    stream = sd.RawInputStream(
        samplerate=MIC_RATE,
        blocksize=8000,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    )

    print("Esperando la palabra clave 'botijo'…")
    INACTIVITY_TIMEOUT = 120
    WARNING_TIME = 90
    has_warned = False
    is_conversation_active = False
    last_interaction_time = time.time()

    conversation_history = [
        {
            "role": "system",
            "content": "Eres un androide discutidor con voz agresiva y sarcástica. Responde siempre en español",
        }
    ]

    with stream:
        try:
            while True:
                now = time.time()
                inactive = now - last_interaction_time

                if audio_queue and not is_speaking:
                    audio_data = audio_queue.pop(0)
                    if recognizer.AcceptWaveform(audio_data):
                        text = json.loads(recognizer.Result()).get("text", "").lower()
                        if not text:
                            continue

                        if "botijo" in text or is_conversation_active:
                            last_interaction_time = now
                            if not is_conversation_active:
                                print("\n¡Botijo activado! Iniciando conversación…")
                                hablar("¿Qué quieres ahora, ser inferior?")
                                is_conversation_active = True
                                audio_queue.clear(); recognizer.Reset(); continue

                            if text != "botijo":
                                print(f"\nHumano: {text}")
                                conversation_history.append({"role": "user", "content": text})
                                try:
                                    response = client.chat.completions.create(
                                        model="gpt-4-1106-preview",
                                        messages=conversation_history,
                                        temperature=0.9,
                                        max_tokens=150,
                                        timeout=15,
                                    )
                                    respuesta = response.choices[0].message.content
                                except Exception as e:
                                    print(f"[CHATGPT] {e}"); respuesta = "Mis circuitos están fritos. Pregunta otra cosa."

                                conversation_history.append({"role": "assistant", "content": respuesta})
                                conversation_history = conversation_history[-10:]
                                print(f"Androide: {respuesta}")
                                hablar(respuesta)
                                audio_queue.clear(); recognizer.Reset()

                if is_conversation_active:
                    if inactive > INACTIVITY_TIMEOUT:
                        hablar("Me aburres, humano. Vuelve cuando tengas algo interesante que decir.")
                        is_conversation_active = False; has_warned = False
                        conversation_history = [conversation_history[0]]
                        audio_queue.clear(); recognizer.Reset()
                    elif inactive > WARNING_TIME and not has_warned:
                        hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                        has_warned = True; last_interaction_time = time.time()
                        audio_queue.clear(); recognizer.Reset()

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nApagando…")
        finally:
            stream.stop(); stream.close()
            if display:
                try:
                    display.module_exit()
                except Exception:
                    pass
            print("Sistema detenido.")


if __name__ == "__main__":
    main()
