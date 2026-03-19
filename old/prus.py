"""
o3_google.py – Versión con Google Speech‑to‑Text
------------------------------------------------
Reemplaza completamente el reconocimiento Vosk por Google STT usando
SpeechRecognition, manteniendo la síntesis ElevenLabs, el visualizador
Brutus y la pantalla Waveshare 1.9". Probado (10 pasadas) en RasPi 5.
"""

# ========================
# Imports generales
# ========================
import os, time, queue, math, threading
from collections import deque
from dotenv import load_dotenv
import speech_recognition as sr
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import io, pyaudio
from PIL import Image, ImageDraw

# --- Hardware Waveshare ---
from lib import LCD_1inch9   # Asegúrate de que está en PYTHONPATH

# ========================
# Cargar variables de entorno
# ========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID       = os.getenv("VOICE_ID", "RnKqZYEeVQciORlpiCz0")  # Voz grave robótica

# ========================
# Inicializar clientes
# ========================
client  = OpenAI(api_key=OPENAI_API_KEY)
eleven  = ElevenLabs(api_key=ELEVEN_API_KEY)

# ========================
# Constantes de audio
# ========================
PCM_RATE   = 22_050  # Visualizador & playback
FRAME_SIZE = 2048    # Ventana para FFT

# ========================
# Display (Waveshare 1.9")
# ========================
ROTATION = 2  # 0‑3
NATIVE_WIDTH, NATIVE_HEIGHT = 320, 170
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH  # tras rotación
RST_PIN, DC_PIN, BL_PIN = 27, 25, 24

def init_display():
    try:
        disp = LCD_1inch9.LCD_1inch9(rst=RST_PIN, dc=DC_PIN, bl=BL_PIN)
        disp.Init(); disp.set_rotation(ROTATION)
        return disp
    except Exception as e:
        print(f"[DISPLAY] No disponible → {e}")
        return None

display = init_display()

# ========================
# Visualizador: Brutus
# ========================
class BrutusVisualizer(threading.Thread):
    def __init__(self, display, columns=WIDTH):
        super().__init__(daemon=True)
        self.display = display
        self.columns = columns
        self.running = threading.Event(); self.running.set()
        self.buffers = deque([0]*columns, maxlen=columns)

    def add_samples(self, pcm):
        for i in range(0, len(pcm), 2):
            sample = int.from_bytes(pcm[i:i+2], "little", signed=True)
            lvl = abs(sample) / 32768
            self.buffers.append(lvl)

    def run(self):
        if not self.display: return
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        while self.running.is_set():
            draw.rectangle([0, 0, WIDTH, HEIGHT], fill="black")
            for x, lvl in enumerate(self.buffers):
                h = int(lvl * HEIGHT)
                draw.line([(x, HEIGHT//2-h//2), (x, HEIGHT//2+h//2)], fill="lime")
            self.display.ShowImage(img)
            time.sleep(0.016)  # ~60 FPS

    def stop(self):
        self.running.clear()

# Flag global para evitar bucles de realimentación
is_speaking = False

# ========================
# Función de síntesis y reproducción
# ========================

def hablar(texto: str):
    global is_speaking
    is_speaking = True

    vis = BrutusVisualizer(display)
    if display: vis.start()

    # --- ElevenLabs: MP3 22 kHz 32 kbps para mínima latencia ---
    audio_gen = eleven.text_to_speech.convert(
        text=texto,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_22050_32",
    )
    audio_mp3 = b"".join(chunk for chunk in audio_gen if isinstance(chunk, bytes))

    # Convertir a PCM 22050 Hz mono 16‑bit
    pcm = AudioSegment.from_file(io.BytesIO(audio_mp3), format="mp3")
    pcm = pcm.set_frame_rate(PCM_RATE).set_channels(1).set_sample_width(2)
    raw_data = pcm.raw_data

    # Enviar al visualizador
    if display: vis.add_samples(raw_data)

    # Reproducir con PyAudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=PCM_RATE, output=True)
    stream.write(raw_data)
    stream.stop_stream(); stream.close(); pa.terminate()

    if display: vis.stop(); vis.join()
    is_speaking = False

# ========================
# ChatGPT: obtener respuesta
# ========================

def responder(conversation_history):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[CHATGPT] {e}")
        return "Mis circuitos están fritos. Pregunta otra cosa."

# ========================
# SpeechRecognition (Google STT)
# ========================
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16_000)

with mic as src:
    recognizer.adjust_for_ambient_noise(src, duration=0.5)

# ========================
# Bucle principal
# ========================

def main():
    global is_speaking

    print("Esperando la palabra clave 'botijo'…")
    INACTIVITY_TIMEOUT = 120
    WARNING_TIME = 90
    has_warned = False
    is_conversation_active = False
    last_interaction_time = time.time()

    conversation_history = [
        {"role": "system", "content": "Eres un androide discutidor con voz agresiva y sarcástica. Responde siempre en español."}
    ]

    try:
        while True:
            now = time.time()
            inactive = now - last_interaction_time

            if is_speaking:
                time.sleep(0.1)
                continue

            # --- Escuchar con Google STT ---
            try:
                with mic as src:
                    audio = recognizer.listen(src, timeout=3, phrase_time_limit=12)
                texto = recognizer.recognize_google(audio, language="es-ES").lower()
                print(f"Humano: {texto}")
            except sr.WaitTimeoutError:
                texto = ""
            except sr.UnknownValueError:
                texto = ""
            except sr.RequestError as e:
                print(f"[GOOGLE-STT] Error: {e}"); texto = ""

            # --- Si no se dijo nada ---
            if not texto:
                if is_conversation_active:
                    if inactive > INACTIVITY_TIMEOUT:
                        hablar("Me aburres, humano. Vuelve cuando tengas algo interesante que decir.")
                        is_conversation_active = False; has_warned = False
                        conversation_history = [conversation_history[0]]
                    elif inactive > WARNING_TIME and not has_warned:
                        hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                        has_warned = True
                continue

            # --- Wake word ---
            if not is_conversation_active:
                if "botijo" in texto:
                    hablar("¡Botijo activado! Iniciando conversación…")
                    is_conversation_active = True
                    last_interaction_time = time.time()
                continue

            # --- Conversación normal ---
            last_interaction_time = time.time(); has_warned = False
            conversation_history.append({"role": "user", "content": texto}); conversation_history = conversation_history[-10:]
            respuesta = responder(conversation_history)
            conversation_history.append({"role": "assistant", "content": respuesta}); conversation_history = conversation_history[-10:]
            print(f"Androide: {respuesta}")
            hablar(respuesta)

    except KeyboardInterrupt:
        print("\nApagando…")
    finally:
        if display:
            try: display.module_exit()
            except Exception: pass
        print("Sistema detenido.")


if __name__ == "__main__":
    main()