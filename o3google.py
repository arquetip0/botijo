# 03google.py
# Script del androide discutidor con Google STT (streaming), ChatGPT y ElevenLabs TTS
# Visualización avanzada: multi‑onda estilo “Proyecto‑M retro” (ecos, color dinámico)
# Pantalla Waveshare 1.9" en landscape (320×170)
# Versión 2025‑06‑09‑google-stt-streaming

import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw
import pyaudio
from google.cloud import speech
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
import io
import threading
import queue

# — Librería Waveshare —
from lib import LCD_1inch9  # Asegúrate de que la carpeta "lib" esté junto al script

# ========================
# Configuración general
# ========================
load_dotenv()

# OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ElevenLabs API
api_key_value = "sk_9144204b0612eed1d5ad1615acd1b840c4257091c343b51d"
eleven = ElevenLabs(api_key=api_key_value)
VOICE_ID = 'RnKqZYEeVQciORlpiCz0'

# Google Cloud Speech-to-Text
speech_client = speech.SpeechClient()

# Tasas de muestreo
MIC_RATE = 16000   # Microphone input rate
TTS_RATE = 22050   # ElevenLabs output rate

# Pantalla Waveshare
NATIVE_WIDTH, NATIVE_HEIGHT = 170, 320
ROTATION = 270
WIDTH, HEIGHT = NATIVE_HEIGHT, NATIVE_WIDTH
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
    disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
    disp.bl_DutyCycle(100)
    return disp

try:
    display = init_display()
except Exception as e:
    print(f"[DISPLAY] No disponible → {e}")
    display = None

# ---------- Hilo de visualización avanzado ----------
class VisualizerThread(threading.Thread):
    """Multi‑onda con ecos y color según volumen (estilo Proyecto‑M retro)."""

    WAVE_COUNT = 4
    DECAY = 0.75
    FPS = 30

    def __init__(self, display):
        super().__init__(daemon=True)
        self.display = display
        self.level_q = queue.Queue(maxsize=300)
        self.stop_event = threading.Event()
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

# ---------- Google Speech-to-Text Streaming ----------
def stream_audio_to_google_stt():
    """Stream audio from the microphone to Google STT and yield transcriptions."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_RATE // 10,  # 0.1-second chunks
    )

    def generator():
        while True:
            data = stream.read(MIC_RATE // 10, exception_on_overflow=False)
            yield speech.StreamingRecognizeRequest(audio_content=data)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=MIC_RATE,
        language_code="es-ES",  # Change to your desired language
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    try:
        responses = speech_client.streaming_recognize(config=streaming_config, requests=generator())
        for response in responses:
            for result in response.results:
                if result.is_final:
                    yield result.alternatives[0].transcript
    except Exception as e:
        print(f"[ERROR] Google STT streaming: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# ---------- ElevenLabs TTS ----------
def hablar(texto):
    """Generate and play audio using ElevenLabs."""
    global is_speaking
    is_speaking = True

    vis_thread = VisualizerThread(display=display) if display else None
    if vis_thread:
        vis_thread.start()

    try:
        audio_gen = eleven.text_to_speech.convert(
            text=texto,
            voice_id=VOICE_ID,
            model_id='eleven_multilingual_v2',
            output_format='mp3_22050_32'
        )
        mp3_bytes = b''.join(chunk for chunk in audio_gen if isinstance(chunk, bytes))
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(TTS_RATE).set_channels(1).set_sample_width(2)
        raw_data = audio.raw_data

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_RATE,
            output=True,
        )

        MAX_AMP = 32768.0
        for i in range(0, len(raw_data), 2048):
            chunk = raw_data[i:i+2048]
            stream.write(chunk)
            if vis_thread:
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(samples ** 2)) / MAX_AMP
                vis_thread.push_level(min(rms * 4.0, 1.0))

    except Exception as e:
        print(f"[ERROR] ElevenLabs TTS: {e}")
    finally:
        if vis_thread:
            vis_thread.stop()
            vis_thread.join()
        is_speaking = False

# =============================================
# Main Loop
# =============================================
def main():
    print("Esperando la palabra clave 'botijo'…")
    hablar("Sistema de voz iniciado. Di botijo para comenzar.")

    for text in stream_audio_to_google_stt():
        print(f"Usuario: {text}")

        if "botijo" in text.lower():
            hablar("¿Qué quieres ahora, humano?")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": text}],
                max_tokens=150,
            )
            respuesta = response.choices[0].message.content
            print(f"Androide: {respuesta}")
            hablar(respuesta)

if __name__ == "__main__":
    main()
