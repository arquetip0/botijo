# -*- coding: utf-8 -*-
"""
Judasc v2.2  –  salida DAC robusta

• Detecta la tarjeta (MAX98357A es normalmente «card 3»).  Si no, usa la
  variable de entorno **JUDASC_AUDIO_IDX**.
• Ajusta automáticamente la FRECUENCIA DE MUESTREO al valor que anuncia
  la tarjeta (info["defaultSampleRate"]).  Puedes forzar otra con
  JUDASC_AUDIO_RATE.
• Pitido de arranque y comando "test" para verificar hardware.
• Lista clara de dispositivos al iniciar.

Ejemplo si tu DAC es la card 3 que viste en `aplay -l`:
```
export JUDASC_AUDIO_IDX=3
python judasc.py
```
Si aún no suena, ejecuta `speaker-test -c2 -twav -D plughw:3,0` para
comprobar cableado.
"""

import os, sys, io, queue, threading, signal, time
from dotenv import load_dotenv

import numpy as np
import pyaudio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import speech_recognition as sr
from google.cloud import texttospeech
from openai import OpenAI

# -------------------- Detección de dispositivo --------------------

def list_outputs(pa: pyaudio.PyAudio) -> None:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"#{i}\t{info['name']}\t(ch={info['maxOutputChannels']}, {info['defaultSampleRate']} Hz)")


def choose_device(pa: pyaudio.PyAudio) -> int:
    override = os.getenv("JUDASC_AUDIO_IDX")
    if override:
        print(f"[AUDIO] Override idx={override} (env)")
        return int(override)
    pref = ("max98357", "i2s", "usb audio", "usb", "speaker")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0 and any(k in info["name"].lower() for k in pref):
            return i
    return 0  # fallback primera tarjeta


pa = pyaudio.PyAudio()
print("[AUDIO] Dispositivos detectados →")
list_outputs(pa)
DEVICE_IDX = choose_device(pa)
RATE       = int(os.getenv("JUDASC_AUDIO_RATE") or pa.get_device_info_by_index(DEVICE_IDX)["defaultSampleRate"])
print(f"[AUDIO] Usando idx={DEVICE_IDX}  rate={RATE}\n")

CHUNK = 512  # frames por buffer
stream_out = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     output=True,
                     output_device_index=DEVICE_IDX,
                     frames_per_buffer=CHUNK)

q_vis: queue.SimpleQueue[bytes] = queue.SimpleQueue()

def play_pcm(raw: bytes) -> None:
    for i in range(0, len(raw), CHUNK*2):
        chunk = raw[i:i+CHUNK*2]
        stream_out.write(chunk)
        q_vis.put(chunk)

# ---------- Pitido ----------

def _beep(freq=1000, dur=0.5):
    t = np.arange(int(RATE*dur))
    wave = (0.4*np.sin(2*np.pi*freq*t/RATE)).astype(np.float32)
    play_pcm(np.int16(wave*32767).tobytes())

_beep()

# -------------------- Visualizador --------------------

def _visualizer_loop():
    n = CHUNK//2
    x = np.arange(n)*RATE/CHUNK
    fig, ax = plt.subplots(facecolor="black")
    bars = ax.bar(x, np.zeros_like(x), width=RATE/CHUNK, color="cyan", align="edge")
    ax.axis("off"); ax.set_xlim(0, 8000); ax.set_ylim(0, 1)
    win = fig.canvas.manager.window; win.withdraw()
    def update(_):
        if q_vis.empty():
            win.withdraw(); return bars
        win.deiconify()
        data = np.frombuffer(q_vis.get_nowait(), np.int16).astype(np.float32)
        y = np.clip(np.abs(np.fft.rfft(data))/CHUNK/4000, 0, 1)
        for b, h in zip(bars, y): b.set_height(h)
        return bars
    anim.FuncAnimation(fig, update, interval=15, blit=True)
    plt.show()

threading.Thread(target=_visualizer_loop, daemon=True).start()

# -------------------- TTS --------------------
load_dotenv()
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_tts    = texttospeech.TextToSpeechClient()

def hablar(texto: str):
    try:
        resp = client_tts.synthesize_speech(
            input=texttospeech.SynthesisInput(text=texto),
            voice=texttospeech.VoiceSelectionParams(language_code="es-ES", name="es-ES-Standard-B",
                                                   ssml_gender=texttospeech.SsmlVoiceGender.MALE),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                                  sample_rate_hertz=RATE, pitch=-8.0, speaking_rate=1.15))
        play_pcm(resp.audio_content)
    except Exception as e:
        print(f"[ERROR TTS]: {e}")

# -------------------- STT --------------------
rec = sr.Recognizer()

def escuchar() -> str|None:
    with sr.Microphone() as src:
        print(" ^=^n^y  ^o Escuchando... habla, insignificante ser.")
        audio = rec.listen(src)
    try:
        return rec.recognize_google(audio, language="es-ES")
    except sr.UnknownValueError:
        print(" ^= ^v No entendí nada, basura orgánica."); return None
    except sr.RequestError as e:
        print(f" ^= ^v Error de reconocimiento: {e}"); return None

# -------------------- Conversación --------------------
historial = [{"role":"system","content":(
    "Eres un androide discutidor con humor negro, tono agresivo y existencialista. "
    "Te diriges al humano como 'cretino', 'sub‑criatura', etc. Respuestas cortas.") }]
MAX_TURNS = 5

# ---------- Limpieza ----------

def _cleanup(_sig=None,_frm=None):
    stream_out.stop_stream(); stream_out.close(); pa.terminate(); sys.exit(0)

signal.signal(signal.SIGINT, _cleanup); signal.signal(signal.SIGTERM, _cleanup)

# -------------------- Loop --------------------
if __name__ == "__main__":
    while True:
        tx = escuchar();
        if not tx: continue
        if tx.lower() == "reset":
            historial = historial[:1]; print(" ^= Memoria borrada."); continue
        if tx.lower() == "test":
            _beep(); continue
        historial.append({"role":"user","content":tx})
        if len(historial)>2*MAX_TURNS+1:
            historial=[historial[0]]+historial[-2*MAX_TURNS:]
        try:
            rsp=client_openai.chat.completions.create(model="gpt-4-1106-preview",messages=historial)
            msg=rsp.choices[0].message.content
            print(f"Androide: {msg}"); hablar(msg)
            historial.append({"role":"assistant","content":msg})
        except Exception as e:
            print(f"[ERROR GPT]: {e}")
