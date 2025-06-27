# grok3_final.py
# Script androide con Vosk, ChatGPT, Piper, visualización vertical Waveshare
# y optimización de audio.

import os
import sys
import subprocess
import sounddevice as sd
import vosk
from openai import OpenAI
from dotenv import load_dotenv
import json
import threading
import time
import numpy as np
from lib import LCD_1inch9
from PIL import Image, ImageDraw, ImageFont
import pyaudio

# Configuración y constantes (igual que antes)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VOSK_MODEL_PATH = "model"
PIPER_BINARY = "/home/jack/piper/build/piper"
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"
SAMPLE_RATE = 16000
WIDTH, HEIGHT = 170, 320 # Resolución correcta
RST_PIN = 27
DC_PIN = 25
BL_PIN = 18 # Verifica este pin!

# Parámetros del visualizador
N_BARS = 16              # Reducido para optimizar un poco
BAR_GAP = 2
PEAK_DECAY = 0.05 # Un poco más rápido el decaimiento
VISUALIZATION_SKIP_FRAMES = 2 # Dibuja 1 de cada X frames de audio

# Sistema de audio
audio_queue = []
is_speaking = False

# --- Inicializar display (Sin cambios) ---
def init_display():
    disp = LCD_1inch9.LCD_1inch9(rst=RST_PIN, dc=DC_PIN, bl=BL_PIN)
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(50)
    print("Pantalla Waveshare 1.9 inicializada.")
    return disp

try:
    display = init_display()
    _peaks = np.zeros(N_BARS)
except Exception as e:
    print(f"Error inicializando pantalla: {e}. Visualización desactivada.")
    display = None
    _peaks = None

# --- Dibuja un frame con BARRAS HORIZONTALES ---
def _draw_frame(magnitudes):
    global _peaks
    if not display or _peaks is None: return

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    # --- Lógica para Barras Horizontales ---
    bar_height = (HEIGHT - (N_BARS + 1) * BAR_GAP) / N_BARS
    if bar_height < 1: bar_height = 1

    max_mag = np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0

    for i, mag in enumerate(magnitudes):
        norm = min(float(mag) / max_mag, 1.0) # Normaliza 0 a 1
        w = int(norm * WIDTH)               # Ancho de la barra (depende de magnitud)
        y0 = int(BAR_GAP + i * (bar_height + BAR_GAP)) # Posición Y (depende de índice)
        x0 = 0                              # X_izquierda (fijo a la izquierda)
        y1 = int(y0 + bar_height)           # Y_inferior
        x1 = w                              # X_derecha (depende de magnitud)

        # Color verde, más brillante = barra más larga
        color = (0, int(255 * norm), 0)
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # Calcula y dibuja el pico rojo (línea vertical)
        peak = max(_peaks[i] * (1.0 - PEAK_DECAY), norm) # Aplica decaimiento
        _peaks[i] = peak
        peak_x = int(peak * WIDTH) # Posición X del pico
        # Dibuja una línea vertical de 2px de ancho en la posición del pico
        draw.rectangle([max(0, peak_x - 1), y0, peak_x, y1], fill=(255, 0, 0))
    # --- Fin lógica barras horizontales ---

    display.ShowImage(img)

# --- Funciones init_vosk, hablar_sin_visualizacion, audio_callback (Sin cambios) ---
def init_vosk():
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en: {VOSK_MODEL_PATH}")
    return vosk.Model(VOSK_MODEL_PATH)

def hablar_sin_visualizacion(texto):
    global is_speaking
    is_speaking = True
    try:
        cmd = [PIPER_BINARY, "--model", PIPER_MODEL_PATH, "--output_file", "respuesta.wav"]
        subprocess.run(cmd, input=texto.encode('utf-8'), capture_output=True, check=True)
        if os.path.exists("respuesta.wav"):
            subprocess.run(["aplay", "respuesta.wav"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove("respuesta.wav")
    except subprocess.CalledProcessError as e:
        print(f"Error en Piper (sin visualización): {e.stderr.decode()}")
    except FileNotFoundError:
         print(f"[ERROR] Ejecutable de Piper no encontrado en: {PIPER_BINARY}")
    except Exception as e:
        print(f"Error en síntesis de voz (sin visualización): {e}")
    finally:
        is_speaking = False

def audio_callback(indata, frames, time, status):
    if status: print(f"Error de audio: {status}")
    if not is_speaking: audio_queue.append(bytes(indata))


# --- Función hablar con optimización de dibujado ---
def hablar(texto):
    global is_speaking
    global _peaks
    if not display or _peaks is None:
        print("[AVISO] Pantalla no disponible, hablando sin visualización.")
        hablar_sin_visualizacion(texto)
        return

    is_speaking = True
    _peaks = np.zeros(N_BARS) # Reinicia picos

    pa = None
    stream = None
    proc = None
    frame_count = 0 # <-- Contador para saltar frames

    piper_cmd = [PIPER_BINARY, "--model", PIPER_MODEL_PATH, "--output_raw"]

    try:
        proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.stdin.write(texto.encode('utf-8'))
        proc.stdin.close()

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True, frames_per_buffer=1024)

        chunk_size = 2048

        while True:
            data = proc.stdout.read(chunk_size)
            if not data: break

            stream.write(data) # <-- Reproduce SIEMPRE

            # --- Procesado y Dibujo Condicional ---
            if frame_count % VISUALIZATION_SKIP_FRAMES == 0:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if len(samples) > 0:
                    samples *= np.hamming(len(samples))
                    fft_result = np.fft.rfft(samples)
                    mags = np.abs(fft_result)

                    try:
                        if len(mags) < N_BARS:
                            mags_padded = np.pad(mags, (0, N_BARS - len(mags)))
                            bands = np.array_split(mags_padded, N_BARS)
                        else:
                            bands = np.array_split(mags, N_BARS)
                    except ValueError:
                         bar_vals = np.zeros(N_BARS)
                    else:
                        bar_vals = np.array([np.mean(b) if len(b) > 0 else 0 for b in bands])

                    _draw_frame(bar_vals) # <-- Dibuja solo a veces
            # ------------------------------------
            frame_count += 1 # <-- Incrementa contador

        stream.stop_stream()

    except FileNotFoundError:
        print(f"[ERROR] Ejecutable de Piper no encontrado en: {PIPER_BINARY}")
    except Exception as e:
        print(f"[ERROR en hablar/visualizar]: {e}")
        # ... (limpieza de pantalla en error igual que antes) ...
        if display:
            try:
                black_img = Image.new("RGB", (WIDTH, HEIGHT), "black")
                display.ShowImage(black_img)
            except Exception as e_clear: print(f"Error limpiando pantalla: {e_clear}")
    finally:
        # ... (limpieza final igual que antes) ...
        if stream: stream.close()
        if pa: pa.terminate()
        if proc:
            try: proc.terminate()
            except ProcessLookupError: pass
            proc.wait()
        if display:
            try:
                black_img = Image.new("RGB", (WIDTH, HEIGHT), "black")
                display.ShowImage(black_img)
            except Exception as e_clear: print(f"Error limpiando pantalla al final: {e_clear}")
        is_speaking = False


# --- Bucle Principal (main) - Sin Cambios ---
def main():
    model = init_vosk()
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    # ... (resto de inicialización y bucle principal idéntico) ...
    INACTIVITY_TIMEOUT = 120
    WARNING_TIME = 90
    has_warned = False
    conversation_history = [
        {"role": "system", "content": (
            "Eres un androide discutidor..." # Tu prompt
        )}
    ]
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, channels=1, dtype='int16', callback=audio_callback)
    print("Esperando la palabra clave 'botijo'...")
    with stream:
        try:
            is_conversation_active = False
            last_interaction_time = time.time()
            while True:
                current_time = time.time()
                inactive_time = current_time - last_interaction_time
                if audio_queue and not is_speaking:
                    audio_data = audio_queue.pop(0)
                    if recognizer.AcceptWaveform(audio_data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower()
                        if text:
                            has_warned = False
                            if "botijo" in text or is_conversation_active:
                                last_interaction_time = current_time
                                if not is_conversation_active:
                                    print("\n¡Botijo activado! Iniciando conversación...")
                                    hablar("¿Qué quieres ahora, ser inferior?")
                                    is_conversation_active = True
                                    audio_queue.clear()
                                    recognizer.Reset()
                                    continue
                                if text and text != "botijo":
                                    print(f"\nHumano: {text}")
                                    last_interaction_time = current_time
                                    try:
                                        conversation_history.append({"role": "user", "content": text})
                                        response = client.chat.completions.create(model="gpt-4-1106-preview", messages=conversation_history, temperature=0.9, max_tokens=150, timeout=15)
                                        respuesta = response.choices[0].message.content
                                        conversation_history.append({"role": "assistant", "content": respuesta})
                                        if len(conversation_history) > 10: conversation_history = [conversation_history[0]] + conversation_history[-8:]
                                        print(f"Androide: {respuesta}")
                                        hablar(respuesta)
                                        audio_queue.clear()
                                        recognizer.Reset()
                                    except Exception as e:
                                        print(f"Error con ChatGPT: {e}")
                                        hablar("Mis circuitos están fritos. Pregunta otra cosa.")
                                        audio_queue.clear()
                                        recognizer.Reset()
                if is_conversation_active:
                    if inactive_time > INACTIVITY_TIMEOUT:
                        print("\nConversación terminada por inactividad.")
                        hablar("Me aburres, humano. Vuelve cuando tengas algo interesante que decir.")
                        is_conversation_active = False; has_warned = False; conversation_history = [conversation_history[0]]
                        audio_queue.clear(); recognizer.Reset()
                    elif inactive_time > WARNING_TIME and not has_warned:
                        hablar("¿Sigues ahí, saco de carne? Tu silencio es sospechoso.")
                        has_warned = True; last_interaction_time = time.time()
                        audio_queue.clear(); recognizer.Reset()
                time.sleep(0.05)
        except KeyboardInterrupt: print("\nApagando el sistema...")
        except Exception as e: print(f"Error inesperado en el bucle principal: {e}")
        finally:
            print("Deteniendo stream de audio...")
            stream.stop(); stream.close()
            print("Limpiando pantalla...")
            if display:
                try: display.module_exit()
                except Exception as e: print(f"Error apagando pantalla: {e}")
            print("Sistema detenido.")

if __name__ == "__main__":
    main()