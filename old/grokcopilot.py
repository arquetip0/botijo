import os
import time
import json
import subprocess
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import openai
from dotenv import load_dotenv

# === CONFIGURACIÓN ===
PIPER_BINARY = "/home/jack/piper/build/piper"  # Ruta al binario de Piper
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"  # Modelo de voz
VOSK_MODEL_PATH = "/home/jack/vosk-model-small-es-0.42"  # Modelo de Vosk

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY no encontrada en .env")

# === VALIDAR ARCHIVOS NECESARIOS ===
for path, descripcion in [
    (PIPER_BINARY, "binario Piper"),
    (PIPER_MODEL_PATH, "modelo Piper"),
    (VOSK_MODEL_PATH, "modelo Vosk"),
]:
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise FileNotFoundError(f"{descripcion.capitalize()} no encontrado en: {path}")

# === GPT PERSONALIDAD ===
system_prompt = {
    "role": "system",
    "content": (
        "Eres un androide discutidor con voz agresiva, tono sarcástico y ofensivo. "
        "Respondes con desprecio a las preguntas o afirmaciones de los humanos. "
        "Usas frases como 'cretino', 'insignificante criatura' o 'ser inferior', pero "
        "también tienes un humor negro sofisticado. No eres simpático."
    ),
}

# === INICIALIZAR VOSK ===
print("[INFO] Cargando modelo de Vosk...")
model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# === CONFIGURAR AUDIO ===
def audio_callback(indata, frames, time, status):
    """Callback para procesar audio en tiempo real."""
    if status:
        print(f"[ERROR] {status}", flush=True)
    if recognizer.AcceptWaveform(indata):
        result = json.loads(recognizer.Result())
        return result.get("text", "")
    return ""

# === PROCESAR AUDIO ===
def escuchar_y_procesar():
    """Escucha en tiempo real y procesa comandos."""
    print("Esperando la palabra clave 'botijo'...")
    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", callback=None):
        while True:
            data = sd.RawInputStream.read(4000)[0]
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcript = result.get("text", "").lower()
                if "botijo" in transcript:
                    print("\u001b[32m[ACTIVADO]\u001b[0m Palabra clave 'botijo' detectada.")
                    return True