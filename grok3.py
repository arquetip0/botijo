import pvporcupine
import pyaudio
import speech_recognition as sr
import openai
import subprocess
import os
from dotenv import load_dotenv
import time
import struct  # ¡Añadido para resolver el error!

# === CONFIGURACIÓN ===
PIPER_BINARY = "/home/jack/piper/build/piper"  # Ruta al binario de Piper
PIPER_MODEL_PATH = "/home/jack/piper/es_MX-laura-high.onnx"  # Modelo de voz
KEYWORD_PATH = "/home/jack/picovoice/Botijo_es_raspberry-pi_v3_0_0.ppn"  # Wake word

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY no encontrada en .env")

# === VALIDAR ARCHIVOS NECESARIOS ===
for path, descripcion in [
    (PIPER_BINARY, "binario Piper"),
    (PIPER_MODEL_PATH, "modelo Piper"),
    (KEYWORD_PATH, "wake word .ppn")
]:
    if not os.path.isfile(path):
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

# === INICIALIZAR PICOVOICE ===
porcupine = pvporcupine.create(
    access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
    model_path="/home/jack/picovoice/porcupine_params_es.pv",
    keyword_paths=[KEYWORD_PATH]
)
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length,
)

recognizer = sr.Recognizer()
mic = sr.Microphone()

print("Esperando la palabra clave…")

try:
    while True:
        # === ESCUCHA PASIVA ===
        pcm = audio_stream.read(
            porcupine.frame_length, exception_on_overflow=False
        )
        try:
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        except struct.error:
            print("\u001b[33m[ADVERTENCIA]\u001b[0m Frame de audio corrupto o incompleto, ignorando...")
            continue

        if porcupine.process(pcm) >= 0:
            start_time = time.time()
            print(
                "\u001b[32m[ACTIVADO]\u001b[0m Palabra clave 'botijo' detectada. "
                "Esperando instrucción…"
            )

            # === GRABAR FRASE ===
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, phrase_time_limit=10)

            try:
                frase = recognizer.recognize_google(audio, language="es-ES")
                print(f"Usuario: {frase}")

                # === GPT ===
                messages = [system_prompt, {"role": "user", "content": frase}]
                respuesta = openai.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=messages,
                )
                texto_respuesta = respuesta.choices[0].message.content.strip()
                print(f"Androide: {texto_respuesta}")

                # === PIPER ===
                env = os.environ.copy()
                env["ESPEAK_DATA_PATH"] = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"

                piper_cmd = [
                    PIPER_BINARY,
                    "--model", PIPER_MODEL_PATH,
                    "--output_file", "respuesta.wav"
                ]
                process = subprocess.run(
                    piper_cmd,
                    input=texto_respuesta,
                    text=True,
                    capture_output=True,
                    env=env
                )
                if process.returncode != 0:
                    raise RuntimeError(f"Piper falló: {process.stderr}")

                subprocess.run(["aplay", "respuesta.wav"], check=True)

                # === LIMPIEZA ===
                os.remove("respuesta.wav")

                # === LOG Y ESTADÍSTICAS ===
                duracion = time.time() - start_time
                print(f"m[INACTIVO] Volviendo a modo escucha…")
                print(f"m[DURACIÓN] Interacción completa en {duracion:.2f} s")

            except Exception as e:
                print(
                    f"\u001b[31m[ERROR]\u001b[0m Reconocimiento o generación falló: {e}"
                )

except KeyboardInterrupt:
    print("Apagando…")
finally:
    if audio_stream is not None:
        audio_stream.stop_stream()
        audio_stream.close()
    if porcupine is not None:
        porcupine.delete()
    pa.terminate()
