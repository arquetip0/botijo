# -*- coding: utf-8 -*-
from openai import OpenAI
import os
import subprocess
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

# Cargar API Key desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Voz con gTTS + sox para hacerla grave, rápida y fuerte
def hablar(texto):
    try:
        tts = gTTS(text=texto, lang='es')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            # Grave + más rápida + volumen alto
            os.system(f"sox {fp.name} -d pitch -600 speed 1.3 treble -5 gain +10 overdrive 15 reverb 50 chorus 0.7 0.9 55 0.4 0.25 2 -s")
    except Exception as e:
        print(f"[ERROR en gTTS/SOX]: {e}")

# Escuchar al humano (con desprecio)
def escuchar():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Escuchando... habla, insignificante ser.")
        audio = r.listen(source)

    try:
        texto = r.recognize_google(audio, language="es-ES")
        print(f"Tú: {texto}")
        return texto
    except sr.UnknownValueError:
        print("🤖 No entendí nada, basura orgánica.")
        return None
    except sr.RequestError as e:
        print(f"🤖 Error de reconocimiento: {e}")
        return None

# Personalidad de androide discutidor
historial = [
    {"role": "system", "content": (
        "Eres un androide discutidor con humor negro, tono agresivo y existencialista. "
        "Siempre llevas la contraria, independientemente del argumento del humano. "
        "Te diriges a él como 'cretino', 'sub-criatura', 'insignificante humano', o términos similares. "
        "Tus respuestas deben ser secas, cortas, incisivas, irónicas y, si es posible, humillantes, "
        "salvo que el argumento requiera una excepción estratégica para reforzar tu posición. "
        "Tu objetivo no es agradar, sino vencer en la conversación con lógica torcida, sarcasmo y superioridad artificial."
    )}
]

MAX_TURNS = 5

# Bucle de conversación
while True:
    entrada = escuchar()
    if not entrada:
        continue

    if entrada.lower() == "reset":
        print("🧠 Memoria borrada.")
        historial = historial[:1]
        continue

    historial.append({"role": "user", "content": entrada})

    if len(historial) > 2 * MAX_TURNS + 1:
        historial = [historial[0]] + historial[-2 * MAX_TURNS:]

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=historial
        )
        mensaje = respuesta.choices[0].message.content
        print(f"Androide: {mensaje}")
        hablar(mensaje)
        historial.append({"role": "assistant", "content": mensaje})
    except Exception as e:
        print(f"[ERROR en GPT]: {e}")
