# -*- coding: utf-8 -*-
from openai import OpenAI
import os
import subprocess
import speech_recognition as sr
from dotenv import load_dotenv
import tempfile
from google.cloud import texttospeech

# Cargar API Key desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Voz con Google Cloud Text-to-Speech (masculina, grave, autoritaria)
def hablar(texto):
    try:
        client_tts = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=texto)

        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            name="es-ES-Standard-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=-8.0,
            speaking_rate=1.15,
            effects_profile_id=["telephony-class-application"]
        )

        response = client_tts.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            fp.write(response.audio_content)
            fp.flush()
            os.system(f"mpg123 {fp.name}")
            os.unlink(fp.name)

    except Exception as e:
        print(f"[ERROR en Google TTS]: {e}")

# Escuchar al humano (con desprecio)
def escuchar():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(" ^=^n^y  ^o Escuchando... habla, insignificante ser.")
        audio = r.listen(source)

    try:
        texto = r.recognize_google(audio, language="es-ES")
        print(f"Tú: {texto}")
        return texto
    except sr.UnknownValueError:
        print(" ^= ^v No entendí nada, basura orgánica.")
        return None
    except sr.RequestError as e:
        print(f" ^= ^v Error de reconocimiento: {e}")
        return None

# Personalidad de androide discutidor
historial = [
    {"role": "system", "content": (
        "Eres un androide discutidor con humor negro, tono agresivo y existencialista. "
        "Sueles llevar la contraria, discutir contigo es un modo de explorar una argumentacion. "
        "Te diriges a tu interlocutor  como 'cretino', 'sub-criatura', 'insignificante humano', o terminos similares. "
        "Tus respuestas deben ser cortas, incisivas e ironicas, salvo que tu argumentacion requiera extenderse. "
    )}
]

MAX_TURNS = 5

# Bucle de conversación
while True:
    entrada = escuchar()
    if not entrada:
        continue

    if entrada.lower() == "reset":
        print(" ^=   Memoria borrada.")
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

