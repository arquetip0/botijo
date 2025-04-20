from openai import OpenAI
import os
import subprocess
from dotenv import load_dotenv

# Cargar API Key desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Voz en español con espeak-ng
def hablar(texto):
    try:
        comando = ["espeak-ng", "-v", "es", "-s", "150", texto]
        subprocess.run(comando)
    except Exception as e:
        print(f"[ERROR en TTS]: {e}")

# Personalidad del androide discutidor
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

MAX_TURNS = 5  # ventana de memoria

# Bucle de conversación
while True:
    entrada = input("Tú: ")

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
        print(f"[ERROR en consulta a GPT]: {e}")
ø
