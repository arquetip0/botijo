from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
import io

api_key_value = "sk_9144204b0612eed1d5ad1615acd1b840c4257091c343b51d"
eleven = ElevenLabs(api_key=api_key_value)

def convertir_a_pcm_22050(audio_bytes_mp3: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes_mp3), format="mp3")
    audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
    return audio.raw_data

def main():
    try:
        audio_gen = eleven.text_to_speech.convert(
            text="Hola, esto es una prueba de audio.",
            voice_id="RnKqZYEeVQciORlpiCz0",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        # Recolectar todos los bytes
        audio_bytes = b"".join(chunk for chunk in audio_gen if isinstance(chunk, bytes))
        print(f"DEBUG: Audio MP3 recogido, {len(audio_bytes)} bytes")

        # Convertir para el visualizador
        raw_data = convertir_a_pcm_22050(audio_bytes)
        print(f"DEBUG: raw_data listo para visualizador, {len(raw_data)} bytes")

        # Reproducir
        print("DEBUG: Reproduciendo audio...")
        play(audio_bytes)
        print("DEBUG: Reproducción finalizada.")

    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    main()
