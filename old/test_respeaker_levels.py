#!/usr/bin/env python3
"""
Test rápido para verificar niveles de audio del ReSpeaker 2-Mic USB Array
"""
import pyaudio
import numpy as np
import time

STT_RATE = 16000
STT_CHUNK = int(STT_RATE * 0.02)  # 20ms chunks

def test_respeaker_levels():
    """Test para ver los niveles de audio del ReSpeaker"""
    print("🎤 Iniciando test del ReSpeaker 2-Mic USB Array...")
    print("Habla ahora para ver los niveles de audio")
    print("Presiona Ctrl+C para salir\n")
    
    # Configurar PyAudio
    pa = pyaudio.PyAudio()
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=STT_RATE,
            input=True,
            frames_per_buffer=STT_CHUNK,
        )
        
        start_time = time.time()
        max_rms = 0
        avg_rms = 0
        sample_count = 0
        
        while True:
            data = stream.read(STT_CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
            
            max_rms = max(max_rms, rms)
            avg_rms = ((avg_rms * sample_count) + rms) / (sample_count + 1)
            sample_count += 1
            
            # Mostrar niveles cada 0.5 segundos
            if sample_count % 25 == 0:  # ~0.5 segundos
                bar_length = 50
                bar_fill = int((rms / 2000) * bar_length) if rms < 2000 else bar_length
                bar = "█" * bar_fill + "░" * (bar_length - bar_fill)
                
                print(f"\r[{bar}] RMS: {rms:6.0f} | Max: {max_rms:6.0f} | Avg: {avg_rms:6.0f}", end="", flush=True)
            
            # Reset cada 10 segundos
            if time.time() - start_time > 10:
                print(f"\n📊 Resumen (10s): Max RMS: {max_rms:.0f}, Promedio: {avg_rms:.0f}")
                print("🔧 Umbrales sugeridos:")
                print(f"   - Silencio: {avg_rms * 1.5:.0f}")
                print(f"   - Voz baja: {avg_rms * 3:.0f}")
                print(f"   - Voz normal: {avg_rms * 5:.0f}")
                print("")
                max_rms = 0
                avg_rms = 0
                sample_count = 0
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("\n\n✅ Test completado")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    test_respeaker_levels()
