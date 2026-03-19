#!/usr/bin/env python3
"""Debug script para listar todos los dispositivos PyAudio"""

import pyaudio

def list_audio_devices():
    pa = pyaudio.PyAudio()
    
    print("🎤 Dispositivos de audio disponibles en PyAudio:")
    print("=" * 60)
    
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        print(f"\nDispositivo {i}:")
        print(f"  Nombre: '{device_info['name']}'")
        print(f"  Canales entrada: {device_info['maxInputChannels']}")
        print(f"  Canales salida: {device_info['maxOutputChannels']}")
        print(f"  Tasa muestreo: {device_info['defaultSampleRate']}")
        print(f"  Host API: {device_info['hostApi']}")
        
        # Marcar si podría ser ReSpeaker
        name_lower = device_info['name'].lower()
        if any(term in name_lower for term in ['respeaker', 'seeed', 'mic array', 'arrayuac', 'uac1.0', 'uac']):
            print(f"  ⭐ POSIBLE RESPEAKER ⭐")
    
    pa.terminate()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    list_audio_devices()
