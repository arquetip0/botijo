#!/usr/bin/env python3
"""
Test rápido SOLO de la función listen_for_command_energy para ReSpeaker
"""
import os
import sys
import numpy as np
import pyaudio
import queue
import time

# Configuración mínima
STT_RATE = 16000
STT_CHUNK = int(STT_RATE * 0.02)

# Clase MicrophoneStream simplificada
class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def test_voice_detection_only():
    """Test SOLO de detección de voz - SIN Google STT"""
    print("🎤 Test de detección de voz ReSpeaker (SIN transcripción)")
    print("Habla para ver si detecta tu voz correctamente...")
    print("Presiona Ctrl+C para salir\n")
    
    # Configuración específica para ReSpeaker (copiada de streambotijo2vad.py)
    BASE_THRESHOLD = 200
    ADAPTIVE_MULTIPLIER = 3.0
    SILENCE_LIMIT = 0.8
    MAX_RECORD = 12
    MIN_SPEECH_DURATION = 0.4
    ENERGY_WINDOW = 3
    VOICE_CONFIRMATION_TIME = 0.15
    
    is_speaking = False  # Simulado
    
    try:
        while True:
            audio_buffer = []
            speech_started = False
            silence_counter = 0.0
            start_time = 0
            speech_start_time = None
            
            # Variables para detección mejorada
            energy_history = []
            voice_confirmation_counter = 0.0
            current_threshold = BASE_THRESHOLD

            with MicrophoneStream(STT_RATE, STT_CHUNK) as stream:
                print("[INFO] 🎤 Calibrando ruido de fondo (ReSpeaker)...")
                
                # Calibración (1 segundo)
                calibration_start = time.time()
                calibration_samples = []
                sample_count = 0
                
                for chunk in stream.generator():
                    sample_count += 1
                    if time.time() - calibration_start > 1.0:  # 1 segundo
                        break
                    data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(data**2)) if len(data) > 0 else 0
                    calibration_samples.append(rms)
                
                if calibration_samples:
                    background_noise_level = np.mean(calibration_samples)
                    current_threshold = max(BASE_THRESHOLD, 
                                          background_noise_level * ADAPTIVE_MULTIPLIER)
                    print(f"[INFO] 📊 ReSpeaker - Ruido: {background_noise_level:.1f}, Umbral: {current_threshold:.1f}")
                else:
                    current_threshold = BASE_THRESHOLD
                
                print("[INFO] 🎤 ReSpeaker listo - habla ahora...")
                
                chunk_count = 0
                detection_start_time = time.time()
                
                for chunk in stream.generator():
                    chunk_count += 1
                    current_time = time.time()
                    
                    # Calcular RMS
                    data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(data**2)) if len(data) > 0 else 0
                    
                    # Mantener historial de energía
                    energy_history.append(rms)
                    if len(energy_history) > ENERGY_WINDOW:
                        energy_history.pop(0)
                    
                    avg_energy = np.mean(energy_history) if energy_history else rms
                    energy_above_threshold = avg_energy > current_threshold
                    
                    # Debug frecuente
                    if chunk_count % 25 == 0:  # Cada 0.5s
                        status = "🗣️ VOZ" if energy_above_threshold else "🤫 silencio"
                        print(f"[DEBUG] RMS: {rms:.0f}, Promedio: {avg_energy:.0f}, Umbral: {current_threshold:.0f} - {status}")
                    
                    if energy_above_threshold:
                        voice_confirmation_counter += STT_CHUNK / STT_RATE
                        
                        if voice_confirmation_counter >= VOICE_CONFIRMATION_TIME:
                            if not speech_started:
                                print(f"🗣️ ¡VOZ DETECTADA! (energía: {avg_energy:.0f} > {current_threshold:.0f})")
                                speech_started = True
                                speech_start_time = current_time
                            silence_counter = 0.0
                    else:
                        voice_confirmation_counter = 0.0
                        
                        if speech_started:
                            silence_counter += STT_CHUNK / STT_RATE
                            
                            if silence_counter > SILENCE_LIMIT:
                                speech_duration = current_time - speech_start_time
                                if speech_duration >= MIN_SPEECH_DURATION:
                                    print(f"✅ CAPTURA COMPLETADA (duración: {speech_duration:.1f}s)")
                                    print("🔄 Esperando siguiente comando...\n")
                                    break
                                else:
                                    print(f"⚡ Audio muy corto ({speech_duration:.1f}s), continuando...")
                                    speech_started = False
                                    silence_counter = 0.0
                                    voice_confirmation_counter = 0.0
                    
                    # Timeout
                    if current_time - detection_start_time > MAX_RECORD:
                        print("⏱️ Timeout - reiniciando...")
                        break
                        
    except KeyboardInterrupt:
        print("\n✅ Test completado")

if __name__ == "__main__":
    test_voice_detection_only()
