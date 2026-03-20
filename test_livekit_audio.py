#!/usr/bin/env python3
"""Minimal LiveKit audio test — no hardware, just mic + speaker.

Connects to lk.nestorcyborg.com, sends mic audio, plays back agent audio.
Run on RPi: cd ~/botijo && source venv_chatgpt/bin/activate && python test_livekit_audio.py
"""
import asyncio
import os
import sys
import time

import numpy as np
import sounddevice as sd

# Suppress ALSA spam
import ctypes
try:
    _EHF = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_EHF(lambda *a: None))
except OSError:
    pass

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from livekit import api, rtc

try:
    import pyaudio
except ImportError:
    pyaudio = None

SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


def generate_token(key, secret, room, identity):
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


async def main():
    key = os.environ.get("LIVEKIT_API_KEY", "")
    secret = os.environ.get("LIVEKIT_API_SECRET", "")
    url = os.environ.get("LIVEKIT_URL", "wss://lk.nestorcyborg.com")
    room_name = "botijo"

    if not key or not secret:
        print("ERROR: LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in ~/.env")
        sys.exit(1)

    token = generate_token(key, secret, room_name, "botijo-test")
    room = rtc.Room()
    stop = asyncio.Event()

    # --- Playback ---
    async def play_track(track):
        print(f"[PLAY] Subscribed to audio track: {track.sid}")
        if pyaudio is None:
            print("[PLAY] ERROR: PyAudio not available")
            return

        pa = pyaudio.PyAudio()
        pa_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=PLAYBACK_RATE,
                            output=True, frames_per_buffer=1024)
        print(f"[PLAY] PyAudio output opened at {PLAYBACK_RATE} Hz")

        audio_stream = rtc.AudioStream(track, sample_rate=PLAYBACK_RATE, num_channels=1)
        count = 0
        nonzero_count = 0
        try:
            async for event in audio_stream:
                if stop.is_set():
                    break
                frame = event.frame
                data = np.frombuffer(frame.data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                count += 1
                if rms > 50:
                    nonzero_count += 1
                if count == 1:
                    print(f"[PLAY] First frame: {len(data)} samples, rms={rms:.0f}")
                elif count % 200 == 0:
                    print(f"[PLAY] {count} frames, {nonzero_count} non-silent, last_rms={rms:.0f}")
                pa_stream.write(data.tobytes())
        except Exception as e:
            print(f"[PLAY] Error: {e}")
        finally:
            pa_stream.stop_stream()
            pa_stream.close()
            pa.terminate()
            print(f"[PLAY] Done — {count} total frames, {nonzero_count} non-silent")

    @room.on("track_subscribed")
    def on_track(track, pub, participant):
        if isinstance(track, rtc.RemoteAudioTrack):
            asyncio.create_task(play_track(track))

    # --- Connect ---
    print(f"[CONN] Connecting to {url} room={room_name}...")
    await room.connect(url, token)
    print(f"[CONN] Connected! Participants: {[p.identity for p in room.remote_participants.values()]}")

    # --- Mic capture ---
    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=1)
    mic_track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(mic_track)
    print(f"[MIC] Published mic track at {SAMPLE_RATE} Hz")

    loop = asyncio.get_running_loop()
    mic_frames = [0]

    def mic_callback(indata, frames, time_info, status):
        if status:
            pass  # ignore overflow
        audio = (indata[:, 0] * 32767).astype(np.int16)
        frame = rtc.AudioFrame(
            data=audio.tobytes(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(audio),
        )
        asyncio.run_coroutine_threadsafe(source.capture_frame(frame), loop)
        mic_frames[0] += 1

    # Find ReSpeaker
    input_dev = None
    for i, dev in enumerate(sd.query_devices()):
        if "respeaker" in dev["name"].lower() and dev["max_input_channels"] > 0:
            input_dev = i
            print(f"[MIC] Using ReSpeaker input: [{i}] {dev['name']}")
            break
    if input_dev is None:
        print("[MIC] Using default input")

    stream = sd.InputStream(device=input_dev, samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=SAMPLES_PER_FRAME, callback=mic_callback)

    with stream:
        print("[MIC] Mic capture running — SPEAK NOW to trigger agent response")
        print("[INFO] Press Ctrl+C to stop (will run for 60s max)")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    stop.set()
    await room.disconnect()
    print(f"[MIC] Sent {mic_frames[0]} mic frames")
    print("[DONE]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted")
