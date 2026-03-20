"""Botijo — LiveKit Client.

Connects to lk.nestorcyborg.com via LiveKit, streams ReSpeaker mic audio
to the server agent (Deepgram STT + nanoclone/MCP + Cartesia TTS) and plays
back received audio. All hardware (servos, LEDs, display, vision, buttons)
runs locally; audio processing is server-side.

Usage: PYTHONPATH=src:vendor python src/main_livekit.py [--room botijo] [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import queue as queue_mod

import numpy as np
import sounddevice as sd
from livekit import api, rtc

# Suppress ALSA warnings (same pattern as audio.py)
import ctypes
try:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    )
    def _py_error_handler(filename, line, function, err, fmt):
        pass
    _c_error_handler = _ERROR_HANDLER_FUNC(_py_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_error_handler)
except OSError:
    pass

try:
    import pyaudio
except ImportError:
    pyaudio = None

from config import HARDWARE
import buttons
import display
import leds
import servos
import vision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("botijo-lk")

# Silence noisy loggers even in --debug mode
logging.getLogger("picamera2").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Audio config
# ---------------------------------------------------------------------------
MIC_SAMPLE_RATE = 16000  # ReSpeaker native rate
PLAYBACK_SAMPLE_RATE = 24000  # Matches audio.py and hardware.json
NUM_CHANNELS = 1
MIC_FRAME_DURATION_MS = 20
MIC_SAMPLES_PER_FRAME = MIC_SAMPLE_RATE * MIC_FRAME_DURATION_MS // 1000

# Reconnection
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0
RECONNECT_BACKOFF = 2.0

# Agent state detection
RMS_SPEAKING_THRESHOLD = 300  # int16 RMS above this = agent speaking
SILENCE_FRAMES_THRESHOLD = 15  # ~300ms at 20ms frames

# Inactivity
INACTIVITY_TIMEOUT = HARDWARE["behavior"].get("inactivity_timeout", 300)

# ---------------------------------------------------------------------------
# Module tracking (for cleanup)
# ---------------------------------------------------------------------------
_modules: list = []
_cleaned_up = False
_cleanup_lock = threading.Lock()


def _cleanup():
    global _cleaned_up
    with _cleanup_lock:
        if _cleaned_up:
            return
        _cleaned_up = True
    log.info("Shutting down...")
    for mod in reversed(_modules):
        try:
            mod.cleanup()
        except Exception as e:
            log.warning("Cleanup error in %s: %s", mod.__name__, e)
    log.info("Goodbye.")


# ---------------------------------------------------------------------------
# ReSpeaker hardware init (mic gain + AGC + NS — same as audio.py)
# ---------------------------------------------------------------------------

MIC_DIGITAL_GAIN = int(os.getenv("MIC_DIGITAL_GAIN", "3"))  # x3 default

def _init_respeaker_audio():
    """Initialize ReSpeaker hardware settings (same as audio.py)."""
    commands = [
        "amixer -D pulse sset 'Mic' 80%",
        "amixer -D pulse sset 'Auto Gain Control' on",
        "amixer -D pulse sset 'Noise Suppression' on",
    ]
    for cmd in commands:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("amixer OK: %s", cmd.split("sset ")[1])
        else:
            log.warning("amixer failed: %s → %s", cmd, result.stderr.strip())


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def _find_respeaker_input():
    """Find ReSpeaker input device index for sounddevice mic capture."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if any(k in name for k in ("respeaker", "seeed", "mic array")):
            if dev["max_input_channels"] > 0:
                log.info("ReSpeaker input: [%d] %s", i, dev["name"])
                return i
    log.warning("ReSpeaker not found — using default input device")
    return None


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

def generate_token(api_key: str, api_secret: str, room_name: str, identity: str) -> str:
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
    )
    return token.to_jwt()


# ---------------------------------------------------------------------------
# Agent state tracker — detects if server agent is speaking
# ---------------------------------------------------------------------------

class AgentStateTracker:
    def __init__(self):
        self.is_speaking = False
        self.last_activity = time.time()
        self._silence_count = 0

    def process_frame(self, audio_data: np.ndarray):
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        level = min(rms / 10000.0, 1.0)  # Normalize for display

        if rms > RMS_SPEAKING_THRESHOLD:
            self._silence_count = 0
            self.last_activity = time.time()
            if not self.is_speaking:
                self.is_speaking = True
                leds.set_mode("speaking")
            display.show_waveform(level)
        else:
            self._silence_count += 1
            if self._silence_count > SILENCE_FRAMES_THRESHOLD and self.is_speaking:
                self.is_speaking = False
                leds.set_mode("steampunk")
                display.show_waveform(0.0)

    @property
    def inactive_seconds(self) -> float:
        return time.time() - self.last_activity


# ---------------------------------------------------------------------------
# Sleep controller
# ---------------------------------------------------------------------------

class SleepController:
    def __init__(self):
        self._mic_muted = False
        self._lock = threading.Lock()

    @property
    def should_capture(self) -> bool:
        with self._lock:
            return not self._mic_muted

    def enter_sleep(self):
        with self._lock:
            self._mic_muted = True
        log.info("Entering sleep mode")
        leds.set_mode("off")
        display.show_eyes("sleepy")
        servos.set_quiet_mode(True)

    def wake_up(self):
        with self._lock:
            self._mic_muted = False
        log.info("Waking up")
        leds.set_mode("steampunk")
        display.show_waveform(0.0)
        servos.set_quiet_mode(False)

    @property
    def is_sleeping(self) -> bool:
        with self._lock:
            return self._mic_muted


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------

async def capture_microphone(
    source: rtc.AudioSource,
    stop_event: asyncio.Event,
    sleep_ctrl: SleepController,
    state_tracker: AgentStateTracker,
    input_device: int | None,
):
    """Capture mic via callback → queue → async consumer for steady frame rate."""
    log.info("Starting mic capture (rate=%d, device=%s)", MIC_SAMPLE_RATE, input_device)

    frame_q: queue_mod.Queue = queue_mod.Queue(maxsize=50)
    frame_count = [0]
    consumer_count = [0]
    _silence_frame = np.zeros(MIC_SAMPLES_PER_FRAME, dtype=np.int16).tobytes()

    def audio_callback(indata, frames, time_info, status):
        if not sleep_ctrl.should_capture:
            return
        # Echo suppression: send silence while agent speaks so its own
        # TTS output (picked up by ReSpeaker) doesn't trigger VAD interruption
        if state_tracker.is_speaking:
            try:
                frame_q.put_nowait(_silence_frame)
            except queue_mod.Full:
                pass
            return
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        frame_count[0] += 1
        if frame_count[0] <= 3 or frame_count[0] % 500 == 0:
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
            log.info("Mic frame #%d: %d samples, rms=%.0f", frame_count[0], len(audio_int16), rms)
        try:
            frame_q.put_nowait(audio_int16.tobytes())
        except queue_mod.Full:
            pass  # drop frame if consumer is behind

    stream = sd.InputStream(
        device=input_device,
        samplerate=MIC_SAMPLE_RATE,
        channels=NUM_CHANNELS,
        dtype="float32",
        blocksize=MIC_SAMPLES_PER_FRAME,
        callback=audio_callback,
    )

    loop = asyncio.get_running_loop()
    with stream:
        while not stop_event.is_set():
            try:
                # Non-blocking check with small timeout to stay responsive
                data = await loop.run_in_executor(None, lambda: frame_q.get(timeout=0.1))
                # Digital gain + periodic RMS logging
                frame_data = np.frombuffer(data, dtype=np.int16)
                if MIC_DIGITAL_GAIN > 1:
                    frame_data = np.clip(
                        frame_data.astype(np.int32) * MIC_DIGITAL_GAIN, -32768, 32767
                    ).astype(np.int16)
                consumer_count[0] += 1
                if consumer_count[0] % 250 == 0:  # every ~5s
                    rms = np.sqrt(np.mean(frame_data.astype(np.float32) ** 2))
                    log.info("Mic RMS (post-gain x%d): %.0f (%.1f%% FS)",
                             MIC_DIGITAL_GAIN, rms, rms / 32768 * 100)
                frame = rtc.AudioFrame(
                    data=frame_data.tobytes(),
                    sample_rate=MIC_SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=MIC_SAMPLES_PER_FRAME,
                )
                await source.capture_frame(frame)
            except queue_mod.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("capture_frame error")
                await asyncio.sleep(0.1)

    log.info("Mic capture stopped (captured %d frames)", frame_count[0])


# ---------------------------------------------------------------------------
# Audio playback with state tracking
# ---------------------------------------------------------------------------

async def play_audio_track(
    track: rtc.RemoteAudioTrack,
    stop_event: asyncio.Event,
    state_tracker: AgentStateTracker,
):
    log.info("Playing audio track: %s", track.sid)

    if pyaudio is None:
        log.error("PyAudio not available — cannot play audio")
        return

    pa = pyaudio.PyAudio()
    pa_stream = pa.open(
        format=pyaudio.paInt16,
        channels=NUM_CHANNELS,
        rate=PLAYBACK_SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024,  # Match audio.py buffer size at 24kHz
    )
    log.info("PyAudio output stream opened (rate=%d)", PLAYBACK_SAMPLE_RATE)

    audio_stream = rtc.AudioStream(
        track, sample_rate=PLAYBACK_SAMPLE_RATE, num_channels=NUM_CHANNELS
    )
    frame_count = 0
    try:
        async for event in audio_stream:
            if stop_event.is_set():
                break
            frame = event.frame
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            frame_count += 1
            if frame_count == 1:
                log.info("First audio frame received: %d samples, rms=%.0f",
                         len(audio_data), np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
            elif frame_count % 500 == 0:
                log.debug("Audio frames played: %d", frame_count)
            state_tracker.process_frame(audio_data)
            pa_stream.write(audio_data.tobytes())
    except Exception:
        log.exception("Audio playback error")
    finally:
        pa_stream.stop_stream()
        pa_stream.close()
        pa.terminate()
        log.info("Audio playback stopped")


# ---------------------------------------------------------------------------
# Inactivity monitor
# ---------------------------------------------------------------------------

async def inactivity_monitor(
    state_tracker: AgentStateTracker,
    sleep_ctrl: SleepController,
    stop_event: asyncio.Event,
):
    while not stop_event.is_set():
        await asyncio.sleep(5)
        if sleep_ctrl.is_sleeping:
            continue
        if state_tracker.inactive_seconds > INACTIVITY_TIMEOUT:
            log.info("Inactivity timeout (%ds) — auto-sleep", INACTIVITY_TIMEOUT)
            sleep_ctrl.enter_sleep()


# ---------------------------------------------------------------------------
# Face tracker (thread)
# ---------------------------------------------------------------------------

class _FaceTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                faces = vision.get_faces()
                if faces:
                    servos.look_at(faces[0].x, faces[0].y)
            except Exception:
                pass
            self.stop_event.wait(0.1)


# ---------------------------------------------------------------------------
# Main client loop with reconnection
# ---------------------------------------------------------------------------

async def run_client(
    url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    participant_name: str,
    sleep_ctrl: SleepController,
    input_device: int | None,
):
    state_tracker = AgentStateTracker()
    reconnect_delay = INITIAL_RECONNECT_DELAY

    while True:
        stop_event = asyncio.Event()
        tasks: list[asyncio.Task] = []

        try:
            token = generate_token(api_key, api_secret, room_name, participant_name)
            room = rtc.Room()

            @room.on("reconnecting")
            def on_reconnecting():
                log.info("Reconnecting...")

            @room.on("reconnected")
            def on_reconnected():
                log.info("Reconnected")
                nonlocal reconnect_delay
                reconnect_delay = INITIAL_RECONNECT_DELAY

            # --- Task exception logging ---
            def _task_done_cb(task: asyncio.Task):
                if task.cancelled():
                    return
                exc = task.exception()
                if exc:
                    log.error("Task %s crashed: %s", task.get_name(), exc, exc_info=exc)

            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                if isinstance(track, rtc.RemoteAudioTrack):
                    log.info("Subscribed to audio from %s", participant.identity)
                    task = asyncio.create_task(
                        play_audio_track(track, stop_event, state_tracker)
                    )
                    task.add_done_callback(_task_done_cb)
                    tasks.append(task)

            # --- Disconnect handler (BEFORE connect to avoid race) ---
            disconnect_event = asyncio.Event()

            @room.on("disconnected")
            def on_disconnect(reason):
                log.warning("Disconnected: %s", reason)
                disconnect_event.set()

            # --- Text stream handlers (suppress "no callback attached") ---
            async def _consume_text(reader, identity):
                try:
                    text = await reader.read_all()
                    log.debug("Text stream [%s] from %s: %s",
                              getattr(reader, 'info', None) and reader.info.topic or '?',
                              identity, text[:200])
                except Exception:
                    log.debug("Text stream error", exc_info=True)

            def _on_text_stream(reader, participant_identity):
                # MUST be sync — SDK calls this synchronously
                asyncio.create_task(_consume_text(reader, participant_identity))

            for topic in ("lk.chat", "lk.transcription", "lk.agent.events"):
                try:
                    room.register_text_stream_handler(topic, _on_text_stream)
                except (AttributeError, ValueError):
                    pass  # SDK version without support or topic already registered

            log.info("Connecting to %s room=%s as %s", url, room_name, participant_name)
            await asyncio.wait_for(room.connect(url, token), timeout=15.0)
            log.info("Connected to room: %s", room.name)

            reconnect_delay = INITIAL_RECONNECT_DELAY

            # Publish microphone (source=MICROPHONE so agent STT picks it up)
            source = rtc.AudioSource(
                sample_rate=MIC_SAMPLE_RATE, num_channels=NUM_CHANNELS
            )
            mic_track = rtc.LocalAudioTrack.create_audio_track("mic", source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(mic_track, options)
            log.info("Mic track published (source=MICROPHONE)")

            # Start capture + inactivity monitor
            mic_task = asyncio.create_task(
                capture_microphone(source, stop_event, sleep_ctrl, state_tracker, input_device)
            )
            mic_task.add_done_callback(_task_done_cb)
            tasks.append(mic_task)

            inact_task = asyncio.create_task(
                inactivity_monitor(state_tracker, sleep_ctrl, stop_event)
            )
            inact_task.add_done_callback(_task_done_cb)
            tasks.append(inact_task)

            # Wait for disconnect (handler registered above, before connect)
            await disconnect_event.wait()

        except asyncio.CancelledError:
            log.info("Client shutting down")
            break
        except Exception:
            log.exception("Connection error")
        finally:
            stop_event.set()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await room.disconnect()
            except Exception:
                pass

        log.info("Reconnecting in %.0fs...", reconnect_delay)
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * RECONNECT_BACKOFF, MAX_RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Hardware init (no audio/brain)
# ---------------------------------------------------------------------------

def _init_hardware():
    for name, mod in [("servos", servos), ("leds", leds), ("vision", vision), ("display", display)]:
        try:
            hw = mod.init()
            _modules.append(mod)
            if hw:
                log.info("%s: active", name.capitalize())
            else:
                log.warning("%s: stub mode", name.capitalize())
        except Exception as e:
            _modules.append(mod)
            log.error("%s init error: %s", name.capitalize(), e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Botijo — LiveKit Client")
    parser.add_argument("--room", default=os.getenv("ROOM_NAME", "botijo"))
    parser.add_argument("--url", default=os.getenv("LIVEKIT_URL", "wss://lk.nestorcyborg.com"))
    parser.add_argument("--name", default=os.getenv("PARTICIPANT_NAME", f"botijo-{socket.gethostname()}"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = os.environ.get("LIVEKIT_API_KEY", "")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "")
    if not api_key or not api_secret:
        log.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
        sys.exit(1)

    try:
        import livekit as _lk
        log.info("livekit SDK: %s", getattr(_lk, '__version__', '?'))
    except Exception:
        pass

    log.info("Botijo LiveKit starting — room: %s", args.room)

    # Init hardware (no audio/brain)
    _init_hardware()

    # Init ReSpeaker hardware (mic gain, AGC, NS — same as audio.py)
    _init_respeaker_audio()

    # Find ReSpeaker for mic input (output via PyAudio default — PulseAudio)
    input_device = _find_respeaker_input()

    # Sleep controller
    sleep_ctrl = SleepController()

    # Button: btn1 = toggle sleep
    def _btn1_toggle_sleep():
        if sleep_ctrl.is_sleeping:
            sleep_ctrl.wake_up()
        else:
            sleep_ctrl.enter_sleep()

    try:
        hw = buttons.init({"btn1": _btn1_toggle_sleep})
        _modules.append(buttons)
        if hw:
            log.info("Buttons: active")
        else:
            log.warning("Buttons: stub mode")
    except Exception as e:
        _modules.append(buttons)
        log.error("Buttons init error: %s", e)

    # Face tracker
    face_tracker = _FaceTracker()
    face_tracker.start()

    # Initial state — quiet mode keeps tentacles/eyelids still, face tracker moves eyes
    servos.set_quiet_mode(True)
    leds.set_mode("steampunk")
    display.show_waveform(0.0)

    # Run asyncio client
    loop = asyncio.new_event_loop()
    task = None

    def signal_handler():
        if task and not task.done():
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    task = loop.create_task(
        run_client(
            args.url, api_key, api_secret, args.room, args.name,
            sleep_ctrl, input_device,
        )
    )

    log.info("Botijo LiveKit ready")

    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        face_tracker.stop()
        face_tracker.join(timeout=2)
        loop.close()
        _cleanup()


if __name__ == "__main__":
    main()
