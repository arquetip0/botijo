"""Audio module — ReSpeaker v2.0, VAD, STT (Google Cloud), TTS (ElevenLabs + espeak-ng).

Public API:
    init()              -> bool          Detect ReSpeaker, open PyAudio, init VAD
    listen()            -> str           VAD + Google Cloud STT -> text
    speak(text)         -> SpeakResult   TTS a single text string
    speak_stream(chunks)-> SpeakResult   TTS from streaming generator
    is_voice_detected() -> bool          For external interruption checks
    stop_speaking()     -> None          Force stop TTS
    cleanup()           -> None          Release all resources
"""

import collections
import io
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

# Hardware-dependent imports — may not be available on dev machines
try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import usb.core
    import usb.util
except ImportError:
    usb = None

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = None
    filtfilt = None

try:
    from google.cloud import speech
except ImportError:
    speech = None

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None

from config import HARDWARE, ELEVENLABS_API_KEY, GOOGLE_CREDENTIALS

log = logging.getLogger("audio")

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class SpeakResult:
    interrupted: bool
    spoken_text: str


# ---------------------------------------------------------------------------
# Module-level state (replaces globals from the monolith)
# ---------------------------------------------------------------------------

_pa = None                          # pyaudio.PyAudio instance
_vad = None                         # webrtcvad.Vad instance
_device_index = None                # int, PyAudio device index for ReSpeaker
_shutdown = threading.Event()       # signals module shutdown
_is_speaking = False                # True while TTS is playing
_interruption_detected = False      # True when voice interruption confirmed
_interrupt_lock = threading.Lock()  # guards _is_speaking + _interruption_detected
_echo_buffer = collections.deque(maxlen=HARDWARE["vad"]["echo_buffer_maxlen"])
_monitor_thread = None              # daemon thread for interruption monitor
_speech_client = None               # Google Cloud Speech client
_eleven_client = None               # ElevenLabs client
_on_audio_level = None              # optional callback(float) for waveform display

# Convenience aliases from config
_RS = HARDWARE["respeaker"]
_VAD_CFG = HARDWARE["vad"]
_STT_CFG = HARDWARE["stt"]
_TTS_CFG = HARDWARE["tts"]

_RESPEAKER_VID = int(_RS["vid"], 16)
_RESPEAKER_PID = int(_RS["pid"], 16)
_RESPEAKER_CHANNELS = _RS["channels"]
_RESPEAKER_RATE = _RS["rate"]
_RESPEAKER_CHUNK = _RS["chunk"]

_VAD_MODE = _VAD_CFG["mode"]
_FRAME_DURATION_MS = _VAD_CFG["frame_duration_ms"]
_PADDING_DURATION_MS = _VAD_CFG["padding_duration_ms"]
_NUM_PADDING_FRAMES = int(_PADDING_DURATION_MS / _FRAME_DURATION_MS)
_VOICE_THRESHOLD = _VAD_CFG["voice_threshold"]
_CONSECUTIVE_REQUIRED = _VAD_CFG["consecutive_required"]
_ECHO_SUPPRESSION_FACTOR = _VAD_CFG["echo_suppression_factor"]

_STT_RATE = _STT_CFG["rate"]
_STT_CHUNK = int(_STT_RATE * _FRAME_DURATION_MS / 1000)

_VOICE_ID = _TTS_CFG["elevenlabs_voice_id"]
_TTS_MODEL = _TTS_CFG["elevenlabs_model"]
_TTS_RATE = _TTS_CFG["elevenlabs_rate"]
_TTS_FORMAT = _TTS_CFG["elevenlabs_format"]
_PLAYBACK_CHUNK = _TTS_CFG["playback_chunk"]

# Sentence boundary characters for flushing TTS buffer
_SENTENCE_BREAKS = set(".!?;:\n")


# ---------------------------------------------------------------------------
# 1. ReSpeaker detection (private)
# ---------------------------------------------------------------------------

def _find_respeaker():
    """Detect ReSpeaker Mic Array v2.0 via USB VID/PID."""
    if usb is None:
        log.warning("pyusb not available — cannot detect ReSpeaker")
        return None
    try:
        device = usb.core.find(idVendor=_RESPEAKER_VID, idProduct=_RESPEAKER_PID)
        if device:
            log.info("ReSpeaker Mic Array v2.0 detected: %s", device)
            return device
        else:
            log.warning("ReSpeaker Mic Array v2.0 not found via USB")
            return None
    except Exception as e:
        log.error("ReSpeaker detection error: %s", e)
        return None


def _configure_respeaker():
    """Configure ReSpeaker Mic Array v2.0 hardware settings."""
    try:
        log.info("Configuring ReSpeaker Mic Array v2.0...")

        # Basic hardware commands (best-effort, may not have pulse)
        commands = [
            "amixer -D pulse sset 'Echo Cancellation' on 2>/dev/null || true",
            "amixer -D pulse sset 'Noise Suppression' on 2>/dev/null || true",
            "amixer -D pulse sset 'Auto Gain Control' on 2>/dev/null || true",
            "amixer -D pulse sset 'Beam Forming' 'straight' 2>/dev/null || true",
            "amixer -D pulse sset 'Mic' 80% 2>/dev/null || true",
        ]

        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
            except Exception as e:
                log.debug("ReSpeaker amixer command failed: %s", e)

        # Try to configure hardware VAD via tuning interface
        try:
            import sys
            sys.path.insert(0, "./usb_4_mic_array")
            from tuning import Tuning

            dev = usb.core.find(idVendor=_RESPEAKER_VID, idProduct=_RESPEAKER_PID)
            if dev:
                tuning = Tuning(dev)
                tuning.write("GAMMAVAD_SR", 2.0)
                vad_threshold = tuning.read("GAMMAVAD_SR")
                agc_status = tuning.read("AGCONOFF")
                log.info("Hardware VAD configured: threshold=%.1f, AGC=%s", vad_threshold, agc_status)
        except Exception as e:
            log.debug("Hardware VAD tuning not available: %s", e)

        log.info("ReSpeaker v2.0 configured")
        return True

    except Exception as e:
        log.error("ReSpeaker configuration error: %s", e)
        return False


def _get_device_index():
    """Find the PyAudio device index for ReSpeaker v2.0."""
    if pyaudio is None:
        return None
    try:
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = info["name"].lower()
                if (("respeaker 4 mic array" in name and "uac1.0" in name) or
                        any(n in name for n in ["respeaker", "seeed", "mic array", "arrayuac"])):
                    if info["maxInputChannels"] > 0:
                        log.info("ReSpeaker v2.0 at PyAudio index %d: %s (channels=%d, rate=%.0f)",
                                 i, info["name"], info["maxInputChannels"], info["defaultSampleRate"])
                        return i
        finally:
            pa.terminate()

        log.warning("ReSpeaker v2.0 not found in PyAudio devices")
        return None

    except Exception as e:
        log.error("PyAudio device scan error: %s", e)
        return None


# ---------------------------------------------------------------------------
# 2. VAD and voice detection (private)
# ---------------------------------------------------------------------------

def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Butterworth bandpass filter for voice frequency range."""
    if butter is None or filtfilt is None:
        return data
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def _voice_detection_mono(audio_float):
    """Detect voice in mono float audio. Returns (is_voice, confidence)."""
    try:
        if len(audio_float) < _RESPEAKER_CHUNK:
            return False, 0.0

        audio_clipped = np.clip(audio_float, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        chunk_size = _RESPEAKER_CHUNK

        if len(audio_int16) >= chunk_size:
            chunk = audio_int16[:chunk_size]

            try:
                is_speech = _vad.is_speech(chunk.tobytes(), _RESPEAKER_RATE)

                if is_speech:
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    confidence = max(0.1, min(1.0, rms / 5000.0))
                    return True, confidence
                else:
                    # Energy fallback when VAD says no
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    if rms > 500.0:
                        confidence = min(1.0, rms / 3000.0)
                        return True, confidence
                    return False, 0.0

            except Exception as vad_err:
                log.debug("VAD frame error, using energy fallback: %s", vad_err)
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                if rms > 800.0:
                    confidence = min(1.0, rms / 4000.0)
                    return True, confidence
                return False, 0.0

        return False, 0.0

    except Exception as e:
        log.debug("Voice detection mono error: %s", e)
        return False, 0.0


# ---------------------------------------------------------------------------
# 3. Interruption monitor (private)
# ---------------------------------------------------------------------------

def _interruption_monitor():
    """Lightweight interruption monitor using ReSpeaker hardware VAD (tuning API).

    Runs in a daemon thread while TTS is playing. Sets _interruption_detected
    when consecutive voice frames exceed the threshold.
    """
    global _interruption_detected

    # Try hardware tuning interface first (no PyAudio conflict)
    respeaker_tuning = None
    try:
        import sys
        sys.path.insert(0, "./usb_4_mic_array")
        from tuning import Tuning

        dev = usb.core.find(idVendor=_RESPEAKER_VID, idProduct=_RESPEAKER_PID)
        if dev:
            respeaker_tuning = Tuning(dev)
            log.debug("Interruption monitor using hardware VAD")
        else:
            log.warning("Cannot connect to ReSpeaker tuning — monitor disabled")
            return
    except Exception as e:
        log.warning("Hardware VAD tuning unavailable: %s — monitor disabled", e)
        return

    consecutive = 0
    silence_frames = 0
    initial_silence = 30  # ~1s warm-up before detection

    required = _CONSECUTIVE_REQUIRED * 3  # Triple for reliability

    try:
        while _is_speaking and not _interruption_detected and not _shutdown.is_set():
            try:
                # Quick exit check
                if consecutive % 10 == 0:
                    if not _is_speaking or _interruption_detected or _shutdown.is_set():
                        break

                # Initial silence period to avoid self-detection
                if initial_silence > 0:
                    initial_silence -= 1
                    time.sleep(0.03)
                    continue

                speech_detected = respeaker_tuning.read("SPEECHDETECTED")
                voice_activity = respeaker_tuning.read("VOICEACTIVITY")
                voice_detected = bool(speech_detected and voice_activity)

                if voice_detected:
                    consecutive += 1
                    silence_frames = 0

                    if consecutive >= required:
                        log.info("Interruption detected (hardware VAD, %d consecutive frames)", consecutive)
                        with _interrupt_lock:
                            _interruption_detected = True
                        break
                else:
                    consecutive = max(0, consecutive - 2)
                    silence_frames += 1

                if silence_frames > 50:
                    consecutive = 0

                time.sleep(0.03)

            except Exception as e:
                log.debug("Monitor loop error: %s", e)
                time.sleep(0.1)

    except Exception as e:
        log.error("Interruption monitor error: %s", e)


def _start_monitor():
    """Start the interruption monitor daemon thread."""
    global _monitor_thread, _interruption_detected

    with _interrupt_lock:
        _interruption_detected = False

    if _monitor_thread and _monitor_thread.is_alive():
        log.debug("Interruption monitor already running")
        return

    _monitor_thread = threading.Thread(target=_interruption_monitor, daemon=True)
    _monitor_thread.start()


def _stop_monitor():
    """Stop the interruption monitor and wait for thread exit."""
    global _monitor_thread, _is_speaking, _interruption_detected

    with _interrupt_lock:
        _is_speaking = False
        _interruption_detected = True  # force loop exit

    if _monitor_thread and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=2.0)
        if _monitor_thread.is_alive():
            log.warning("Monitor thread did not terminate cleanly")

    _monitor_thread = None


def _add_echo(audio_chunk):
    """Add played audio to echo buffer for cancellation."""
    if isinstance(audio_chunk, bytes):
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        _echo_buffer.extend(samples)


# ---------------------------------------------------------------------------
# 4. STT — Google Cloud Speech (public: listen)
# ---------------------------------------------------------------------------

class _MicrophoneStream:
    """Opens a PyAudio mic stream and exposes it as a generator of audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Yield audio chunks with adaptive timeout."""
        chunk_timeout = 2.0
        chunks_without_data = 0
        max_empty = 3

        while not self.closed:
            try:
                chunk = self._buff.get(timeout=chunk_timeout)
                if chunk is None:
                    return

                chunks_without_data = 0
                data = [chunk]

                # Drain any additional buffered chunks
                while True:
                    try:
                        chunk = self._buff.get(block=False)
                        if chunk is None:
                            return
                        data.append(chunk)
                    except queue.Empty:
                        break

                if data:
                    yield b"".join(data)

            except queue.Empty:
                chunks_without_data += 1
                if chunks_without_data >= max_empty:
                    log.debug("Prolonged silence on microphone")
                    chunk_timeout = 0.5
                else:
                    continue


def listen() -> str:
    """Listen via Google Cloud STT and return transcribed text, or empty string on failure.

    Uses streaming_recognize with single_utterance=True and es-ES language.
    Enters quiet mode internally — callers should not manage that.
    """
    global _is_speaking

    if _speech_client is None:
        log.error("Google STT client not available")
        return ""

    if speech is None:
        log.error("google.cloud.speech not importable")
        return ""

    # Wait for any ongoing TTS to finish
    if _is_speaking:
        log.info("Waiting for TTS to finish before listening...")
        timeout = 30
        start = time.time()
        while _is_speaking and (time.time() - start < timeout):
            time.sleep(0.1)
        time.sleep(0.5)

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=_STT_RATE,
        language_code=_STT_CFG["language"],
        enable_automatic_punctuation=True,
        use_enhanced=_STT_CFG.get("use_enhanced", True),
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=False,
        single_utterance=True,
    )

    log.info("Listening... (speak now)")

    start_time = time.time()
    max_listen = _STT_CFG.get("max_listen_seconds", 30)
    silence_threshold = _STT_CFG.get("silence_threshold_seconds", 5)
    timeout_seconds = _STT_CFG.get("timeout_seconds", 15)

    try:
        with _MicrophoneStream(_STT_RATE, _STT_CHUNK) as stream:
            audio_generator = stream.generator()
            last_audio_time = start_time

            def request_generator():
                nonlocal last_audio_time
                chunks_sent = 0

                for content in audio_generator:
                    now = time.time()

                    if now - start_time > max_listen:
                        log.info("Total listen timeout (%ds) reached", max_listen)
                        break

                    if chunks_sent > 0 and now - last_audio_time > silence_threshold:
                        log.info("Silence timeout (%ds) reached", silence_threshold)
                        break

                    if _is_speaking:
                        log.info("Interrupting STT — android is speaking")
                        break

                    last_audio_time = now
                    chunks_sent += 1
                    yield speech.StreamingRecognizeRequest(audio_content=content)

            try:
                responses = _speech_client.streaming_recognize(
                    config=streaming_config,
                    requests=request_generator(),
                )

                result_q = queue.Queue()

                def process_responses():
                    try:
                        for response in responses:
                            if _is_speaking:
                                result_q.put("")
                                return
                            if response.results and response.results[0].alternatives:
                                transcript = response.results[0].alternatives[0].transcript.strip()
                                if transcript:
                                    result_q.put(transcript)
                                    return
                        result_q.put("")
                    except Exception as e:
                        log.error("STT response processing error: %s", e)
                        result_q.put("")

                t = threading.Thread(target=process_responses, daemon=True)
                t.start()

                try:
                    result = result_q.get(timeout=timeout_seconds)
                    if result:
                        return result
                except queue.Empty:
                    log.info("STT timeout (%ds) — no clear speech detected", timeout_seconds)

            except Exception as e:
                err = str(e)
                if "Deadline" in err or "DEADLINE_EXCEEDED" in err:
                    log.info("STT deadline exceeded")
                elif "inactive" in err.lower():
                    log.info("STT stream inactive — retry may help")
                elif "OutOfRange" in err:
                    log.info("STT audio out of range — speak closer to mic")
                elif "InvalidArgument" in err:
                    log.info("STT invalid audio — speak more clearly")
                else:
                    log.error("Google STT error: %s", e)
                time.sleep(0.5)

    except Exception as e:
        log.error("listen() error: %s", e)

    return ""


# ---------------------------------------------------------------------------
# 5. TTS — ElevenLabs + espeak-ng fallback (public: speak, speak_stream)
# ---------------------------------------------------------------------------

def _speak_espeak(text):
    """Fallback TTS via espeak-ng subprocess. Returns SpeakResult (never interrupted)."""
    try:
        subprocess.run(
            ["espeak-ng", "-v", "es", "-s", "160", text],
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        log.warning("espeak-ng not installed — cannot speak")
    except Exception as e:
        log.error("espeak-ng error: %s", e)
    return SpeakResult(interrupted=False, spoken_text=text)


def _speak_elevenlabs_sentence(pa_stream, text):
    """Synthesize and play one sentence via ElevenLabs streaming API.

    Returns False if interrupted, True if completed normally.
    """
    global _interruption_detected

    if not text.strip():
        return True

    if _interruption_detected:
        log.info("Interrupted before synthesis of: %s...", text[:40])
        return False

    try:
        # Small pause to let interruption system stabilize
        time.sleep(0.2)

        audio_iter = _eleven_client.text_to_speech.stream(
            text=text,
            voice_id=_VOICE_ID,
            model_id=_TTS_MODEL,
            output_format=_TTS_FORMAT,
        )

        chunk_count = 0

        for audio_chunk in audio_iter:
            chunk_count += 1

            # Grace period: ignore interruptions during first 5 chunks
            if chunk_count <= 5:
                pa_stream.write(audio_chunk)
                continue

            # Check interruption
            if _interruption_detected:
                log.info("Interrupted at chunk %d", chunk_count)
                return False

            # Add to echo buffer for cancellation
            _add_echo(audio_chunk)

            # Push RMS level to waveform callback
            if _on_audio_level is not None:
                try:
                    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
                    _on_audio_level(min(1.0, rms * 3.0))  # amplify for visibility
                except Exception:
                    pass

            pa_stream.write(audio_chunk)

            # Post-write check
            if _interruption_detected:
                log.info("Interrupted after chunk %d", chunk_count)
                return False

        return True

    except Exception as e:
        log.error("ElevenLabs TTS error: %s", e)
        return False


def _speak_elevenlabs(chunks):
    """Core ElevenLabs streaming TTS with interruption support.

    Args:
        chunks: iterable of text strings (may be a generator).

    Returns:
        SpeakResult with interruption status and all spoken text.
    """
    global _is_speaking, _interruption_detected

    with _interrupt_lock:
        _is_speaking = True
        _interruption_detected = False

    _start_monitor()

    pa_inst = None
    pa_stream = None
    full_response = ""
    was_interrupted = False

    try:
        pa_inst = pyaudio.PyAudio()
        pa_stream = pa_inst.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_TTS_RATE,
            output=True,
            frames_per_buffer=_PLAYBACK_CHUNK,
        )

        sentence_buffer = ""

        # Handle both string and generator inputs
        if isinstance(chunks, str):
            text_iterator = [chunks]
        else:
            text_iterator = chunks

        for text_chunk in text_iterator:
            # Check interruption before processing each chunk
            if _interruption_detected:
                was_interrupted = True
                log.info("Interrupted during text generation")
                break

            if not text_chunk:
                continue

            full_response += text_chunk
            sentence_buffer += text_chunk

            # Flush on sentence boundary
            if any(c in text_chunk for c in _SENTENCE_BREAKS):
                if not _speak_elevenlabs_sentence(pa_stream, sentence_buffer.strip()):
                    was_interrupted = True
                    break
                sentence_buffer = ""

            # Post-chunk check
            if _interruption_detected:
                was_interrupted = True
                break

        # Flush remaining text
        if sentence_buffer.strip() and not _interruption_detected:
            if not _speak_elevenlabs_sentence(pa_stream, sentence_buffer.strip()):
                was_interrupted = True

    except Exception as e:
        log.error("ElevenLabs streaming error: %s", e)
    finally:
        _stop_monitor()

        # Reset audio level so waveform decays
        if _on_audio_level is not None:
            try:
                _on_audio_level(0.0)
            except Exception:
                pass

        with _interrupt_lock:
            _is_speaking = False
            _interruption_detected = False

        if pa_stream:
            try:
                pa_stream.stop_stream()
                pa_stream.close()
            except Exception:
                pass
        if pa_inst:
            try:
                pa_inst.terminate()
            except Exception:
                pass

    return SpeakResult(interrupted=was_interrupted, spoken_text=full_response)


def speak(text: str) -> SpeakResult:
    """Speak a single text string via TTS.

    Uses ElevenLabs if available, falls back to espeak-ng.
    Supports voice interruption detection via the ReSpeaker monitor.
    """
    if not text or not text.strip():
        return SpeakResult(interrupted=False, spoken_text="")

    if _eleven_client is None:
        log.warning("ElevenLabs not available, using espeak-ng fallback")
        return _speak_espeak(text)

    if pyaudio is None:
        log.warning("PyAudio not available, using espeak-ng fallback")
        return _speak_espeak(text)

    return _speak_elevenlabs(text)


def speak_stream(chunks) -> SpeakResult:
    """Speak from a streaming text generator.

    Each yielded item should be a text string. Sentences are flushed
    to ElevenLabs on boundary characters (.!?;:\\n) for low latency.
    Supports voice interruption detection.
    """
    if _eleven_client is None:
        # Collect all chunks and use espeak fallback
        full = "".join(c for c in chunks if c)
        if not full.strip():
            return SpeakResult(interrupted=False, spoken_text="")
        log.warning("ElevenLabs not available, using espeak-ng fallback")
        return _speak_espeak(full)

    if pyaudio is None:
        full = "".join(c for c in chunks if c)
        if not full.strip():
            return SpeakResult(interrupted=False, spoken_text="")
        log.warning("PyAudio not available, using espeak-ng fallback")
        return _speak_espeak(full)

    return _speak_elevenlabs(chunks)


# ---------------------------------------------------------------------------
# 6. Public helpers
# ---------------------------------------------------------------------------

def set_audio_level_callback(callback):
    """Set a callback that receives RMS audio level (0.0-1.0) during TTS playback.

    Used by main.py to drive the display waveform visualizer.
    """
    global _on_audio_level
    _on_audio_level = callback


def is_voice_detected() -> bool:
    """Check if a voice interruption has been detected (thread-safe)."""
    return _interruption_detected


def stop_speaking() -> None:
    """Force-stop any ongoing TTS playback."""
    global _is_speaking, _interruption_detected
    with _interrupt_lock:
        _interruption_detected = True
        _is_speaking = False
    _stop_monitor()


# ---------------------------------------------------------------------------
# 7. init() and cleanup()
# ---------------------------------------------------------------------------

def init() -> bool:
    """Initialize the audio subsystem: ReSpeaker, VAD, STT, TTS clients.

    Returns True if core components initialized, False if hardware not found.
    Gracefully degrades — STT/TTS clients are initialized even without ReSpeaker.
    """
    global _pa, _vad, _device_index, _speech_client, _eleven_client

    log.info("Initializing audio subsystem...")

    # 1. Initialize Google STT client
    if speech is not None and GOOGLE_CREDENTIALS:
        try:
            _speech_client = speech.SpeechClient()
            log.info("Google Cloud STT client initialized")
        except Exception as e:
            log.error("Google STT init failed: %s", e)
            _speech_client = None
    elif speech is not None:
        try:
            _speech_client = speech.SpeechClient()
            log.info("Google Cloud STT client initialized (default credentials)")
        except Exception as e:
            log.warning("Google STT init failed (no explicit credentials): %s", e)
            _speech_client = None
    else:
        log.warning("google.cloud.speech not available — STT disabled")

    # 2. Initialize ElevenLabs client
    if ElevenLabs is not None and ELEVENLABS_API_KEY:
        try:
            _eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            log.info("ElevenLabs TTS client initialized")
        except Exception as e:
            log.error("ElevenLabs init failed: %s", e)
            _eleven_client = None
    else:
        log.warning("ElevenLabs not available — TTS will use espeak-ng fallback")

    # 3. Initialize WebRTC VAD
    if webrtcvad is not None:
        try:
            _vad = webrtcvad.Vad(_VAD_MODE)
            log.info("WebRTC VAD initialized (mode=%d)", _VAD_MODE)
        except Exception as e:
            log.error("WebRTC VAD init failed: %s", e)
    else:
        log.warning("webrtcvad not available — VAD disabled")

    # 4. Detect and configure ReSpeaker
    device = _find_respeaker()
    if not device:
        log.warning("ReSpeaker not found — using default microphone")
        return False

    _configure_respeaker()

    # 5. Get PyAudio device index
    _device_index = _get_device_index()
    if _device_index is None:
        log.warning("ReSpeaker not available in PyAudio")
        return False

    log.info("Audio subsystem initialized (ReSpeaker at index %d)", _device_index)
    return True


def cleanup() -> None:
    """Release all audio resources."""
    global _pa, _vad, _device_index, _speech_client, _eleven_client

    log.info("Cleaning up audio subsystem...")

    _shutdown.set()

    # Stop any active monitor
    _stop_monitor()

    _pa = None
    _vad = None
    _device_index = None
    _speech_client = None
    _eleven_client = None

    log.info("Audio subsystem cleaned up")
