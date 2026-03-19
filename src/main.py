"""Botijo — Androide Conversacional.

Entry point. Parses args, loads config, initializes modules, runs main loop.
Supports two modes:
  - botijo/botija: full conversational loop with inactivity timeout
  - barbacoa: autonomous phrase emission + face reactions + conversation

Usage: PYTHONPATH=src:vendor python src/main.py [--mode botijo|botija|barbacoa]
"""

import argparse
import logging
import random
import signal
import sys
import threading
import time

from config import HARDWARE
import audio
import brain
import buttons
import display
import leds
import personality
import servos
import vision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("botijo")

# ---------------------------------------------------------------------------
# Module tracking
# ---------------------------------------------------------------------------
_modules: list = []
_cleaned_up = False


def _cleanup():
    """Call cleanup() on all initialized modules (idempotent)."""
    global _cleaned_up
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


def _signal_handler(signum, frame):
    _cleanup()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Barbacoa mode helpers
# ---------------------------------------------------------------------------

class PhraseManager:
    """Manages phrase pools with no-repeat guarantee.

    Draws from per-category pools and a global pool.  When a pool is
    exhausted it refills from the used-phrase buffer so every phrase
    gets a turn before any repeats.
    """

    def __init__(self, phrases_dict: dict):
        self._pools = {k: list(v) for k, v in phrases_dict.items()}
        for pool in self._pools.values():
            random.shuffle(pool)
        # Flat pool of all phrases for any() picks
        self._all: list[str] = []
        for v in phrases_dict.values():
            self._all.extend(v)
        random.shuffle(self._all)
        # Track used phrases per category for refill
        self._used_by_cat: dict[str, list[str]] = {k: [] for k in phrases_dict}

    def any(self) -> str:
        """Return a random phrase from any category (no repeats until pool exhausted)."""
        if not self._all:
            for v in self._pools.values():
                self._all.extend(v)
            random.shuffle(self._all)
        return self._all.pop()

    def from_cat(self, category: str) -> str:
        """Return a random phrase from a specific category, falling back to any()."""
        pool = self._pools.get(category)
        if not pool:
            # Category empty or unknown — refill or fallback
            used = self._used_by_cat.get(category, [])
            if used:
                pool = self._pools[category] = list(used)
                used.clear()
                random.shuffle(pool)
            else:
                return self.any()
        choice = pool.pop()
        self._used_by_cat.setdefault(category, []).append(choice)
        return choice


class AutoBanterThread(threading.Thread):
    """Daemon thread that speaks a random phrase every min_s..max_s seconds."""

    def __init__(self, speaker, phrase_source, min_s=120, max_s=240,
                 consult_is_speaking=None, preferred_cats=None):
        super().__init__(daemon=True)
        self.speaker = speaker
        self.src = phrase_source
        self.min_s = min_s
        self.max_s = max_s
        self.consult_is_speaking = consult_is_speaking or (lambda: False)
        self.preferred_cats = preferred_cats or []
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            wait = random.randint(self.min_s, self.max_s)
            # Sleep in small increments so stop_event is responsive
            for _ in range(wait * 10):
                if self.stop_event.is_set():
                    return
                time.sleep(0.1)
            if self.consult_is_speaking():
                continue
            try:
                if self.preferred_cats:
                    cat = random.choice(self.preferred_cats)
                    text = self.src.from_cat(cat)
                else:
                    text = self.src.any()
                log.info("[AUTOBANTER] Speaking: %s", text[:60])
                self.speaker(text)
            except Exception as e:
                log.warning("[AUTOBANTER] Error: %s", e)


class FaceBanter:
    """Reacts to detected faces by speaking a phrase (with cooldown)."""

    def __init__(self, speaker, phrase_source, cooldown_s=35,
                 category_sequence=None, consult_is_speaking=None):
        self.speaker = speaker
        self.src = phrase_source
        self.cooldown_s = cooldown_s
        self.last_time = 0.0
        self.consult_is_speaking = consult_is_speaking or (lambda: False)
        self.category_sequence = category_sequence or [
            "ninos_humor_blanco", "fuego_barbacoa",
            "aristocracia_culta", "interaccion_directa",
        ]

    def maybe_emit(self):
        """Check cooldown and emit a phrase if enough time has passed."""
        if self.consult_is_speaking():
            return
        now = time.time()
        if now - self.last_time < self.cooldown_s:
            return
        cat = self.category_sequence[int(now) % len(self.category_sequence)]
        self.last_time = now
        try:
            text = self.src.from_cat(cat)
            log.info("[FACEBANTER] Speaking: %s", text[:60])
            self.speaker(text)
        except Exception as e:
            log.warning("[FACEBANTER] Error: %s", e)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def _init_modules(mode: str) -> None:
    """Initialize all hardware modules with graceful degradation.

    Each module can fail independently — the system continues with
    whatever hardware is available.
    """
    # Brain first (loads personality + LLM clients)
    try:
        brain.init(mode)
        _modules.append(brain)
    except Exception as e:
        log.error("Brain init failed: %s", e)

    # Audio
    try:
        hw = audio.init()
        _modules.append(audio)
        if hw:
            log.info("Audio: ReSpeaker v2.0 active (VAD + interruption detection)")
        else:
            log.warning("Audio: initialized without ReSpeaker (degraded mode)")
    except Exception as e:
        _modules.append(audio)
        log.error("Audio init error: %s", e)

    # Servos
    try:
        hw = servos.init()
        _modules.append(servos)
        if hw:
            log.info("Servos: PCA9685 active (eyes + eyelids + tentacles)")
        else:
            log.warning("Servos: initialized without hardware (stub mode)")
    except Exception as e:
        _modules.append(servos)
        log.error("Servos init error: %s", e)

    # LEDs
    try:
        hw = leds.init()
        _modules.append(leds)
        if hw:
            log.info("LEDs: NeoPixel active (steampunk animation running)")
        else:
            log.warning("LEDs: initialized without hardware (stub mode)")
    except Exception as e:
        _modules.append(leds)
        log.error("LEDs init error: %s", e)

    # Vision
    try:
        hw = vision.init()
        _modules.append(vision)
        if hw:
            log.info("Vision: IMX500 face detection active")
        else:
            log.warning("Vision: initialized without hardware (stub mode)")
    except Exception as e:
        _modules.append(vision)
        log.error("Vision init error: %s", e)

    # Display
    try:
        hw = display.init()
        _modules.append(display)
        if hw:
            log.info("Display: Waveshare 1.9\" LCD active (%dx%d)",
                     display.WIDTH, display.HEIGHT)
        else:
            log.warning("Display: initialized without hardware (stub mode)")
    except Exception as e:
        _modules.append(display)
        log.error("Display init error: %s", e)


def _init_buttons(callbacks: dict) -> None:
    """Initialize buttons with the given callbacks."""
    try:
        hw = buttons.init(callbacks)
        _modules.append(buttons)
        if hw:
            log.info("Buttons: 4 GPIO buttons active")
        else:
            log.warning("Buttons: initialized without hardware (stub mode)")
    except Exception as e:
        _modules.append(buttons)
        log.error("Buttons init error: %s", e)


def _speak_safe(text: str) -> audio.SpeakResult:
    """Speak text, returning SpeakResult. Catches errors gracefully."""
    try:
        return audio.speak(text)
    except Exception as e:
        log.warning("Could not speak: %s", e)
        return audio.SpeakResult(interrupted=False, spoken_text="")


def _track_faces():
    """Check for detected faces and move eyes toward the best one."""
    try:
        faces = vision.get_faces()
        if faces:
            face = faces[0]  # Largest/closest face
            servos.look_at(face.x, face.y)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Face tracking background thread
# ---------------------------------------------------------------------------

class _FaceTracker(threading.Thread):
    """Continuously tracks faces in the background (~10 Hz)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            _track_faces()
            self.stop_event.wait(0.1)


# ---------------------------------------------------------------------------
# Main conversational loop (botijo / botija modes)
# ---------------------------------------------------------------------------

def _run_conversational():
    """Full conversational loop with inactivity timeout and sleep mode."""
    behavior = HARDWARE["behavior"]
    inactivity_timeout = behavior.get("inactivity_timeout", 300)
    warning_time = behavior.get("warning_time", 240)

    last_activity = time.time()
    has_warned = False
    sleeping = False

    # Start background face tracking
    face_tracker = _FaceTracker()
    face_tracker.start()

    log.info("Conversational loop running — listening...")

    try:
        while True:
            now = time.time()
            inactive_secs = now - last_activity

            # --- Sleep mode ---
            if sleeping:
                # In sleep mode: listen for reactivation
                servos.set_quiet_mode(True)
                text = audio.listen()
                if text:
                    log.info("Reactivating from sleep — heard: %s", text)
                    sleeping = False
                    has_warned = False
                    last_activity = time.time()

                    # Wake up hardware
                    servos.set_quiet_mode(False)
                    leds.set_mode("steampunk")
                    display.show_eyes("neutral")

                    _speak_safe("Ah, has vuelto. Procesando tu peticion.")

                    # Process the wake-up text as a normal query
                    leds.set_mode("speaking")
                    display.show_waveform(0.5)
                    chunks = brain.chat_stream(text)
                    result = audio.speak_stream(chunks)
                    if result.interrupted:
                        brain.note_interruption(result.spoken_text)
                        log.info("Response interrupted")
                    leds.set_mode("steampunk")
                    display.show_eyes("neutral")
                    last_activity = time.time()
                else:
                    time.sleep(0.1)
                continue

            # --- Inactivity timeout check ---
            if inactive_secs > inactivity_timeout:
                log.info("Inactivity timeout (%ds) — entering sleep mode",
                         inactivity_timeout)
                sleeping = True
                has_warned = False
                servos.set_quiet_mode(True)
                leds.set_mode("off")
                display.show_eyes("sleepy")
                continue

            if inactive_secs > warning_time and not has_warned:
                log.info("Inactivity warning at %ds", int(inactive_secs))
                _speak_safe("Sigues ahi, saco de carne? Tu silencio es sospechoso.")
                has_warned = True
                # Don't reset last_activity — warning doesn't count as interaction

            # --- Normal listening ---
            servos.set_quiet_mode(True)
            leds.set_mode("listening")
            display.show_eyes("neutral")

            text = audio.listen()

            servos.set_quiet_mode(False)

            if text:
                log.info("Heard: %s", text)
                last_activity = time.time()
                has_warned = False

                # Respond
                leds.set_mode("speaking")
                display.show_waveform(0.5)

                chunks = brain.chat_stream(text)
                result = audio.speak_stream(chunks)

                if result.interrupted:
                    brain.note_interruption(result.spoken_text)
                    log.info("Response interrupted — user can say 'continua'/'sigue'")

                leds.set_mode("steampunk")
                display.show_eyes("neutral")
            else:
                # No speech detected — idle
                leds.set_mode("steampunk")
                time.sleep(0.1)

    finally:
        face_tracker.stop()
        face_tracker.join(timeout=2)


# ---------------------------------------------------------------------------
# Barbacoa mode loop
# ---------------------------------------------------------------------------

def _run_barbacoa():
    """Barbacoa mode: autonomous phrases + face reactions + conversation."""
    persona = personality._current or {}

    # Load phrase pools
    phrases_dict = personality.get_phrases()
    if not phrases_dict:
        log.error("No phrases found for barbacoa mode — falling back to conversational")
        _run_conversational()
        return

    phrase_mgr = PhraseManager(phrases_dict)

    # Speaking state checker for banter threads
    def is_speaking():
        return audio._is_speaking

    # Auto banter config from personality
    auto_cfg = persona.get("auto_banter", {})
    auto_banter = AutoBanterThread(
        speaker=lambda text: _speak_safe(text),
        phrase_source=phrase_mgr,
        min_s=auto_cfg.get("min_seconds", 120),
        max_s=auto_cfg.get("max_seconds", 240),
        consult_is_speaking=is_speaking,
        preferred_cats=auto_cfg.get("preferred_categories", [
            "fuego_barbacoa", "carne_comida", "aristocracia_culta",
        ]),
    )

    # Face banter config from personality
    face_cfg = persona.get("face_banter", {})
    face_banter = FaceBanter(
        speaker=lambda text: _speak_safe(text),
        phrase_source=phrase_mgr,
        cooldown_s=face_cfg.get("cooldown_seconds", 35),
        category_sequence=face_cfg.get("category_sequence", [
            "ninos_humor_blanco", "fuego_barbacoa",
            "aristocracia_culta", "interaccion_directa",
        ]),
        consult_is_speaking=is_speaking,
    )

    auto_banter.start()
    log.info("Barbacoa mode — auto-banter every %d-%ds, face cooldown %ds",
             auto_banter.min_s, auto_banter.max_s, face_banter.cooldown_s)

    try:
        while True:
            # Face detection — may trigger a phrase
            faces = vision.get_faces()
            if faces:
                face = faces[0]
                servos.look_at(face.x, face.y)
                face_banter.maybe_emit()

            # Listen for conversation between autonomous phrases
            servos.set_quiet_mode(True)
            leds.set_mode("listening")

            text = audio.listen()

            servos.set_quiet_mode(False)

            if text:
                log.info("[BARBACOA] Heard: %s", text)
                leds.set_mode("speaking")
                display.show_waveform(0.5)

                chunks = brain.chat_stream(text)
                result = audio.speak_stream(chunks)

                if result.interrupted:
                    brain.note_interruption(result.spoken_text)

                leds.set_mode("steampunk")
                display.show_eyes("neutral")
            else:
                leds.set_mode("steampunk")
                time.sleep(0.1)

    finally:
        auto_banter.stop()
        auto_banter.join(timeout=3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Botijo — Androide Conversacional")
    parser.add_argument("--mode", default="botijo",
                        choices=["botijo", "botija", "barbacoa"],
                        help="Personality mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log.info("Botijo starting — mode: %s", args.mode)
    log.info("Hardware config loaded: %d servo channels, %d LEDs, %d buttons",
             16, HARDWARE["leds"]["count"],
             len([k for k in HARDWARE["buttons"] if k.startswith("btn")]))

    # --- Button callbacks ---
    _sleep_toggle = {"sleeping": False}

    def _btn1_toggle_sleep():
        """Toggle between listening and sleep mode."""
        _sleep_toggle["sleeping"] = not _sleep_toggle["sleeping"]
        if _sleep_toggle["sleeping"]:
            log.info("Button 1: entering sleep mode")
            servos.set_quiet_mode(True)
            leds.set_mode("off")
            display.show_eyes("sleepy")
        else:
            log.info("Button 1: waking up")
            servos.set_quiet_mode(False)
            leds.set_mode("steampunk")
            display.show_eyes("neutral")

    def _btn2_reset_history():
        """Reset conversation history."""
        log.info("Button 2: resetting conversation history")
        brain.reset_history()
        _speak_safe("Memoria borrada. Empezamos de cero.")

    button_callbacks = {
        "btn1": _btn1_toggle_sleep,
        "btn2": _btn2_reset_history,
    }

    # --- Initialize all modules ---
    _init_modules(args.mode)
    _init_buttons(button_callbacks)

    # --- Speak greeting ---
    greeting = personality.get_greeting()
    log.info("Greeting: %s", greeting)
    _speak_safe(greeting)

    # --- Show initial eyes ---
    display.show_eyes("neutral")
    leds.set_mode("steampunk")

    # --- Run appropriate loop ---
    log.info("Botijo ready — entering %s loop", args.mode)
    try:
        if args.mode == "barbacoa":
            _run_barbacoa()
        else:
            _run_conversational()
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
