"""NeoPixel LED animations — steampunk, listening, speaking.

Drives 13 WS2812 NeoPixel LEDs on GPIO 18 via the neopixel library.
A single daemon thread runs continuously and switches behavior based on
_current_mode.  Modes:

  steampunk  — copper/brass/rust colour pulsation (idle state)
  listening  — purple quiet-glow pulsation
  speaking   — red Knight-Rider sweep
  off        — all LEDs off, thread sleeps

All constants come from config.HARDWARE["leds"].  The module degrades
gracefully when board/neopixel are unavailable (e.g. on a dev laptop).
"""

import logging
import math
import random
import threading
import time

from config import HARDWARE

log = logging.getLogger("leds")

# ---------------------------------------------------------------------------
# Hardware abstraction
# ---------------------------------------------------------------------------
try:
    import board
    import neopixel
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False
    log.warning("board/neopixel not available — LED module in stub mode")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG = HARDWARE["leds"]
_COUNT      = _CFG["count"]                          # 13
_BRIGHTNESS = _CFG.get("brightness", 0.05)
_PALETA     = [tuple(c) for c in _CFG["paleta"]]     # copper/brass/rust colours
_KR_COLOR   = tuple(_CFG["knight_rider_color"])       # red
_KR_DELAY   = _CFG.get("knight_rider_delay", 0.04)
_QG_COLOR   = tuple(_CFG["quiet_glow_color"])         # purple

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_pixels      = None
_thread      = None
_current_mode = "off"
_mode_changed = threading.Event()
_shutdown     = threading.Event()
_lock         = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pulso_oxidado(t: float, i: int) -> tuple:
    """Return a steampunk-palette colour for LED i at time t."""
    intensidad = (math.sin(t + i * 0.5) + 1) / 2
    base = _PALETA[i % len(_PALETA)]
    return tuple(int(c * intensidad) for c in base)


def _run_steampunk(delay: float = 0.04) -> bool:
    """Run one frame of steampunk animation. Returns False when mode changed."""
    t = 0.0
    glitch_timer = 0
    orig_brightness = _pixels.brightness
    try:
        while _current_mode == "steampunk" and not _shutdown.is_set():
            for i in range(_COUNT):
                if glitch_timer > 0 and i == random.randint(0, _COUNT - 1):
                    _pixels[i] = random.choice([(0, 255, 180), (255, 255, 255)])
                else:
                    _pixels[i] = _pulso_oxidado(t, i)
            _pixels.show()
            _mode_changed.wait(timeout=delay)
            _mode_changed.clear()
            t += 0.1
            glitch_timer = (glitch_timer - 1) if glitch_timer > 0 else random.randint(0, 20)
    finally:
        _pixels.fill((0, 0, 0))
        _pixels.show()


def _run_listening(
    pulse_period: float = 2.0,
    delay: float = 0.01,
    min_intensity: float = 0.2,
    max_intensity: float = 0.9,
    smooth_factor: float = 0.9,
) -> None:
    """Purple quiet-glow pulsation for listening mode."""
    t = 0.0
    center = (_COUNT - 1) / 2.0
    prev_intensity = (min_intensity + max_intensity) / 2

    while _current_mode == "listening" and not _shutdown.is_set():
        raw = (math.sin(2 * math.pi * t / pulse_period) + 1) / 2
        target = min_intensity + raw * (max_intensity - min_intensity)
        intensity = prev_intensity * smooth_factor + target * (1 - smooth_factor)
        prev_intensity = intensity

        for i in range(_COUNT):
            spatial = max(0.0, 1.0 - abs(i - center) / center)
            val = intensity * spatial
            _pixels[i] = tuple(int(c * val) for c in _QG_COLOR)

        _pixels.show()
        _mode_changed.wait(timeout=delay)
        _mode_changed.clear()
        t += delay

    _pixels.fill((0, 0, 0))
    _pixels.show()


def _run_speaking() -> None:
    """Red Knight-Rider sweep for speaking mode."""
    pos = 0
    forward = True

    while _current_mode == "speaking" and not _shutdown.is_set():
        _pixels.fill((0, 0, 0))

        for i in range(_COUNT):
            if i == pos:
                _pixels[i] = _KR_COLOR
            elif abs(i - pos) == 1:
                _pixels[i] = tuple(int(c * 0.4) for c in _KR_COLOR)
            elif abs(i - pos) == 2:
                _pixels[i] = tuple(int(c * 0.1) for c in _KR_COLOR)

        _pixels.show()

        if forward:
            pos += 1
            if pos >= _COUNT - 1:
                forward = False
        else:
            pos -= 1
            if pos <= 0:
                forward = True

        _mode_changed.wait(timeout=_KR_DELAY)
        _mode_changed.clear()

    _pixels.fill((0, 0, 0))
    _pixels.show()


def _animation_loop() -> None:
    """Single long-lived daemon thread that dispatches to mode functions."""
    log.debug("LED animation thread started")
    while not _shutdown.is_set():
        mode = _current_mode
        try:
            if mode == "steampunk":
                _run_steampunk()
            elif mode == "listening":
                _run_listening()
            elif mode == "speaking":
                _run_speaking()
            else:
                # "off" or unknown: LEDs already dark, just wait for a mode change
                _mode_changed.wait(timeout=1.0)
                _mode_changed.clear()
        except Exception as e:
            log.error("LED animation error in mode '%s': %s", mode, e)
            # Brief pause before retrying to avoid tight error loops
            time.sleep(0.5)
    log.debug("LED animation thread stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init() -> bool:
    """Initialise the NeoPixel strip and start the steampunk animation.

    Returns True when real hardware is available, False in stub mode.
    """
    global _pixels, _thread, _current_mode

    if not _HAS_HARDWARE:
        log.warning("LEDs: no hardware — stub mode active")
        return False

    try:
        _pixels = neopixel.NeoPixel(
            board.D18,
            _COUNT,
            brightness=_BRIGHTNESS,
            auto_write=False,
        )
        _pixels.fill((0, 0, 0))
        _pixels.show()
    except Exception as e:
        log.error("LEDs: failed to initialise NeoPixel: %s", e)
        return False

    _current_mode = "steampunk"
    _mode_changed.set()

    _thread = threading.Thread(target=_animation_loop, daemon=True, name="led-anim")
    _thread.start()

    log.info("LEDs: NeoPixel x%d on GPIO18 — steampunk mode active", _COUNT)
    return True


def set_mode(mode: str) -> None:
    """Switch the LED animation mode.

    Valid modes: "steampunk", "listening", "speaking", "off".
    Safe to call from any thread.  No-op in stub mode.
    """
    global _current_mode

    if not _HAS_HARDWARE or _pixels is None:
        return

    valid = {"steampunk", "listening", "speaking", "off"}
    if mode not in valid:
        log.warning("LEDs: unknown mode '%s' — ignoring", mode)
        return

    with _lock:
        if _current_mode == mode:
            return
        log.debug("LEDs: mode %s → %s", _current_mode, mode)
        _current_mode = mode
        _mode_changed.set()

    if mode == "off":
        # Give the animation thread a moment to clear the strip
        time.sleep(0.05)


def cleanup() -> None:
    """Stop all animations and turn off all LEDs."""
    global _current_mode

    if not _HAS_HARDWARE or _pixels is None:
        return

    log.info("LEDs: shutting down")
    _current_mode = "off"
    _shutdown.set()
    _mode_changed.set()

    if _thread and _thread.is_alive():
        _thread.join(timeout=2)

    try:
        _pixels.fill((0, 0, 0))
        _pixels.show()
    except Exception as e:
        log.warning("LEDs: cleanup error: %s", e)
