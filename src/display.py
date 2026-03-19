"""Waveshare 1.9" LCD display — eye expressions and waveform visualizer.

Drives the 170x320 (rotated to 320x170 landscape) SPI LCD for showing
eye expressions (neutral, angry, sleepy) and audio-reactive waveforms.

All constants come from config.HARDWARE["display"].
The Waveshare driver is vendored at vendor/lib/ and loaded via PYTHONPATH.
Graceful degradation: if the LCD driver is unavailable (e.g. on a dev
laptop), init() returns False and all rendering functions become no-ops.
"""

import logging
import math
import threading
import time

from config import HARDWARE

log = logging.getLogger("botijo.display")

# ---------------------------------------------------------------------------
# Hardware abstraction
# ---------------------------------------------------------------------------
try:
    from lib import LCD_1inch9
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False
    log.info("LCD_1inch9 not available — display module will run in stub mode")

# PIL is always available (required for rendering even in stub mode for API compat)
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    log.warning("PIL not available — display rendering disabled")

# numpy for waveform rendering (optional, falls back to pure math)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = HARDWARE["display"]
_NATIVE_W = _cfg.get("native_width", 170)
_NATIVE_H = _cfg.get("native_height", 320)
_ROTATION = _cfg.get("rotation", 270)

# After rotation: effective display is 320x170
WIDTH = _NATIVE_H   # 320
HEIGHT = _NATIVE_W   # 170

_RST_PIN = _cfg.get("rst_pin", 27)
_DC_PIN = _cfg.get("dc_pin", 25)
_BL_PIN = _cfg.get("bl_pin", 24)

# ---------------------------------------------------------------------------
# Waveform constants (from BrutusVisualizer)
# ---------------------------------------------------------------------------
_WV_FPS = 20
_WV_BASE_AMP = 0.30
_WV_WAVELENGTH = 40
_WV_PHASE_SPEED = 10
_WV_GAIN_NOISE = 1.0
_WV_SMOOTH = 0.85
_WV_BASE_COLOR = (255, 255, 255)
_WV_NOISE_COLOR = (255, 0, 255)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_disp = None
_initialized = False
_waveform_thread = None
_waveform_stop = threading.Event()
_waveform_level = 0.0
_waveform_level_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Eye expression rendering (PIL-based)
# ---------------------------------------------------------------------------

def _draw_eye(draw, cx, cy, w, h, expression="neutral"):
    """Draw a single eye on the given ImageDraw context.

    cx, cy: center of the eye
    w, h: size of the eye
    expression: neutral, angry, sleepy
    """
    half_w = w // 2
    half_h = h // 2

    if expression == "neutral":
        # Rounded rectangle eye
        draw.ellipse(
            [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
            fill=(0, 200, 255),
            outline=(0, 255, 255),
            width=2,
        )
        # Pupil
        pw, ph = w // 4, h // 4
        draw.ellipse(
            [cx - pw, cy - ph, cx + pw, cy + ph],
            fill=(0, 0, 0),
        )
        # Highlight
        hw, hh = w // 8, h // 8
        draw.ellipse(
            [cx - pw + hw, cy - ph + hh, cx - pw + hw * 3, cy - ph + hh * 3],
            fill=(255, 255, 255),
        )

    elif expression == "angry":
        # Narrowed, angled eyes
        draw.ellipse(
            [cx - half_w, cy - half_h // 2, cx + half_w, cy + half_h],
            fill=(255, 50, 0),
            outline=(255, 100, 0),
            width=2,
        )
        # Angry brow line
        draw.line(
            [cx - half_w - 5, cy - half_h - 5, cx + half_w + 5, cy - half_h // 2 + 5],
            fill=(255, 0, 0),
            width=3,
        )
        # Pupil (small, intense)
        pw, ph = w // 5, h // 5
        draw.ellipse(
            [cx - pw, cy - ph + 5, cx + pw, cy + ph + 5],
            fill=(0, 0, 0),
        )

    elif expression == "sleepy":
        # Half-closed eyes
        draw.ellipse(
            [cx - half_w, cy, cx + half_w, cy + half_h],
            fill=(100, 150, 200),
            outline=(80, 120, 180),
            width=2,
        )
        # Droopy eyelid
        draw.rectangle(
            [cx - half_w - 2, cy - half_h, cx + half_w + 2, cy + 2],
            fill=(0, 0, 0),
        )
        # Small pupil
        pw, ph = w // 5, h // 6
        draw.ellipse(
            [cx - pw, cy + 5, cx + pw, cy + ph + 10],
            fill=(0, 0, 0),
        )


def _render_eyes(expression="neutral"):
    """Render both eyes and return a PIL Image (WIDTH x HEIGHT)."""
    if not _HAS_PIL:
        return None

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    # Eye dimensions
    eye_w = 60
    eye_h = 50
    gap = 40  # gap between eyes

    cx_left = WIDTH // 2 - gap
    cx_right = WIDTH // 2 + gap
    cy = HEIGHT // 2

    _draw_eye(draw, cx_left, cy, eye_w, eye_h, expression)
    _draw_eye(draw, cx_right, cy, eye_w, eye_h, expression)

    return img


# ---------------------------------------------------------------------------
# Waveform rendering thread (BrutusVisualizer equivalent)
# ---------------------------------------------------------------------------

def _waveform_loop():
    """Daemon thread: render audio-reactive waveform on the LCD."""
    if not _HAS_PIL or _disp is None:
        return

    log.info("Waveform thread started")
    phase = 0.0
    smoothed_level = 0.0
    frame_t = 1.0 / _WV_FPS

    if _HAS_NUMPY:
        x_vals = np.arange(WIDTH, dtype=np.float32)

    while not _waveform_stop.is_set():
        try:
            start = time.time()

            # Phase scroll
            phase += _WV_PHASE_SPEED

            # Smoothed audio level
            with _waveform_level_lock:
                raw_level = _waveform_level
            smoothed_level = _WV_SMOOTH * smoothed_level + (1 - _WV_SMOOTH) * raw_level

            # Generate waveform
            img = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            cy = HEIGHT // 2

            if _HAS_NUMPY:
                base_wave = np.sin((x_vals / _WV_WAVELENGTH) + phase) * _WV_BASE_AMP

                if smoothed_level > 0.02:
                    noise_strength = _WV_GAIN_NOISE * smoothed_level ** 1.5
                    noise = np.random.normal(0.0, noise_strength, size=WIDTH)
                    wave = np.clip(base_wave + noise, -1.0, 1.0)
                else:
                    wave = base_wave

                # Draw base wave
                for x in range(WIDTH - 1):
                    y1 = int(cy - base_wave[x] * cy)
                    y2 = int(cy - base_wave[x + 1] * cy)
                    draw.line((x, y1, x + 1, y2), fill=_WV_BASE_COLOR)

                # Draw noise overlay when speaking
                if smoothed_level > 0.02:
                    for x in range(WIDTH - 1):
                        y1 = int(cy - wave[x] * cy)
                        y2 = int(cy - wave[x + 1] * cy)
                        draw.line((x, y1, x + 1, y2), fill=_WV_NOISE_COLOR)
            else:
                # Pure-math fallback (no numpy)
                for x in range(WIDTH - 1):
                    y1 = int(cy - math.sin((x / _WV_WAVELENGTH) + phase) * _WV_BASE_AMP * cy)
                    y2 = int(cy - math.sin(((x + 1) / _WV_WAVELENGTH) + phase) * _WV_BASE_AMP * cy)
                    draw.line((x, y1, x + 1, y2), fill=_WV_BASE_COLOR)

            # Send to display
            try:
                if not _waveform_stop.is_set() and _disp is not None:
                    _disp.ShowImage(img)
            except (OSError, AttributeError):
                break

            # Frame rate limiting
            elapsed = time.time() - start
            sleep_time = frame_t - elapsed
            if sleep_time > 0:
                _waveform_stop.wait(sleep_time)

        except Exception as e:
            log.warning("Waveform loop error: %s", e)
            _waveform_stop.wait(0.5)

    # Clear screen on exit
    try:
        if _disp is not None:
            _disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
    except Exception:
        pass

    log.info("Waveform thread stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init() -> bool:
    """Initialize the Waveshare LCD and clear the screen.

    Returns True if hardware is available, False otherwise (stub mode).
    """
    global _disp, _initialized

    if _initialized:
        log.warning("display.init() called twice")
        return _disp is not None

    if not _HAS_HARDWARE:
        log.warning("No display hardware — running in stub mode")
        _initialized = True
        return False

    try:
        _disp = LCD_1inch9.LCD_1inch9(rst=_RST_PIN, dc=_DC_PIN, bl=_BL_PIN)
        _disp.Init()

        # Set rotation
        try:
            _disp.set_rotation(_ROTATION)
        except AttributeError:
            try:
                _disp.rotate(_ROTATION)
            except Exception:
                pass

        # Clear screen and set backlight
        if _HAS_PIL:
            _disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
        _disp.bl_DutyCycle(100)

        _initialized = True
        log.info("Display initialized (%dx%d, rotation %d)", WIDTH, HEIGHT, _ROTATION)
        return True

    except Exception as e:
        log.error("Failed to initialize display: %s", e)
        _disp = None
        _initialized = True
        return False


def show_eyes(expression="neutral"):
    """Draw an eye expression on the display.

    Supported expressions: neutral, angry, sleepy.
    """
    if _disp is None or not _HAS_PIL:
        return

    # Stop waveform if running
    stop_waveform()

    img = _render_eyes(expression)
    if img is not None:
        try:
            _disp.ShowImage(img)
        except Exception as e:
            log.warning("Error showing eyes: %s", e)


def show_waveform(audio_level):
    """Update the audio-reactive waveform visualizer.

    audio_level: float 0.0 to 1.0 (RMS level from audio module).
    Call this repeatedly while speaking to drive the animation.
    The waveform thread handles rendering at ~20 FPS.
    """
    global _waveform_thread

    with _waveform_level_lock:
        _waveform_level = max(0.0, min(float(audio_level), 1.0))

    # Start waveform thread if not running
    if _waveform_thread is None or not _waveform_thread.is_alive():
        _waveform_stop.clear()
        _waveform_thread = threading.Thread(
            target=_waveform_loop, daemon=True, name="display-waveform"
        )
        _waveform_thread.start()


def stop_waveform():
    """Stop the waveform rendering thread."""
    global _waveform_thread

    if _waveform_thread is not None and _waveform_thread.is_alive():
        _waveform_stop.set()
        _waveform_thread.join(timeout=2)
        _waveform_thread = None


def show_text(text):
    """Show a text message on the display."""
    if _disp is None or not _HAS_PIL:
        return

    # Stop waveform if running
    stop_waveform()

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    # Use default font (no external font file needed)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Word-wrap text
    max_chars = WIDTH // 10  # approximate chars per line
    lines = []
    for line in text.split("\n"):
        while len(line) > max_chars:
            # Find last space before max_chars
            split_at = line[:max_chars].rfind(" ")
            if split_at == -1:
                split_at = max_chars
            lines.append(line[:split_at])
            line = line[split_at:].lstrip()
        lines.append(line)

    # Draw text centered vertically
    line_height = 20
    total_height = len(lines) * line_height
    y_start = max(5, (HEIGHT - total_height) // 2)

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        if y + line_height > HEIGHT:
            break
        draw.text((10, y), line, fill=(0, 255, 200), font=font)

    try:
        _disp.ShowImage(img)
    except Exception as e:
        log.warning("Error showing text: %s", e)


def cleanup():
    """Stop threads, clear the display, and turn off the backlight."""
    global _initialized, _disp

    if not _initialized:
        return

    log.info("Display cleanup starting...")

    # Stop waveform thread
    stop_waveform()

    # Clear display and turn off backlight
    if _disp is not None:
        try:
            if _HAS_PIL:
                _disp.ShowImage(Image.new("RGB", (WIDTH, HEIGHT), "black"))
            _disp.module_exit()
        except Exception as e:
            log.warning("Error during display cleanup: %s", e)

    _disp = None
    _initialized = False
    log.info("Display cleanup complete")
