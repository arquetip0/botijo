"""Servo control module — eyes, eyelids, tentacles (ears).

Drives 11 servos via PCA9685 (I2C) through adafruit_servokit.
Two daemon threads provide idle behavior:
  - _eye_idle_loop: blinks, saccades, micro-movements, breathing eyelids
  - _tentacle_loop: random mirrored left/right ear movements

All hardware constants come from config.HARDWARE["servos"] and config.HARDWARE["eyes"].
Graceful degradation: if adafruit_servokit is unavailable (e.g. on MacBook), init()
returns False and all functions become no-ops.
"""

import logging
import math
import random
import threading
import time

from config import HARDWARE

log = logging.getLogger("servos")

# ---------------------------------------------------------------------------
# Hardware abstraction
# ---------------------------------------------------------------------------
try:
    from adafruit_servokit import ServoKit
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False
    log.info("adafruit_servokit not available — servo module will run in stub mode")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_kit = None                     # ServoKit instance (or None)
_shutdown = threading.Event()   # signal threads to stop
_quiet = threading.Event()      # set = quiet mode ON (pause mechanical movements)
_initialized = False

_eye_thread = None
_tentacle_thread = None

# Current eye position (used by idle loop for smooth transitions)
_prev_lr = 90.0
_prev_ud = 90.0
_eye_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Config shortcuts (populated in init())
# ---------------------------------------------------------------------------
_SERVOS = {}   # HARDWARE["servos"]
_EYES = {}     # HARDWARE["eyes"]

# Eyelid channel map: key -> (channel, open_angle, closed_angle)
_EYELIDS = {}

# Servo channel numbers
_CH_LR = 0
_CH_UD = 1
_CH_TENT_L = 15
_CH_TENT_R = 12

# Ranges
_LR_MIN = 40
_LR_MAX = 140
_UD_MIN = 30
_UD_MAX = 150


def _load_config():
    """Populate module-level config from HARDWARE dict."""
    global _SERVOS, _EYES, _EYELIDS
    global _CH_LR, _CH_UD, _CH_TENT_L, _CH_TENT_R
    global _LR_MIN, _LR_MAX, _UD_MIN, _UD_MAX

    _SERVOS = HARDWARE["servos"]
    _EYES = HARDWARE["eyes"]

    _CH_LR = _SERVOS["eye_lr_channel"]
    _CH_UD = _SERVOS["eye_ud_channel"]
    _CH_TENT_L = _SERVOS["tentacle_left_channel"]
    _CH_TENT_R = _SERVOS["tentacle_right_channel"]

    _LR_MIN, _LR_MAX = _SERVOS["lr_range"]
    _UD_MIN, _UD_MAX = _SERVOS["ud_range"]

    # Build eyelid map from config
    for name, info in _SERVOS["eyelids"].items():
        _EYELIDS[name] = (info["channel"], info["open"], info["closed"])


# ---------------------------------------------------------------------------
# Low-level servo helpers
# ---------------------------------------------------------------------------

def _set_angle(channel, angle):
    """Set servo angle, silently ignoring errors if kit is None."""
    if _kit is None:
        return
    try:
        _kit.servo[channel].angle = angle
    except Exception as e:
        log.debug("servo[%d] error: %s", channel, e)


def _release(channel):
    """Release servo (no holding torque)."""
    if _kit is None:
        return
    try:
        _kit.servo[channel].angle = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Eye helpers
# ---------------------------------------------------------------------------

def _init_eye_servos():
    """Set pulse width ranges and center the eye servos."""
    global _prev_lr, _prev_ud
    if _kit is None:
        return

    pw_min, pw_max = _SERVOS["pulse_width_range"]
    _kit.servo[_CH_LR].set_pulse_width_range(pw_min, pw_max)
    _kit.servo[_CH_UD].set_pulse_width_range(pw_min, pw_max)

    center_lr = (_LR_MIN + _LR_MAX) // 2
    center_ud = (_UD_MIN + _UD_MAX) // 2
    _set_angle(_CH_LR, center_lr)
    _set_angle(_CH_UD, center_ud)

    _prev_lr = float(center_lr)
    _prev_ud = float(center_ud)
    log.debug("Eye servos centered: LR=%d UD=%d", center_lr, center_ud)


def _open_eyelids():
    """Move eyelids to their open positions."""
    if _kit is None:
        return
    pw_min, pw_max = _SERVOS["pulse_width_range"]
    for name, (ch, open_angle, _closed) in _EYELIDS.items():
        _kit.servo[ch].set_pulse_width_range(pw_min, pw_max)
        _set_angle(ch, open_angle)
    log.debug("Eyelids opened")


def _close_eyelids():
    """Move all eyelids to closed position (90 degrees)."""
    if _kit is None:
        return
    for name, (ch, _open, closed) in _EYELIDS.items():
        _set_angle(ch, closed)
    log.debug("Eyelids closed")


def _breathing_adjustment():
    """Compute a subtle eyelid offset based on a sine breathing cycle."""
    if not _EYES.get("breathing_enabled", True):
        return 0.0
    cycle = _EYES.get("breathing_cycle", 4.0)
    intensity = _EYES.get("breathing_intensity", 0.15)
    phase = (time.time() % cycle) / cycle
    return math.sin(phase * 2 * math.pi) * intensity


def _update_eyelids_breathing():
    """Adjust eyelid positions with breathing pattern."""
    if _kit is None:
        return
    offset = _breathing_adjustment()
    for name, (ch, open_angle, _closed) in _EYELIDS.items():
        adjusted = open_angle + offset * 5  # scale for visibility
        adjusted = max(10, min(170, adjusted))
        _set_angle(ch, adjusted)


def _do_blink():
    """Execute one blink: close eyelids briefly, reopen with breathing."""
    if _kit is None:
        return
    duration = _EYES.get("blink_duration", 0.12)

    # Close
    for name, (ch, _open, closed) in _EYELIDS.items():
        _set_angle(ch, closed)
    time.sleep(duration)

    # Open with breathing
    offset = _breathing_adjustment()
    for name, (ch, open_angle, _closed) in _EYELIDS.items():
        _set_angle(ch, open_angle + offset * 5)

    # Possible double blink
    if random.random() < _EYES.get("double_blink_probability", 0.3):
        time.sleep(0.1)
        for name, (ch, _open, closed) in _EYELIDS.items():
            _set_angle(ch, closed)
        time.sleep(duration * 0.8)
        offset = _breathing_adjustment()
        for name, (ch, open_angle, _closed) in _EYELIDS.items():
            _set_angle(ch, open_angle + offset * 5)


def _do_squint():
    """Squint (partially close eyelids) then gradually reopen."""
    if _kit is None:
        return
    intensity = _EYES.get("squint_intensity", 0.6)
    duration = _EYES.get("squint_duration", 1.5)

    # Partially close
    for name, (ch, open_angle, _closed) in _EYELIDS.items():
        squint_angle = open_angle + (90 - open_angle) * intensity
        _set_angle(ch, squint_angle)
    time.sleep(duration)

    # Gradual reopen
    steps = 5
    for step in range(steps):
        for name, (ch, open_angle, _closed) in _EYELIDS.items():
            squint_angle = open_angle + (90 - open_angle) * intensity
            progress = (step + 1) / steps
            current = squint_angle + (open_angle - squint_angle) * progress
            _set_angle(ch, current + _breathing_adjustment() * 5)
        time.sleep(0.1)


def _do_look_random():
    """Execute a random saccade: smooth movement to a random point, hold with micro-movements."""
    global _prev_lr, _prev_ud
    if _kit is None:
        return

    micro_prob = _EYES.get("micro_movement_probability", 0.15)
    micro_range = _EYES.get("micro_movement_range", 3)

    target_lr = random.uniform(_LR_MIN, _LR_MAX)
    target_ud = random.uniform(_UD_MIN, _UD_MAX)

    # Smooth movement in 8 steps
    with _eye_lock:
        start_lr, start_ud = _prev_lr, _prev_ud
    steps = 8
    for step in range(steps):
        if _shutdown.is_set():
            return
        progress = (step + 1) / steps
        cur_lr = start_lr + (target_lr - start_lr) * progress
        cur_ud = start_ud + (target_ud - start_ud) * progress
        _set_angle(_CH_LR, cur_lr)
        _set_angle(_CH_UD, cur_ud)
        time.sleep(0.05)

    with _eye_lock:
        _prev_lr = target_lr
        _prev_ud = target_ud

    # Hold with micro-movements
    hold_duration = _EYES.get("random_look_duration", 0.1)
    hold_steps = max(1, int(hold_duration / 0.1))
    for _ in range(hold_steps):
        if _shutdown.is_set():
            return
        mlr = target_lr + (random.uniform(-1, 1) if random.random() < micro_prob else 0)
        mud = target_ud + (random.uniform(-1, 1) if random.random() < micro_prob else 0)
        _set_angle(_CH_LR, mlr)
        _set_angle(_CH_UD, mud)
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Tentacle helpers
# ---------------------------------------------------------------------------

def _init_tentacles():
    """Set pulse width ranges and center tentacle servos."""
    if _kit is None:
        return
    tl_min, tl_max = _SERVOS["tentacle_left_pulse"]
    tr_min, tr_max = _SERVOS["tentacle_right_pulse"]
    _kit.servo[_CH_TENT_L].set_pulse_width_range(tl_min, tl_max)
    _kit.servo[_CH_TENT_R].set_pulse_width_range(tr_min, tr_max)
    _set_angle(_CH_TENT_L, 90)
    _set_angle(_CH_TENT_R, 90)
    log.debug("Tentacle servos centered")


def _stop_tentacles():
    """Center tentacles and release them."""
    _set_angle(_CH_TENT_L, 90)
    _set_angle(_CH_TENT_R, 90)
    time.sleep(0.3)
    _release(_CH_TENT_L)
    _release(_CH_TENT_R)


# ---------------------------------------------------------------------------
# Daemon threads
# ---------------------------------------------------------------------------

def _eye_idle_loop():
    """Idle eye behavior: blinks, saccades, micro-movements, breathing.

    Runs until _shutdown is set. Pauses all movement when _quiet is set
    (blinks, squints, and random looks are suppressed).
    """
    log.info("Eye idle thread started")
    last_breathing = time.time()

    while not _shutdown.is_set():
        try:
            # Breathing update every ~200ms regardless of quiet mode
            if time.time() - last_breathing > 0.2:
                _update_eyelids_breathing()
                last_breathing = time.time()

            if not _quiet.is_set():
                # Random behaviors
                r = random.random()
                if r < _EYES.get("blink_probability", 0.01):
                    _do_blink()
                elif r < _EYES.get("blink_probability", 0.01) + _EYES.get("squint_probability", 0.02):
                    _do_squint()
                elif r < (_EYES.get("blink_probability", 0.01)
                          + _EYES.get("squint_probability", 0.02)
                          + _EYES.get("random_look_probability", 0.01)):
                    _do_look_random()

            # Main loop tick ~50ms
            _shutdown.wait(0.05)

        except Exception as e:
            log.warning("Eye idle error: %s", e)
            _shutdown.wait(0.5)

    log.info("Eye idle thread stopped")


def _tentacle_loop():
    """Random tentacle movement with mirrored left/right.

    Runs until _shutdown is set. Pauses when _quiet is set.
    """
    log.info("Tentacle thread started")
    angle_left = 90
    angle_right = 90

    while not _shutdown.is_set():
        try:
            # Wait while in quiet mode (check every 100ms)
            if _quiet.is_set():
                _shutdown.wait(0.1)
                continue

            # Pick a target for left ear (extremes: 0-30 or 150-180)
            target_left = random.choice([
                random.randint(0, 30),
                random.randint(150, 180),
            ])
            # Mirror: left 0 (up) -> right 180 (up), left 180 (down) -> right 0 (down)
            target_right = 180 - target_left

            step = random.randint(3, 7)

            # Build step ranges
            if target_left > angle_left:
                rng_left = range(angle_left, target_left + 1, step)
            else:
                rng_left = range(angle_left, target_left - 1, -step)

            if target_right > angle_right:
                rng_right = range(angle_right, target_right + 1, step)
            else:
                rng_right = range(angle_right, target_right - 1, -step)

            # Move both in parallel steps
            actual_left = angle_left
            actual_right = angle_right
            for a_left, a_right in zip(rng_left, rng_right):
                if _shutdown.is_set() or _quiet.is_set():
                    break
                _set_angle(_CH_TENT_L, a_left)
                _set_angle(_CH_TENT_R, a_right)
                actual_left = a_left
                actual_right = a_right
                time.sleep(random.uniform(0.005, 0.015))

            angle_left = actual_left
            angle_right = actual_right

            # Random pause 3-6s (check exit every 100ms)
            pause = random.uniform(3, 6)
            for _ in range(int(pause * 10)):
                if _shutdown.is_set() or _quiet.is_set():
                    break
                time.sleep(0.1)

        except Exception as e:
            log.warning("Tentacle loop error: %s", e)
            _shutdown.wait(1)

    log.info("Tentacle thread stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init() -> bool:
    """Initialize PCA9685, set pulse ranges, center eyes, open eyelids, start idle threads.

    Returns True if hardware is available and initialized, False otherwise
    (stub mode — module is still safe to call).
    """
    global _kit, _initialized, _eye_thread, _tentacle_thread

    if _initialized:
        log.warning("servos.init() called twice")
        return _kit is not None

    _load_config()

    if not _HAS_HARDWARE:
        log.warning("No servo hardware — running in stub mode")
        _initialized = True
        return False

    # Try to create ServoKit
    try:
        _kit = ServoKit(channels=16)
        log.info("PCA9685 ServoKit initialized (16 channels)")
    except Exception as e:
        log.error("Failed to initialize ServoKit: %s", e)
        _kit = None
        _initialized = True
        return False

    # Set up eye servos and eyelids
    _init_eye_servos()
    _open_eyelids()

    # Set up tentacle servos
    _init_tentacles()

    # Start idle threads
    _shutdown.clear()

    _eye_thread = threading.Thread(target=_eye_idle_loop, daemon=True, name="eye-idle")
    _eye_thread.start()

    _tentacle_thread = threading.Thread(target=_tentacle_loop, daemon=True, name="tentacle")
    _tentacle_thread.start()

    _initialized = True
    log.info("Servo module initialized: eyes + eyelids + tentacles active")
    return True


def look_at(x, y):
    """Move eyes to normalized coordinates (0.0-1.0).

    x: 0.0 = full left, 1.0 = full right
    y: 0.0 = full up, 1.0 = full down
    """
    global _prev_lr, _prev_ud
    if _kit is None:
        return

    # Map normalized coords to servo ranges (inverted — camera image is mirrored)
    lr = _LR_MAX - x * (_LR_MAX - _LR_MIN)
    ud = _UD_MAX - y * (_UD_MAX - _UD_MIN)

    # Clamp
    lr = max(_LR_MIN, min(_LR_MAX, lr))
    ud = max(_UD_MIN, min(_UD_MAX, ud))

    _set_angle(_CH_LR, lr)
    _set_angle(_CH_UD, ud)

    with _eye_lock:
        _prev_lr = lr
        _prev_ud = ud


def blink():
    """Close eyelids briefly then reopen."""
    _do_blink()


def set_quiet_mode(on):
    """Pause (True) or resume (False) all mechanical movements.

    When quiet mode is on:
    - Eye idle thread suppresses blinks/saccades/squints (breathing continues)
    - Tentacle thread pauses movement
    """
    if on:
        if not _quiet.is_set():
            _quiet.set()
            log.debug("Quiet mode ON — mechanical movements paused")
    else:
        if _quiet.is_set():
            _quiet.clear()
            log.debug("Quiet mode OFF — movements resumed")


def cleanup():
    """Stop threads, return servos to neutral, release hardware."""
    global _initialized, _kit

    if not _initialized:
        return

    log.info("Servo cleanup starting...")

    # Signal threads to stop
    _shutdown.set()

    # Wait for threads
    for t in (_eye_thread, _tentacle_thread):
        if t is not None and t.is_alive():
            t.join(timeout=2)

    # Return to neutral positions
    if _kit is not None:
        # Center eyes
        center_lr = (_LR_MIN + _LR_MAX) // 2
        center_ud = (_UD_MIN + _UD_MAX) // 2
        _set_angle(_CH_LR, center_lr)
        _set_angle(_CH_UD, center_ud)

        # Close eyelids
        _close_eyelids()

        # Center and release tentacles
        _stop_tentacles()

    _kit = None
    _initialized = False
    log.info("Servo cleanup complete")
