"""GPIO button input.

Manages 4 GPIO buttons with debounce using gpiozero.  Buttons use
pull_up=False (the hardware modules generate HIGH when pressed).

Button GPIOs come from config.HARDWARE["buttons"].
Graceful degradation: if gpiozero is unavailable (e.g. on a dev laptop),
init() returns False and all functions become no-ops.
"""

import logging
import os

from config import HARDWARE

log = logging.getLogger("botijo.buttons")

# ---------------------------------------------------------------------------
# Hardware abstraction
# ---------------------------------------------------------------------------

# gpiozero needs lgpio on RPi 5 — set pin factory before import
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")

try:
    from gpiozero import Button
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False
    log.info("gpiozero not available — buttons module will run in stub mode")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = HARDWARE["buttons"]
_DEBOUNCE = 0.02  # 20ms debounce

# Button names and their GPIO pins
_BUTTON_MAP = {
    "btn1": _cfg.get("btn1_gpio", 5),
    "btn2": _cfg.get("btn2_gpio", 6),
    "btn3": _cfg.get("btn3_gpio", 13),
    "btn4": _cfg.get("btn4_gpio", 12),
}

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_buttons = {}  # name -> Button instance
_initialized = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(callbacks=None) -> bool:
    """Initialize GPIO buttons with optional callbacks.

    callbacks: dict mapping button name to callable, e.g.
        {"btn1": my_handler, "btn3": other_handler}
    Each callback is called with no arguments when the button is pressed.

    Returns True if hardware is available, False otherwise (stub mode).
    """
    global _initialized

    if _initialized:
        log.warning("buttons.init() called twice")
        return bool(_buttons)

    if not _HAS_HARDWARE:
        log.warning("No button hardware — running in stub mode")
        _initialized = True
        return False

    callbacks = callbacks or {}

    try:
        for name, gpio in _BUTTON_MAP.items():
            btn = Button(gpio, pull_up=False, bounce_time=_DEBOUNCE)

            # Attach callback if provided
            if name in callbacks:
                btn.when_pressed = callbacks[name]
                log.debug("Button %s (GPIO %d): callback attached", name, gpio)
            else:
                btn.when_pressed = lambda n=name: log.debug("Button %s pressed", n)

            _buttons[name] = btn
            log.debug("Button %s initialized on GPIO %d", name, gpio)

        _initialized = True
        log.info("Buttons initialized: %s", ", ".join(
            f"{name}=GPIO{gpio}" for name, gpio in _BUTTON_MAP.items()
        ))
        return True

    except Exception as e:
        log.error("Failed to initialize buttons: %s", e)
        _buttons.clear()
        _initialized = True
        return False


def is_pressed(name) -> bool:
    """Check if a button is currently pressed.

    name: one of "btn1", "btn2", "btn3", "btn4"
    Returns False if button doesn't exist or hardware is unavailable.
    """
    btn = _buttons.get(name)
    if btn is None:
        return False
    return btn.is_pressed


def set_callback(name, callback):
    """Set or replace the press callback for a button.

    name: one of "btn1", "btn2", "btn3", "btn4"
    callback: callable (no arguments) or None to remove
    """
    btn = _buttons.get(name)
    if btn is None:
        log.debug("Button %s not available, ignoring set_callback", name)
        return
    btn.when_pressed = callback


def cleanup():
    """Close all button resources."""
    global _initialized

    if not _initialized:
        return

    log.info("Buttons cleanup starting...")

    for name, btn in _buttons.items():
        try:
            btn.close()
        except Exception as e:
            log.debug("Error closing button %s: %s", name, e)

    _buttons.clear()
    _initialized = False
    log.info("Buttons cleanup complete")
