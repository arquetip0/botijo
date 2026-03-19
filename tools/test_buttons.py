#!/usr/bin/env python3
"""Test GPIO buttons — print a message when each button is pressed.

Press Ctrl-C to exit.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/test_buttons.py
"""

import logging
import signal
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_buttons")

# GPIO pins from hardware.json: btn1=5, btn2=6, btn3=13, btn4=12
_BUTTON_LABELS = {
    "btn1": "GPIO 5  — top-left",
    "btn2": "GPIO 6  — top-right",
    "btn3": "GPIO 13 — bottom-left",
    "btn4": "GPIO 12 — bottom-right",
}

_press_count = {}


def make_callback(name):
    def cb():
        _press_count[name] = _press_count.get(name, 0) + 1
        label = _BUTTON_LABELS.get(name, name)
        log.info("PRESSED  %s  (total: %d)", label, _press_count[name])
    return cb


def main():
    import buttons

    log.info("=== Button Test ===")

    callbacks = {name: make_callback(name) for name in _BUTTON_LABELS}
    ok = buttons.init(callbacks=callbacks)

    if not ok:
        log.warning("buttons.init() returned False — gpiozero not available or no GPIO hardware")
        log.warning("On RPi: ensure gpiozero and lgpio are installed")
        sys.exit(1)

    log.info("Button init OK")
    for name, label in _BUTTON_LABELS.items():
        log.info("  %s  %s", name, label)

    log.info("Waiting for button presses — Ctrl-C to exit...")

    def _handle_sigint(sig, frame):
        log.info("\n=== Test complete ===")
        log.info("Total presses: %s", dict(_press_count))
        buttons.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
