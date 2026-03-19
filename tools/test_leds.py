#!/usr/bin/env python3
"""Test LED animations — cycle through all modes.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/test_leds.py
"""

import logging
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_leds")

MODES = [
    ("steampunk", 4.0, "Steampunk copper/brass pulsation (idle mode)"),
    ("listening", 3.0, "Purple quiet-glow pulsation (listening mode)"),
    ("speaking",  3.0, "Red Knight-Rider sweep (speaking mode)"),
    ("off",       1.0, "All LEDs off"),
]


def main():
    import leds

    log.info("=== LED Test ===")
    ok = leds.init()
    if not ok:
        log.warning("leds.init() returned False — no NeoPixel hardware found")
        log.warning("On RPi: ensure board/neopixel are installed and GPIO18 is free")
        sys.exit(1)

    log.info("LED init OK (13 NeoPixels on GPIO18)")

    for mode, duration, description in MODES:
        log.info("Mode: %-12s — %s (%.0fs)", mode, description, duration)
        leds.set_mode(mode)
        time.sleep(duration)

    log.info("=== Test complete ===")
    leds.cleanup()


if __name__ == "__main__":
    main()
