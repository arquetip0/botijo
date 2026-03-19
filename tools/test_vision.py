#!/usr/bin/env python3
"""Test vision subsystem — print detected faces for 10 seconds.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/test_vision.py
"""

import logging
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_vision")

DURATION = 10  # seconds to run detection loop


def main():
    import vision

    log.info("=== Vision Test ===")
    log.info("Initializing IMX500 camera (loads AI firmware, takes ~5s)...")

    ok = vision.init()
    if not ok:
        log.warning("vision.init() returned False — IMX500/picamera2 not available")
        log.warning("On RPi: ensure picamera2 is installed and camera cable is connected")
        sys.exit(1)

    log.info("Vision init OK — running for %d seconds", DURATION)
    log.info("Point camera at faces to test detection")
    log.info("")

    start = time.time()
    last_print = 0.0
    total_detections = 0

    while time.time() - start < DURATION:
        faces = vision.get_faces()
        now = time.time()
        elapsed = now - start

        if faces:
            total_detections += 1
            for i, face in enumerate(faces):
                log.info(
                    "[%.1fs] Face %d: center=(%.2f, %.2f) size=(%.2f x %.2f) confidence=%.2f",
                    elapsed, i + 1, face.x, face.y, face.w, face.h, face.confidence,
                )
        elif now - last_print > 2.0:
            log.info("[%.1fs] No faces detected", elapsed)
            last_print = now

        time.sleep(0.1)

    log.info("")
    log.info("=== Test complete — %d detection frames in %ds ===", total_detections, DURATION)
    vision.cleanup()


if __name__ == "__main__":
    main()
