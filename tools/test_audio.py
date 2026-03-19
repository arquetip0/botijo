#!/usr/bin/env python3
"""Test audio subsystem — init, listen once, echo back via TTS.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/test_audio.py
"""

import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_audio")


def main():
    import audio

    log.info("=== Audio Test ===")
    ok = audio.init()
    if not ok:
        log.warning("audio.init() returned False — ReSpeaker not found, continuing with degraded mode")
    else:
        log.info("Audio init OK")

    log.info("Speaking test phrase...")
    result = audio.speak("Sistema de audio inicializado. Di algo y te lo repito.")
    log.info("Speak result: interrupted=%s", result.interrupted)

    log.info("Listening for speech (up to 15 seconds)...")
    text = audio.listen()

    if text:
        log.info("Heard: %r", text)
        log.info("Echoing back...")
        result = audio.speak(f"Dijiste: {text}")
        log.info("Echo result: interrupted=%s", result.interrupted)
    else:
        log.warning("No speech detected")
        audio.speak("No escuché nada.")

    log.info("=== Test complete ===")
    audio.cleanup()


if __name__ == "__main__":
    main()
