"""Botijo — Androide Conversacional.

Entry point. Parses args, loads config, initializes modules, runs main loop.
Usage: PYTHONPATH=src:vendor python src/main.py [--mode botijo|botija|barbacoa]
"""

import argparse
import signal
import sys
import logging

from config import get_personality, HARDWARE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("botijo")

# Module stubs — will be replaced as modules are implemented
_modules = []
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


def main():
    parser = argparse.ArgumentParser(description="Botijo — Androide Conversacional")
    parser.add_argument("--mode", default="botijo", choices=["botijo", "botija", "barbacoa"],
                        help="Personality mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    personality = get_personality(args.mode)
    log.info("Botijo starting — mode: %s, llm: %s", personality["name"], personality["llm"])
    log.info("Hardware config loaded: %d servo channels, %d LEDs, %d buttons",
             16, HARDWARE["leds"]["count"], len([k for k in HARDWARE["buttons"] if k.startswith("btn")]))

    # TODO: Initialize modules here (audio, brain, servos, leds, vision, display, buttons)
    # TODO: Main loop here

    log.info("Botijo ready — no modules loaded yet. Mode: %s", args.mode)
    log.info("Press Ctrl+C to exit")

    try:
        signal.pause()  # Wait for signal (Unix only)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
