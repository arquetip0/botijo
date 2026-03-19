#!/usr/bin/env python3
"""Reset all servos to neutral position.

Use this after a crash or when servos are stuck in odd positions.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/reset_servos.py
"""

import logging
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reset_servos")

# (label, channel, neutral_angle)
# Neutrals: eyes=90, eyelids=open position, tentacles=90
NEUTRAL_POSITIONS = [
    ("Eye LR        ch 0",  0,  90),
    ("Eye UD        ch 1",  1,  90),
    ("Eyelid TL     ch 2",  2,  50),   # open
    ("Eyelid BL     ch 3",  3, 155),   # open
    ("Eyelid TR     ch 4",  4, 135),   # open
    ("Eyelid BR     ch 5",  5,  25),   # open
    ("Tentacle Pit  ch 8",  8,  90),
    ("Tentacle L    ch15", 15,  90),
    ("Tentacle R    ch12", 12,  90),
]


def main():
    try:
        from adafruit_servokit import ServoKit
    except ImportError:
        log.error("adafruit_servokit not available — this tool only runs on RPi")
        sys.exit(1)

    from config import HARDWARE

    log.info("=== Servo Reset ===")

    try:
        kit = ServoKit(channels=16)
        log.info("ServoKit OK")
    except Exception as e:
        log.error("Failed to init ServoKit: %s", e)
        sys.exit(1)

    # Set pulse width ranges
    pw_min, pw_max = HARDWARE["servos"]["pulse_width_range"]
    tl_min, tl_max = HARDWARE["servos"]["tentacle_left_pulse"]
    tr_min, tr_max = HARDWARE["servos"]["tentacle_right_pulse"]

    for label, ch, neutral in NEUTRAL_POSITIONS:
        if ch == 15:
            kit.servo[ch].set_pulse_width_range(tl_min, tl_max)
        elif ch == 12:
            kit.servo[ch].set_pulse_width_range(tr_min, tr_max)
        else:
            kit.servo[ch].set_pulse_width_range(pw_min, pw_max)

    # Move to neutral positions
    for label, ch, neutral in NEUTRAL_POSITIONS:
        try:
            kit.servo[ch].angle = neutral
            log.info("  %-22s → %d°", label, neutral)
        except Exception as e:
            log.warning("  %-22s ERROR: %s", label, e)

    # Brief pause to let servos reach position
    time.sleep(0.5)

    # Release tentacles (no holding torque when idle)
    for label, ch, neutral in NEUTRAL_POSITIONS:
        if ch in (15, 12, 8):
            try:
                kit.servo[ch].angle = None
                log.info("  %-22s released", label)
            except Exception:
                pass

    log.info("=== Reset complete ===")


if __name__ == "__main__":
    main()
