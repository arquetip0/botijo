#!/usr/bin/env python3
"""Interactive servo calibration — move each servo to test its range.

Usage:
    cd ~/botijo && PYTHONPATH=src:vendor python tools/calibrate_servos.py

For each servo, enter angles to test (0-180).  Press Enter to skip,
'q' to quit, 'n' for next servo.
"""

import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("calibrate_servos")


# Servo definitions: (label, channel, min_angle, max_angle, neutral)
# Channels and ranges from config/hardware.json
SERVOS = [
    # Eye movement
    ("Eye LR       (ch 0)",  0,  40, 140,  90),
    ("Eye UD       (ch 1)",  1,  30, 150,  90),
    # Eyelids
    ("Eyelid TL    (ch 2)",  2,  50,  90,  50),   # top-left:    open=50, closed=90
    ("Eyelid BL    (ch 3)",  3,  90, 155, 155),   # bottom-left: open=155, closed=90
    ("Eyelid TR    (ch 4)",  4,  90, 135, 135),   # top-right:   open=135, closed=90
    ("Eyelid BR    (ch 5)",  5,  25,  90,  25),   # bottom-right:open=25, closed=90
    # Tentacle pitch (ch 8)
    ("Tentacle Pit (ch 8)",  8,   0, 180,  90),
    # Tentacles (ears)
    ("Tentacle L   (ch15)", 15,   0, 180,  90),
    ("Tentacle R   (ch12)", 12,   0, 180,  90),
]


def calibrate_one(kit, label, channel, min_a, max_a, neutral):
    print(f"\n--- {label} ---")
    print(f"  Range: {min_a}–{max_a}°   Neutral: {neutral}°")
    print(f"  Enter angle ({min_a}-{max_a}), 'n' next, 'q' quit")

    # Start at neutral
    try:
        kit.servo[channel].angle = neutral
        print(f"  → Set to neutral ({neutral}°)")
    except Exception as e:
        print(f"  ERROR setting neutral: {e}")

    while True:
        try:
            raw = input(f"  [{label.split()[0]}] angle> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"

        if raw == "q":
            return "quit"
        if raw in ("n", ""):
            return "next"

        try:
            angle = float(raw)
        except ValueError:
            print("  Invalid — enter a number")
            continue

        if angle < 0 or angle > 180:
            print("  Out of range (0-180)")
            continue

        try:
            kit.servo[channel].angle = angle
            print(f"  → Set to {angle}°")
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    # Import hardware only — don't use the servos module (its threads would interfere)
    try:
        from adafruit_servokit import ServoKit
    except ImportError:
        log.error("adafruit_servokit not available — this tool only runs on RPi")
        sys.exit(1)

    from config import HARDWARE

    log.info("=== Servo Calibration ===")
    log.info("Initializing PCA9685 ServoKit...")

    try:
        kit = ServoKit(channels=16)
        log.info("ServoKit OK")
    except Exception as e:
        log.error("Failed to init ServoKit: %s", e)
        sys.exit(1)

    # Set pulse width ranges from config
    pw_min, pw_max = HARDWARE["servos"]["pulse_width_range"]
    tl_min, tl_max = HARDWARE["servos"]["tentacle_left_pulse"]
    tr_min, tr_max = HARDWARE["servos"]["tentacle_right_pulse"]

    for label, ch, min_a, max_a, neutral in SERVOS:
        if ch in (15,):
            kit.servo[ch].set_pulse_width_range(tl_min, tl_max)
        elif ch in (12,):
            kit.servo[ch].set_pulse_width_range(tr_min, tr_max)
        else:
            kit.servo[ch].set_pulse_width_range(pw_min, pw_max)

    print("\nCalibration starting. For each servo:")
    print("  - Type an angle to move it")
    print("  - Press Enter or 'n' to move to next servo")
    print("  - Type 'q' to quit\n")

    for label, channel, min_a, max_a, neutral in SERVOS:
        action = calibrate_one(kit, label, channel, min_a, max_a, neutral)
        if action == "quit":
            break

    # Center everything on exit
    log.info("Centering all servos...")
    for label, channel, min_a, max_a, neutral in SERVOS:
        try:
            kit.servo[channel].angle = neutral
        except Exception:
            pass

    log.info("=== Calibration complete ===")


if __name__ == "__main__":
    main()
