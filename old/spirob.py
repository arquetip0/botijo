#!/usr/bin/env python3
"""
Spirob – Control del tentáculo (v4‑d)
• Giro lateral (izq/der) reforzado → scale 1.4 en cables 9 y 10
• Pitch (arriba/abajo) suavizado → scale 0.8 en cable 8
• Neutral = 0 / 90 / 90 °  (servo 8 suelto)
"""

from __future__ import annotations
import json, time, math
from pathlib import Path
from adafruit_servokit import ServoKit

# ───────────────────────── CALIBRACIÓN ──────────────────────────
CALIB_PATH = Path(__file__).with_name("spirob_calib.json")
DEFAULT_CALIB = {
    "offset_c": 45.0,       # pretensión neutra
    "k_max": 120.0,         # curvatura base máxima
    "step_ms": 15,
    "servos": {
        "8":  {"tight": 0,   "slack": 180, "pwm_min": 310, "pwm_max": 2500,
                "alpha": 90,  "scale": 0.8},   # ↓ pitch suavizado
        "9":  {"tight": 20,  "slack": 180, "pwm_min": 420, "pwm_max": 2700,
                "alpha": 210, "scale": 1.4},   # ← giro potenciado
        "10": {"tight": 180, "slack": 0,   "pwm_min": 500, "pwm_max": 2700,
                "alpha": 330, "scale": 1.4}    # → giro potenciado (nuevo tight 180)    # → giro potenciado
    }
}

def _load(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        path.write_text(json.dumps(DEFAULT_CALIB, indent=4))
        return DEFAULT_CALIB

cfg        = _load(CALIB_PATH)
O_C        = cfg["offset_c"]
K_MAX      = cfg["k_max"]
STEP_MS    = cfg["step_ms"]
SERVO_IDS  = tuple(int(k) for k in cfg["servos"])
SERVOS     = {int(k): v for k, v in cfg["servos"].items()}

# ───────────────────────── HARDWARE ────────────────────────────
kit = ServoKit(channels=16)
for sid, s in SERVOS.items():
    kit.servo[sid].set_pulse_width_range(s["pwm_min"], s["pwm_max"])

# ───────────────────── UTILIDADES BÁSICAS ─────────────────────

def angle_to_T(sid: int, ang: float) -> float:
    t, s = SERVOS[sid]["tight"], SERVOS[sid]["slack"]
    return (s - ang) / (s - t)

def T_to_angle(sid: int, T: float) -> float:
    T = max(0.0, min(1.0, T))
    t, s = SERVOS[sid]["tight"], SERVOS[sid]["slack"]
    return s - T * (s - t)

def get_T():
    return {sid: angle_to_T(sid, kit.servo[sid].angle or SERVOS[sid]["slack"])
            for sid in SERVO_IDS}

# ─────────────────── INTERPOLACIÓN SUAVE ──────────────────────
STEPS_PER_SEC = 1000 // STEP_MS

def set_T(target: dict[int,float], duration=0.4, normalise=True):
    start = get_T()
    n = max(1, int(duration * STEPS_PER_SEC))
    for i in range(1, n+1):
        mix = i/n
        step = {sid: (1-mix)*start[sid] + mix*target[sid] for sid in SERVO_IDS}
        if normalise:
            excess = (sum(step.values()) - 1.5)/3.0
            step = {sid: max(0, min(1, t - excess)) for sid, t in step.items()}
        for sid, T in step.items():
            kit.servo[sid].angle = T_to_angle(sid, T)
        time.sleep(STEP_MS/1000)

# ────────────────── φ/amount → tensiones objetivo ─────────────

def _contractions(phi: float, amount: float):
    k = amount * K_MAX
    T = {}
    for sid in SERVO_IDS:
        diff = math.radians(phi - SERVOS[sid]["alpha"])
        c_deg = O_C + SERVOS[sid]["scale"]*k*math.cos(diff)
        T[sid] = (c_deg - O_C)/K_MAX + 0.5
        T[sid] = max(0, min(1, T[sid]))
    excess = (sum(T.values()) - 1.5)/3.0
    return {sid: max(0, min(1, t - excess)) for sid, t in T.items()}

def move_to_bend(phi: float, amount: float, duration=0.5):
    set_T(_contractions(phi, amount), duration)

# ───────────────────── NEUTRO 0 / 90 / 90 ─────────────────────
NEUTRAL_OVERRIDE = {8: 0, 9: 90, 10: 90}

def neutral(duration=0.6):
    target = {sid: angle_to_T(sid, NEUTRAL_OVERRIDE[sid]) for sid in SERVO_IDS}
    set_T(target, duration, normalise=False)

# ───────────────────── PEQUEÑA DEMO ───────────────────────────
if __name__ == "__main__":
    try:
        neutral(1.0)
        time.sleep(1)
        print("→ izquierda fuerte (φ=210, amt=0.8)")
        move_to_bend(210, 0.8, 0.5)
        time.sleep(1)
        print("→ derecha fuerte (φ=330, amt=0.8)")
        move_to_bend(330, 0.8, 0.5)
        time.sleep(1)
        print("→ sube leve (φ=90, amt=0.3)")
        move_to_bend(90, 0.3, 0.4)
        time.sleep(1)
        neutral()
    except KeyboardInterrupt:
        neutral(0.6)
