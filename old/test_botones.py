#!/usr/bin/env python3
# Prueba de botones S-V-G (HIGH al pulsar) en GPIO5,6,13,12
# Si usas pulsadores “desnudos”, lee la nota al final.

import os
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")  # estable si falla la autodetección

from gpiozero import Button
from signal import pause
from datetime import datetime

BTN_PINS = {
    "BTN1": 5,   # pin físico 29
    "BTN2": 6,   # pin físico 31
    "BTN3": 13,  # pin físico 33
    "BTN4": 12,  # pin físico 32
}

def log(msg):
    t = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{t} {msg}", flush=True)

buttons = {}
for name, pin in BTN_PINS.items():
    # Módulos S-V-G: generan HIGH al pulsar → pull_up=False
    b = Button(pin, pull_up=False, bounce_time=0.02)  # 20 ms de antirrebote
    b.when_pressed = (lambda n: (lambda: log(f"{n} PRESIONADO")))(name)
    b.when_released = (lambda n: (lambda: log(f"{n} LIBERADO")))(name)
    buttons[name] = b

log("Listo. Pulsa los botones… (Ctrl+C para salir)")
pause()