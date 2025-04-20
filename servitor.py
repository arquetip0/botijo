"""
Script ORO: Control de ojos animatrónicos para Raspberry Pi 5 con PCA9685.
yupi
Calibrado con ServoKit y verificado manualmente por el Amo para una respuesta óptima en cada servo.
"""

import time
import math
import random
import threading
from evdev import InputDevice, ecodes, list_devices
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# Establecer el rango de pulso para todos los servos basado en la calibración
for i in range(6):
    kit.servo[i].set_pulse_width_range(500, 2500)

# Mapeo de canales
SERVO_CHANNELS = {
    "LR": 0,   # Movimiento horizontal de ojos
    "UD": 1,   # Movimiento vertical de ojos
    "TL": 2,   # Párpado superior izquierdo
    "BL": 3,   # Párpado inferior izquierdo
    "TR": 4,   # Párpado superior derecho
    "BR": 5    # Párpado inferior derecho
}

# Límites calibrados
servo_limits = {
    "LR": (40, 140),   # Centro visual real a 90°, amplitud extendida izquierda-derecha
    "UD": (50, 130),   # Se restablece amplitud, pero con centro ajustado por inicialización
    "TL": (90, 10),    # Sup. izq
    "BL": (90, 160),   # Inf. izq
    "TR": (90, 180),   # Sup. der
    "BR": (90, 20)     # Inf. der
}

MAX_SPEED = 5

# Estado inicial
kit.servo[SERVO_CHANNELS["LR"]].angle = 90
kit.servo[SERVO_CHANNELS["UD"]].angle = 125  # Baja ligeramente la mirada inicial
kit.servo[SERVO_CHANNELS["TL"]].angle = 10
kit.servo[SERVO_CHANNELS["BL"]].angle = 160
kit.servo[SERVO_CHANNELS["TR"]].angle = 180
kit.servo[SERVO_CHANNELS["BR"]].angle = 20

# Variables de estado
auto_mode = True
controller_connected = False
exit_flag = False
joy_left_x = 0.0
joy_left_y = 0.0
joy_right_y = 0.0

DEAD_ZONE = 0.05


def normalize_axis(value, min_val=-32768, max_val=32767):
    norm = (value - min_val) / (max_val - min_val)
    return 2.0 * norm - 1.0


def controller_thread():
    global controller_connected, auto_mode, joy_left_x, joy_left_y, joy_right_y
    gamepad = None
    while not exit_flag:
        if not controller_connected:
            devices = [InputDevice(path) for path in list_devices()]
            for dev in devices:
                if dev.name and ("xbox" in dev.name.lower() or "controller" in dev.name.lower()):
                    print(f"[INFO] Mando detectado: {dev.name} en {dev.path}")
                    gamepad = InputDevice(dev.path)
                    controller_connected = True
                    auto_mode = False
                    break
        else:
            try:
                for event in gamepad.read_loop():
                    if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_EAST and event.value == 1:
                        auto_mode = not auto_mode
                        print(f"[INFO] Modo {'AUTOMÁTICO' if auto_mode else 'MANUAL'}")
                    elif event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_X:
                            joy_left_x = normalize_axis(event.value)
                            if abs(joy_left_x) < DEAD_ZONE:
                                joy_left_x = 0.0
                        elif event.code == ecodes.ABS_Y:
                            joy_left_y = normalize_axis(event.value)
                            if abs(joy_left_y) < DEAD_ZONE:
                                joy_left_y = 0.0
                        elif event.code == ecodes.ABS_RY:
                            raw = normalize_axis(event.value)
                            joy_right_y = max(0.0, min(1.0, raw))
            except OSError:
                print("[WARN] Mando desconectado")
                controller_connected = False
        time.sleep(0.5)


thr = threading.Thread(target=controller_thread, daemon=True)
thr.start()

# Estados actuales
current_lr = 90
current_ud = 125  # Acorde al ajuste de inicialización
current_blink = 0.0

try:
    while True:
        if auto_mode:
            # Movimiento aleatorio
            target_lr = random.uniform(*servo_limits["LR"])
            target_ud = random.uniform(*servo_limits["UD"])
            target_blink = random.choice([0.0, 1.0])
            time.sleep(random.uniform(0.1, 0.3))
        else:
            target_lr = servo_limits["LR"][0] + (joy_left_x + 1) / 2 * (servo_limits["LR"][1] - servo_limits["LR"][0])
            target_ud = servo_limits["UD"][0] + (joy_left_y + 1) / 2 * (servo_limits["UD"][1] - servo_limits["UD"][0])
            target_blink = joy_right_y

        # Suavizado
        def approach(current, target):
            if abs(target - current) <= MAX_SPEED:
                return target
            return current + MAX_SPEED if target > current else current - MAX_SPEED

        current_lr = approach(current_lr, target_lr)
        current_ud = approach(current_ud, target_ud)
        current_blink = target_blink

        kit.servo[SERVO_CHANNELS["LR"]].angle = current_lr
        kit.servo[SERVO_CHANNELS["UD"]].angle = current_ud

        for lid in ["TL", "BL", "TR", "BR"]:
            a = servo_limits[lid][1] + current_blink * (servo_limits[lid][0] - servo_limits[lid][1])
            kit.servo[SERVO_CHANNELS[lid]].angle = a

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] Salida por teclado")
finally:
    exit_flag = True
    for c in SERVO_CHANNELS.values():
        kit.servo[c].angle = None
    print("[INFO] Servos desactivados")
