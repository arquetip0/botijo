import time
import random
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# --- Calibraciones fijas --------------------------------------------------
# Oreja izquierda  (canal 15) → 0° arriba, 180° abajo
kit.servo[15].set_pulse_width_range(min_pulse=390, max_pulse=2640)

# Oreja derecha   (canal 12) → 180° arriba, 0°  abajo (invertido)
kit.servo[12].set_pulse_width_range(min_pulse=380, max_pulse=2720)

# --- Estado inicial -------------------------------------------------------
angle_left  = 90   # mitad de recorrido (izquierda)
angle_right = 90   # mitad de recorrido (derecha; 90 mantiene el eje medio)

kit.servo[15].angle = angle_left
kit.servo[12].angle = angle_right

# --- Bucle principal ------------------------------------------------------
try:
    while True:
        # Elegir un extremo para la oreja izquierda (0-30 o 150-180)
        target_left = random.choice([random.randint(0, 30), random.randint(150, 180)])
        # Para que ambos suban y bajen simétricos, reflejamos la posición:
        #   izquierda 0° (arriba)  → derecha 180° (arriba)
        #   izquierda 180° (abajo) → derecha 0°   (abajo)
        target_right = 180 - target_left

        step = random.randint(3, 7)   # tamaño del paso

        # ---- Movimiento oreja izquierda ----
        if target_left > angle_left:
            rng_left = range(angle_left, target_left + 1, step)
        else:
            rng_left = range(angle_left, target_left - 1, -step)

        # ---- Movimiento oreja derecha ----
        if target_right > angle_right:
            rng_right = range(angle_right, target_right + 1, step)
        else:
            rng_right = range(angle_right, target_right - 1, -step)

        # Recorremos ambos rangos en paralelo
        for a_left, a_right in zip(rng_left, rng_right):
            kit.servo[15].angle = a_left
            kit.servo[12].angle = a_right
            time.sleep(random.uniform(0.005, 0.015))

        # Actualizamos ángulos actuales
        angle_left  = target_left
        angle_right = target_right

        # Pausa imprevisible antes del siguiente golpe de tentáculos
        time.sleep(random.uniform(3, 6))

except KeyboardInterrupt:
    print("\nTentáculos detenidos por el operador.")
    # Opcional: suelta los servos para que queden sin tensión
    kit.servo[15].angle = None
    kit.servo[12].angle = None
