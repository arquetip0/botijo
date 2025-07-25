from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# Cambia estos valores para adaptarlos a tu servo:
kit.servo[10].set_pulse_width_range(min_pulse=500, max_pulse=2700)

# Ahora 0 debería acercarse al mínimo físico
# y 180 al máximo físico.

# Luego prueba:
kit.servo[10].angle =180





















