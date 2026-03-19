from adafruit_servokit import ServoKit
from time import sleep

# Inicializar la placa PCA9685 con 16 canales
kit = ServoKit(channels=16)

# Poner los servos del canal 0 al 5 en 90 grados
for i in range(6):
    kit.servo[i].angle = 90

print("Todos los servos en posición 90°.")
