from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)  # PCA9685 estándar
kit.servo[15].angle = 0





