#!/usr/bin/env python3
# detection_eye_tracking.py — seguimiento horizontal de caras con IMX500 y ServoKit

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from adafruit_servokit import ServoKit
import time
import threading

# Parámetros del servo LR (eje horizontal)
LR_CHANNEL = 0
MIN_ANGLE, MAX_ANGLE = 40, 140

# 1) Inicializa el servo (PCA9685)
kit = ServoKit(channels=16)                                              
kit.servo[LR_CHANNEL].set_pulse_width_range(500, 2500)  # rango PWM :contentReference[oaicite:4]{index=4}
kit.servo[LR_CHANNEL].angle = (MIN_ANGLE + MAX_ANGLE) / 2

# 2) Carga el modelo RPK en el IMX500
rpk_path = "/home/jack/botijo/packett/network.rpk"
imx500 = IMX500(rpk_path)  # firmware de AI en el sensor :contentReference[oaicite:5]{index=5}

# 3) Verifica o crea network_intrinsics para detección de objetos
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"                                  # según ejemplo oficial :contentReference[oaicite:6]{index=6}
elif intrinsics.task != "object detection":
    raise RuntimeError("El modelo RPK no es de detección de objetos")

# 4) Aplica valores por defecto si faltan etiquetas u otros parámetros
if intrinsics.labels is None:
    with open("/home/jack/botijo/labels.txt", "r") as f:
        intrinsics.labels = f.read().splitlines()
intrinsics.update_with_defaults()                                       

# 5) Configura Picamera2 con el sensor IMX500
picam2 = Picamera2(imx500.camera_num)                                   # el índice del sensor IMX500 :contentReference[oaicite:7]{index=7}
config = picam2.create_preview_configuration(buffer_count=4)             # vista previa mínima :contentReference[oaicite:8]{index=8}
imx500.show_network_fw_progress_bar()  # carga el firmware del modelo   :contentReference[oaicite:9]{index=9}
picam2.start(config, show_preview=False)

# 6) Bucle de seguimiento: lee detecciones y mueve servo LR
def track_loop():
    while True:
        metadata = picam2.capture_metadata()                              # métadatos incluyen detecciones :contentReference[oaicite:10]{index=10}
        detections = metadata.get("object", [])
        if detections:
            x0, _, x1, _ = detections[0].box                               # valores normalizados 0–1
            cx = (x0 + x1) / 2
            angle = MIN_ANGLE + cx * (MAX_ANGLE - MIN_ANGLE)
            kit.servo[LR_CHANNEL].angle = angle
            print(f"[TRACK] Centro X={cx:.2f} → ángulo LR={angle:.1f}")
        time.sleep(0.02)

if __name__ == "__main__":
    try:
        threading.Thread(target=track_loop, daemon=True).start()
        print("Eye tracking iniciado. Ctrl+C para detener.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        kit.servo[LR_CHANNEL].angle = (MIN_ANGLE + MAX_ANGLE) / 2       # centra al salir
        print("Deteniendo. Servo centrado.")
