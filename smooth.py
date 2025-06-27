#!/usr/bin/env python3
# eye_tracker_smooth.py — Seguimiento facial suave para servo LR

import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from adafruit_servokit import ServoKit

# ———————————————————————————————
# CONFIGURACIÓN
# ———————————————————————————————
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.2

SERVO_CHANNEL = 0
ANGLE_MIN = 40
ANGLE_MAX = 140
MAX_SPEED = 2.5  # grados por iteración

kit = ServoKit(channels=16)

def map_range(x, in_min, in_max, out_min, out_max):
    """Mapea x del rango in a rango out"""
    return out_min + (float(x - in_min) / float(in_max - in_min)) * (out_max - out_min)

def process_detections(outputs, threshold=0.2):
    try:
        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]
        num_dets = int(outputs[3][0][0])
        detections = []
        for i in range(min(num_dets, len(scores))):
            score = float(scores[i])
            class_id = int(classes[i])
            if score >= threshold:
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'class_id': class_id,
                    'confidence': score,
                    'bbox': (float(x1), float(y1), float(x2), float(y2))
                })
        return detections
    except Exception as e:
        print(f"[ERROR] Procesando detecciones: {e}")
        return []

def approach(current, target, max_step):
    """Acerca suavemente el valor actual hacia el objetivo"""
    if abs(target - current) <= max_step:
        return target
    return current + max_step if target > current else current - max_step

# ———————————————————————————————
# PRINCIPAL
# ———————————————————————————————

def main():
    print("🔧 Cargando modelo y cámara...")
    imx500 = IMX500(RPK_PATH)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": 30}, buffer_count=12)
    picam2.configure(config)

    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        labels = ["face"]

    kit.servo[SERVO_CHANNEL].set_pulse_width_range(500, 2500)
    current_angle = (ANGLE_MIN + ANGLE_MAX) // 2
    kit.servo[SERVO_CHANNEL].angle = current_angle

    picam2.start()
    time.sleep(0.5)
    print("👁️ Seguimiento activo – Ctrl+C para salir")

    try:
        while True:
            metadata = picam2.capture_metadata()
            outputs = imx500.get_outputs(metadata, add_batch=True)

            if outputs is None:
                print("— Sin salida de red neuronal —")
                time.sleep(0.1)
                continue

            detections = process_detections(outputs, threshold=THRESHOLD)
            faces = [d for d in detections if labels[d["class_id"]] == "face"]

            if not faces:
                print("— Sin cara —")
                time.sleep(0.1)
                continue

            best = max(faces, key=lambda d: d["confidence"])
            x1, _, x2, _ = best["bbox"]
            x_center = (x1 + x2) / 2

            input_width = imx500.get_input_size()[0]
            target_angle = map_range(x_center, 0, input_width, ANGLE_MAX, ANGLE_MIN)  # ← INVERTIDO

            target_angle = max(ANGLE_MIN, min(ANGLE_MAX, target_angle))
            current_angle = approach(current_angle, target_angle, MAX_SPEED)

            print(f"🎯 X={int(x_center)} → target={int(target_angle)}°, actual={int(current_angle)}°")
            kit.servo[SERVO_CHANNEL].angle = current_angle
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("👁️ Servo centrado y cámara detenida.")
        kit.servo[SERVO_CHANNEL].angle = (ANGLE_MIN + ANGLE_MAX) // 2
        picam2.stop()

if __name__ == "__main__":
    main()
