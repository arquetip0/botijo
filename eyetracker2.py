#!/usr/bin/env python3
# eye_tracker_jump.py — Seguimiento facial por saltos oculares, estilo “mirada reactiva”

import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from adafruit_servokit import ServoKit

# ———————————————————————————————————————
# CONFIGURACIÓN
# ———————————————————————————————————————
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.2

SERVO_CHANNEL = 0
ANGLE_MIN = 40
ANGLE_MAX = 140

DELAY_BETWEEN_LOOKS = 0.8  # segundos

kit = ServoKit(channels=16)

def map_range(x, in_min, in_max, out_min, out_max):
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

# ———————————————————————————————————————
# MAIN LOOP
# ———————————————————————————————————————

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
    kit.servo[SERVO_CHANNEL].angle = (ANGLE_MIN + ANGLE_MAX) // 2

    picam2.start()
    time.sleep(0.5)
    print("🚀 Seguimiento activado – Ctrl+C para salir")

    try:
        iteration = 0
        while True:
            iteration += 1
            metadata = picam2.capture_metadata()
            outputs = imx500.get_outputs(metadata, add_batch=True)

            if outputs is None:
                print(f"[{iteration}] No outputs")
                time.sleep(DELAY_BETWEEN_LOOKS)
                continue

            detections = process_detections(outputs, threshold=THRESHOLD)
            faces = [d for d in detections if labels[d["class_id"]] == "face"]

            if not faces:
                print(f"[{iteration}] Sin caras detectadas")
                time.sleep(DELAY_BETWEEN_LOOKS)
                continue

            best = max(faces, key=lambda d: d["confidence"])
            x1, _, x2, _ = best["bbox"]
            x_center = (x1 + x2) / 2

            input_width = imx500.get_input_size()[0]
            angle = map_range(x_center, 0, input_width, ANGLE_MAX, ANGLE_MIN)
            angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))

            print(f"[{iteration}] 🎯 Centro X: {int(x_center)} → Servo: {int(angle)}°")
            kit.servo[SERVO_CHANNEL].angle = angle

            time.sleep(DELAY_BETWEEN_LOOKS)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("👁️ Servo centrado y cámara detenida.")
        kit.servo[SERVO_CHANNEL].angle = (ANGLE_MIN + ANGLE_MAX) // 2
        picam2.stop()

if __name__ == "__main__":
    main()
