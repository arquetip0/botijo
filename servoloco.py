#!/usr/bin/env python3
# eye_track_headless.py — seguimiento horizontal con IMX500 + servo LR

import sys, time, threading
from functools import lru_cache
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from adafruit_servokit import ServoKit

# ---------------- Configuración modelo / etiquetas ---------------- #
RPK = "/home/jack/botijo/packett/network.rpk"
LABELS_TXT = "/home/jack/botijo/labels.txt"

imx = IMX500(RPK)
intr = imx.network_intrinsics or NetworkIntrinsics()
intr.task = "object detection"
intr.threshold = 0.2
intr.iou = 0.5
intr.max_detections = 5
intr.labels = open(LABELS_TXT).read().splitlines()
intr.update_with_defaults()

# ---------------- Cámara ---------------- #
picam2 = Picamera2(imx.camera_num)
cfg = picam2.create_preview_configuration(
        controls={"FrameRate": intr.inference_rate},
        buffer_count=12)
picam2.configure(cfg)

imx.show_network_fw_progress_bar()

# Lanzamos un preview Qt para que la IA se active (no importa que no lo veas)
from picamera2.previews.qt import QGlPicamera2
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
preview = QGlPicamera2(picam2)      # Ventana mínima
preview.resize(320, 240)
preview.show()

picam2.start()                      # ¡IA activa!

# ---------------- Servo ---------------- #
LR_CH = 0
MIN_A, MAX_A = 40, 140
kit = ServoKit(channels=16)
kit.servo[LR_CH].set_pulse_width_range(500, 2500)
kit.servo[LR_CH].angle = (MIN_A + MAX_A) / 2

# ---------------- Decodificador YOLO (igual que demo) ------------- #
def detections_from_metadata(md):
    np_outputs = imx.get_outputs(md, add_batch=True)
    if np_outputs is None:
        return []
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    if intr.bbox_normalization:
        boxes = boxes / imx.get_input_size()[1]   # normalizar alto
    if intr.bbox_order == "xy":
        boxes = boxes[:, [1,0,3,2]]
    boxes = np.array_split(boxes, 4, axis=1)
    return [
        (box.reshape(-1), score, cls)
        for box, score, cls in zip(zip(*boxes), scores, classes)
        if score > intr.threshold
    ]

# ---------------- Bucle de tracking ---------------- #
def track_loop():
    while True:
        md = picam2.capture_metadata()
        dets = detections_from_metadata(md)
        if dets:
            box, conf, cls = dets[0]
            x0, y0, x1, y1 = imx.convert_inference_coords(box, md, picam2)
            cx = (x0 + x1) / 2 / picam2.capture_metadata()['ScalerCrop'][2]  # normalizado 0-1
            angle = MIN_A + cx * (MAX_A - MIN_A)
            kit.servo[LR_CH].angle = angle
        time.sleep(0.05)

threading.Thread(target=track_loop, daemon=True).start()
print("Tracking activo — mueve la cara delante de la cámara")
sys.exit(app.exec_())
