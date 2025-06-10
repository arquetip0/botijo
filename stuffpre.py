#!/usr/bin/env python3
import sys, time
from functools import lru_cache
import numpy as np
from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Parámetros
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
MAX_ITERACIONES = 40

# --- Configuración (sin cambios) ---
imx500 = IMX500(RPK_PATH)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "object detection"
intrinsics.threshold = 0.20
intrinsics.iou = 0.50
intrinsics.max_detections = 5
try:
    intrinsics.labels = open(LABELS_PATH).read().splitlines()
except FileNotFoundError:
    intrinsics.labels = ["face"]
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
cfg = picam2.create_preview_configuration(
    buffer_count=12,
    controls={"FrameRate": intrinsics.inference_rate}
)
picam2.configure(cfg)

app = QApplication(sys.argv)
preview = QGlPicamera2(picam2)
preview.resize(1, 1)
preview.move(-100, -100)
preview.show()

imx500.show_network_fw_progress_bar()
picam2.start()
# Le damos un par de segundos de margen para que la IA arranque bien
print("✅ Cámara iniciada. Dando 2 segundos a la IA para arrancar...")
time.sleep(2) 

# --- Funciones de parseo (sin cambios) ---
@lru_cache
def get_labels():
    return intrinsics.labels

def parse_detections(md):
    outputs = imx500.get_outputs(md, add_batch=True)
    if outputs is None: return []
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    if intrinsics.bbox_normalization: boxes = boxes / imx500.get_input_size()[1]
    if intrinsics.bbox_order == "xy": boxes = boxes[:, [1,0,3,2]]
    parts = np.array_split(boxes, 4, axis=1)
    boxes = zip(*parts)
    dets = []
    for (b0,b1,b2,b3), score, cls in zip(boxes, scores, classes):
        if score < intrinsics.threshold: continue
        x0,y0,x1,y1 = imx500.convert_inference_coords((b0,b1,b2,b3), md, picam2)
        w, h = x1 - x0, y1 - y0
        dets.append((get_labels()[int(cls)], float(score), (int(x0), int(y0), int(w), int(h))))
    return dets

# --- Lógica del diagnóstico (CON LA CORRECCIÓN) ---
iteracion_actual = 0
def realizar_diagnostico():
    global iteracion_actual
    iteracion_actual += 1
    
    if iteracion_actual > MAX_ITERACIONES:
        print("\n✅ Diagnóstico finalizado. Cerrando aplicación.")
        app.quit()
        return

    # 👇👇👇 LA CORRECCIÓN CLAVE ESTÁ AQUÍ 👇👇👇
    md = picam2.capture_metadata(wait=False)
    
    # Si los metadatos están vacíos, es que la IA no estaba lista. No hacemos nada.
    if not md:
        print(f"🌀 Iteración {iteracion_actual}/{MAX_ITERACIONES}: Aún no hay datos de la IA, esperando...")
        return
    # 👆👆👆 FIN DE LA CORRECCIÓN 👆👆👆

    dets = parse_detections(md)
    print(f"🌀 Iteración {iteracion_actual}/{MAX_ITERACIONES}")
    # Descomenta la siguiente línea si quieres ver todos los metadatos que llegan
    # print("🔑 Metadatos disponibles:", list(md.keys()))
    print("📦 Detecciones parseadas:", dets if dets else "— ninguna —")
    print("-" * 60)

print("\n🚀 Diagnóstico: configurando QTimer y arrancando app...\n")
timer = QTimer()
timer.setSingleShot(False)
timer.timeout.connect(realizar_diagnostico)
timer.start(500) # Lo volvemos a poner a 0.5 segundos

sys.exit(app.exec_())