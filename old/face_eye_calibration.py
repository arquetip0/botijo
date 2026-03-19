#!/usr/bin/env python3
"""
face_eye_calibrator.py (Versión final y documentada)

Calibra la posición de los servos de ojos (LR, UD) con la cara detectada
por la AI-Camera IMX500. Este script sigue las prácticas recomendadas
de la librería picamera2 para la inferencia en hardware.
"""

# --- Imports (solo lo estrictamente necesario) ---
import sys
import json
import time
import curses
import threading
import signal
from pathlib import Path

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from servitor import set_angle


# --- Configuración ---
MODEL = "/home/jack/botijo/packett/network.rpk"
OUTMAP = Path.home() / "eye_mapping.json"
LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 40, 120
FRAME_W, FRAME_H = 640, 480
FPS = 20

# --- Estado Global ---
servo_lr = (LR_MIN + LR_MAX) // 2
servo_ud = (UD_MIN + UD_MAX) // 2
calib_samples = []
current_face = None # Coordenadas compartidas entre hilos

# --- Utilidades ---
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# --- Inicialización de Cámara e IA ---
print("Iniciando cámara e IMX500...")
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (FRAME_W, FRAME_H)})
picam2.configure(config)

# La forma correcta de vincular la IA: se usa el pre_callback.
imx = IMX500(MODEL)
picam2.pre_callback = lambda request: imx.run(request)


picam2.start()
print("Cámara e IMX500 listos.")

# --- Interfaz de Usuario (sin cambios, ya era correcta) ---
def ui(stdscr):
    # (El código de la función ui es idéntico al de la respuesta anterior)
    # ...
    global servo_lr, servo_ud
    
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.addstr(0, 0, "← → ↑ ↓ mueve | Enter guarda punto | s guarda json | q sale")

    while True:
        face_at_this_moment = current_face
        key = stdscr.getch()
        lr_changed, ud_changed = False, False

        if key == curses.KEY_LEFT:
            new_lr = clamp(servo_lr - 1, LR_MIN, LR_MAX)
            if new_lr != servo_lr:
                servo_lr = new_lr
                lr_changed = True
        elif key == curses.KEY_RIGHT:
            new_lr = clamp(servo_lr + 1, LR_MIN, LR_MAX)
            if new_lr != servo_lr:
                servo_lr = new_lr
                lr_changed = True
        elif key == curses.KEY_UP:
            new_ud = clamp(servo_ud - 1, UD_MIN, UD_MAX)
            if new_ud != servo_ud:
                servo_ud = new_ud
                ud_changed = True
        elif key == curses.KEY_DOWN:
            new_ud = clamp(servo_ud + 1, UD_MIN, UD_MAX)
            if new_ud != servo_ud:
                servo_ud = new_ud
                ud_changed = True

        elif key in (10, 13):
            if face_at_this_moment:
                cx, cy = face_at_this_moment
                calib_samples.append({"x": cx, "y": cy, "servo_lr": servo_lr, "servo_ud": servo_ud})
                stdscr.addstr(2, 0, f"⭑ Grabado: ({cx:.0f},{cy:.0f})→{servo_lr},{servo_ud}   ")
            else:
                stdscr.addstr(2, 0, "⚠ No hay cara detectada para guardar. ")

        elif key in (ord('s'), ord('S')):
            with open(OUTMAP, "w") as f:
                json.dump(calib_samples, f, indent=2)
            stdscr.addstr(3, 0, f"💾 {len(calib_samples)} puntos guardados en {OUTMAP}")

        elif key in (ord('q'), ord('Q')):
            break

        if lr_changed:
            set_angle("LR", servo_lr)
        if ud_changed:
            set_angle("UD", servo_ud)
            
        stdscr.addstr(1, 0, f"LR={servo_lr:>3}°  UD={servo_ud:>3}° | Muestras: {len(calib_samples)}   ")
        stdscr.refresh()
        time.sleep(0.02)

# --- Bucle de Detección (sin cambios, ya era correcto) ---
def detect_loop():
    global current_face
    while True:
        # Los resultados de la IA se obtienen de los metadatos.
        meta = picam2.capture_metadata()
        detections = meta.get("inference", [])
        
        if detections:
            best_detection = max(detections, key=lambda d: d.conf)
            cx = best_detection.x + best_detection.w / 2
            cy = best_detection.y + best_detection.h / 2
            current_face = (cx, cy)
        else:
            current_face = None
            
        time.sleep(1 / FPS)

# --- Arranque del Programa ---
if __name__ == "__main__":
    detection_thread = threading.Thread(target=detect_loop, daemon=True)
    detection_thread.start()

    def signal_handler(sig, frame):
        print("\nSaliendo del programa...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    curses.wrapper(ui)
    
    picam2.stop()
    print("Fin de calibración.")