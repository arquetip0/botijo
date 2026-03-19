#!/usr/bin/env python3
# eye_tracker_jump.py — Seguimiento facial por saltos oculares, estilo "mirada reactiva"

import time
import numpy as np
import random
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from adafruit_servokit import ServoKit

# ———————————————————————————————————————
# CONFIGURACIÓN
# ———————————————————————————————————————
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.2

SERVO_CHANNEL_LR = 0  # izquierda-derecha
SERVO_CHANNEL_UD = 1  # arriba-abajo

# ✅ Canales de párpados
SERVO_CHANNELS_EYELIDS = {
    "TL": 2, "BL": 3, "TR": 4, "BR": 5
}

# ✅ Posiciones de párpados personalizadas
EYELID_POSITIONS = {
    "TL": 50,   # Párpado Superior Izq
    "BL": 135,  # Párpado Inferior Izq  
    "TR": 135,  # Párpado Superior Der
    "BR": 45    # Párpado Inferior Der
}

LR_MIN, LR_MAX = 40, 140
UD_MIN, UD_MAX = 30, 150  # 70: abajo, 120: arriba

DELAY_BETWEEN_LOOKS = 1  # Reducido para más fluidez

# ✅ Configuración de parpadeo
BLINK_PROBABILITY = 0.05  # 10% probabilidad de parpadear cada iteración
BLINK_DURATION = 0.15     # Duración del parpadeo en segundos

# ✅ Configuración de mirada aleatoria
RANDOM_LOOK_PROBABILITY = 0.05  # 10% probabilidad de mirar a punto aleatorio
RANDOM_LOOK_DURATION = 0.5     # Tiempo mirando al punto aleatorio

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

def initialize_eyelids():
    """✅ Inicializar párpados en posiciones personalizadas"""
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        kit.servo[servo_num].set_pulse_width_range(500, 2500)
        angle = EYELID_POSITIONS[channel]
        kit.servo[servo_num].angle = angle
        print(f"🔧 [INIT] Servo {channel} (Párpado) inicializado → {angle}°")

def blink():
    """✅ Ejecutar un parpadeo completo"""
    print("👁️ ¡Parpadeo!")
    
    # Cerrar párpados (todos a 90°)
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        kit.servo[servo_num].angle = 90
    
    # Mantener cerrados
    time.sleep(BLINK_DURATION)
    
    # Abrir párpados (volver a posiciones originales)
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        angle = EYELID_POSITIONS[channel]
        kit.servo[servo_num].angle = angle

def should_blink():
    """✅ Determinar si debe parpadear basado en probabilidad"""
    return random.random() < BLINK_PROBABILITY

def should_look_random():
    """✅ Determinar si debe mirar a punto aleatorio"""
    return random.random() < RANDOM_LOOK_PROBABILITY

def look_random():
    """✅ Mirar a un punto aleatorio y luego volver"""
    # Generar punto aleatorio dentro de los límites
    random_lr = random.uniform(LR_MIN, LR_MAX)
    random_ud = random.uniform(UD_MIN, UD_MAX)
    
    print(f"👀 Mirada curiosa → LR:{int(random_lr)}° UD:{int(random_ud)}°")
    
    # Mover a punto aleatorio
    kit.servo[SERVO_CHANNEL_LR].angle = random_lr
    kit.servo[SERVO_CHANNEL_UD].angle = random_ud
    
    # Mantener la mirada
    time.sleep(RANDOM_LOOK_DURATION)
    
    return random_lr, random_ud  # Devolver posición para logging

# ———————————————————————————————————————
# MAIN LOOP
# ———————————————————————————————————————

def main():
    print("🔧 Cargando modelo y cámara...")
    
    # Configuración mejorada del IMX500
    imx500 = IMX500(RPK_PATH)
    
    # Configurar network intrinsics (clave para la estabilidad)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    intrinsics.threshold = THRESHOLD
    intrinsics.iou = 0.5
    intrinsics.max_detections = 5
    
    # Cargar labels
    try:
        with open(LABELS_PATH, 'r') as f:
            intrinsics.labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        intrinsics.labels = ["face"]
    
    intrinsics.update_with_defaults()
    
    # Configuración de cámara optimizada
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        buffer_count=12,  # Más buffers para estabilidad  
        controls={"FrameRate": intrinsics.inference_rate}  # Usar la tasa de inferencia del modelo
    )
    picam2.configure(config)

    # Inicializar servos de movimiento ocular
    kit.servo[SERVO_CHANNEL_LR].set_pulse_width_range(500, 2500)
    kit.servo[SERVO_CHANNEL_UD].set_pulse_width_range(500, 2500)
    kit.servo[SERVO_CHANNEL_LR].angle = (LR_MIN + LR_MAX) // 2
    kit.servo[SERVO_CHANNEL_UD].angle = (UD_MIN + UD_MAX) // 2
    print(f"🔧 [INIT] Servo LR inicializado → {(LR_MIN + LR_MAX) // 2}°")
    print(f"🔧 [INIT] Servo UD inicializado → {(UD_MIN + UD_MAX) // 2}°")

    # ✅ Inicializar párpados
    initialize_eyelids()

    # Mostrar progreso de carga del firmware (importante)
    print("🔄 Cargando firmware de IA...")
    imx500.show_network_fw_progress_bar()
    
    picam2.start()
    
    # Tiempo de estabilización más largo
    print("⏳ Esperando estabilización de la IA...")
    time.sleep(3)  # Crucial para que la IA se inicialice completamente
    
    print("🚀 Seguimiento activado – Ctrl+C para salir")
    print(f"📊 Usando threshold: {intrinsics.threshold}, tasa: {intrinsics.inference_rate} FPS")
    print(f"👁️ Probabilidad de parpadeo: {BLINK_PROBABILITY*100:.1f}% por iteración")
    print(f"👀 Probabilidad de mirada curiosa: {RANDOM_LOOK_PROBABILITY*100:.1f}% por iteración")

    try:
        iteration = 0
        consecutive_no_outputs = 0
        
        while True:
            iteration += 1
            
            # ✅ Verificar si debe parpadear (antes de procesar detecciones)
            if should_blink():
                blink()
            
            # ✅ Verificar si debe mirar a punto aleatorio
            if should_look_random():
                look_random()
                print("🔄 Volviendo al seguimiento facial...")
            
            # Captura con timeout para evitar bloqueos
            try:
                metadata = picam2.capture_metadata(wait=True)
            except Exception as e:
                print(f"[{iteration}] Error capturando metadata: {e}")
                time.sleep(0.1)
                continue
                
            outputs = imx500.get_outputs(metadata, add_batch=True)

            if outputs is None:
                consecutive_no_outputs += 1
                if consecutive_no_outputs % 10 == 0:  # Solo mostrar cada 10 fallos
                    print(f"[{iteration}] No outputs (consecutivos: {consecutive_no_outputs})")
                
                # Si hay muchos fallos consecutivos, reintentar con pausa más larga
                if consecutive_no_outputs > 50:
                    print("⚠️ Demasiados fallos, reiniciando...")
                    time.sleep(1)
                    consecutive_no_outputs = 0
                else:
                    time.sleep(0.05)  # Pausa corta para no saturar
                continue
            
            consecutive_no_outputs = 0  # Reset contador de fallos

            detections = process_detections(outputs, threshold=THRESHOLD)
            faces = [d for d in detections if intrinsics.labels[d["class_id"]] == "face"]

            if not faces:
                if iteration % 20 == 0:  # Solo mostrar esporádicamente
                    print(f"[{iteration}] Sin caras detectadas")
                time.sleep(DELAY_BETWEEN_LOOKS)
                continue

            best = max(faces, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["bbox"]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            input_width, input_height = imx500.get_input_size()

            # Movimiento horizontal (LR)
            angle_lr = map_range(x_center, 0, input_width, LR_MAX, LR_MIN)
            angle_lr = max(LR_MIN, min(LR_MAX, angle_lr))
            kit.servo[SERVO_CHANNEL_LR].angle = angle_lr

            # Movimiento vertical (UD)
            angle_ud = map_range(y_center, 0, input_height, UD_MAX, UD_MIN)
            angle_ud = max(UD_MIN, min(UD_MAX, angle_ud))
            kit.servo[SERVO_CHANNEL_UD].angle = angle_ud

            print(f"[{iteration}] 🎯 X:{int(x_center)} Y:{int(y_center)} → LR:{int(angle_lr)}° UD:{int(angle_ud)}° (conf:{best['confidence']:.2f})")

            time.sleep(DELAY_BETWEEN_LOOKS)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("👁️ Servo centrado y cámara detenida.")
        kit.servo[SERVO_CHANNEL_LR].angle = (LR_MIN + LR_MAX) // 2
        kit.servo[SERVO_CHANNEL_UD].angle = (UD_MIN + UD_MAX) // 2
        
        # ✅ Restaurar párpados a posición abierta al salir
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            angle = EYELID_POSITIONS[channel]
            kit.servo[servo_num].angle = angle
            print(f"🔧 [SHUTDOWN] Servo {channel} restaurado → {angle}°")
        
        picam2.stop()

if __name__ == "__main__":
    main()