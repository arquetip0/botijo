#!/usr/bin/env python3
# eye_tracker_jump.py — Seguimiento facial por saltos oculares, estilo "mirada reactiva"

import time
import numpy as np
import random
import math
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
UD_MIN, UD_MAX = 30, 150

DELAY_BETWEEN_LOOKS = 1

# ✅ Configuración de parpadeo realista
BLINK_PROBABILITY = 0.03  # 3% - más natural
BLINK_DURATION = 0.12     # Parpadeo más rápido
DOUBLE_BLINK_PROBABILITY = 0.3  # 30% de hacer parpadeo doble

# ✅ Configuración de mirada aleatoria
RANDOM_LOOK_PROBABILITY = 0.04
RANDOM_LOOK_DURATION = 0.6

# ✅ NUEVO: Configuración de micro-movimientos
MICRO_MOVEMENT_PROBABILITY = 0.15  # 15% probabilidad de micro-movimiento
MICRO_MOVEMENT_RANGE = 3  # ±3 grados de movimiento sutil

# ✅ NUEVO: Configuración de seguimiento suave
SMOOTHING_FACTOR = 0  # Cuánto del movimiento anterior mantener (0-1)
previous_lr = 90  # Posición anterior LR
previous_ud = 90  # Posición anterior UD

# ✅ NUEVO: Configuración de entrecerrar ojos
SQUINT_PROBABILITY = 0.02  # 2% probabilidad de entrecerrar
SQUINT_DURATION = 1.5
SQUINT_INTENSITY = 0.6  # Cuánto cerrar (0-1)

# ✅ NUEVO: Patrón de respiración en párpados
BREATHING_ENABLED = True
BREATHING_CYCLE = 4.0  # Segundos por ciclo completo
BREATHING_INTENSITY = 0.15  # Cuánto abrir/cerrar con respiración

kit = ServoKit(channels=16)

def map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (float(x - in_min) / float(in_max - in_min)) * (out_max - out_min)

def smooth_movement(current, target, factor):
    """✅ Movimiento suavizado para evitar saltos bruscos"""
    return current + (target - current) * (1 - factor)

def add_micro_movement(angle, range_limit=MICRO_MOVEMENT_RANGE):
    """✅ Añadir micro-movimientos naturales"""
    if random.random() < MICRO_MOVEMENT_PROBABILITY:
        micro_offset = random.uniform(-range_limit, range_limit)
        return angle + micro_offset
    return angle

def breathing_adjustment():
    """✅ Calcular ajuste de párpados basado en patrón de respiración"""
    if not BREATHING_ENABLED:
        return 0
    
    # Usar seno para patrón suave de respiración
    time_in_cycle = (time.time() % BREATHING_CYCLE) / BREATHING_CYCLE
    breathing_offset = math.sin(time_in_cycle * 2 * math.pi) * BREATHING_INTENSITY
    return breathing_offset

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
    """✅ Parpadeo realista con velocidad variable"""
    print("👁️ ¡Parpadeo!")
    
    # Cerrar párpados rápidamente
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        kit.servo[servo_num].angle = 90
    
    time.sleep(BLINK_DURATION)
    
    # Abrir párpados más lentamente (más natural)
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        angle = EYELID_POSITIONS[channel] + breathing_adjustment()
        kit.servo[servo_num].angle = angle
    
    # ✅ Posibilidad de parpadeo doble
    if random.random() < DOUBLE_BLINK_PROBABILITY:
        print("👁️👁️ ¡Parpadeo doble!")
        time.sleep(0.1)  # Pausa corta
        # Repetir parpadeo
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            kit.servo[servo_num].angle = 90
        time.sleep(BLINK_DURATION * 0.8)  # Segundo parpadeo más rápido
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            angle = EYELID_POSITIONS[channel] + breathing_adjustment()
            kit.servo[servo_num].angle = angle

def squint():
    """✅ Entrecerrar los ojos (concentración/sospecha)"""
    print("😑 Entrecerrando ojos...")
    
    # Cerrar párpados parcialmente
    for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
        base_angle = EYELID_POSITIONS[channel]
        squint_angle = base_angle + (90 - base_angle) * SQUINT_INTENSITY
        kit.servo[servo_num].angle = squint_angle
    
    time.sleep(SQUINT_DURATION)
    
    # Volver a posición normal gradualmente
    steps = 5
    for step in range(steps):
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            base_angle = EYELID_POSITIONS[channel]
            squint_angle = base_angle + (90 - base_angle) * SQUINT_INTENSITY
            progress = (step + 1) / steps
            current_angle = squint_angle + (base_angle - squint_angle) * progress
            kit.servo[servo_num].angle = current_angle + breathing_adjustment()
        time.sleep(0.1)

def update_eyelids_with_breathing():
    """✅ Actualizar párpados con patrón de respiración"""
    if BREATHING_ENABLED:
        breathing_offset = breathing_adjustment()
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            base_angle = EYELID_POSITIONS[channel]
            # Aplicar offset de respiración (sutil)
            adjusted_angle = base_angle + breathing_offset * 5  # Multiplicar para hacer más visible
            adjusted_angle = max(10, min(170, adjusted_angle))  # Mantener en límites seguros
            kit.servo[servo_num].angle = adjusted_angle

def should_blink():
    return random.random() < BLINK_PROBABILITY

def should_look_random():
    return random.random() < RANDOM_LOOK_PROBABILITY

def should_squint():
    return random.random() < SQUINT_PROBABILITY

def look_random():
    """✅ Mirada aleatoria con movimiento más natural"""
    global previous_lr, previous_ud
    
    # Generar punto aleatorio
    random_lr = random.uniform(LR_MIN, LR_MAX)
    random_ud = random.uniform(UD_MIN, UD_MAX)
    
    # Añadir micro-movimientos
    random_lr = add_micro_movement(random_lr)
    random_ud = add_micro_movement(random_ud)
    
    print(f"👀 Mirada curiosa → LR:{int(random_lr)}° UD:{int(random_ud)}°")
    
    # Movimiento suavizado hacia el punto aleatorio
    steps = 8  # Número de pasos para llegar
    for step in range(steps):
        progress = (step + 1) / steps
        current_lr = previous_lr + (random_lr - previous_lr) * progress
        current_ud = previous_ud + (random_ud - previous_ud) * progress
        
        kit.servo[SERVO_CHANNEL_LR].angle = current_lr
        kit.servo[SERVO_CHANNEL_UD].angle = current_ud
        time.sleep(0.05)  # Pequeña pausa entre pasos
    
    # Actualizar posiciones anteriores
    previous_lr = random_lr
    previous_ud = random_ud
    
    # Mantener la mirada con micro-movimientos
    hold_steps = int(RANDOM_LOOK_DURATION / 0.1)
    for _ in range(hold_steps):
        micro_lr = add_micro_movement(random_lr, 1)  # Micro-movimientos más sutiles
        micro_ud = add_micro_movement(random_ud, 1)
        kit.servo[SERVO_CHANNEL_LR].angle = micro_lr
        kit.servo[SERVO_CHANNEL_UD].angle = micro_ud
        time.sleep(0.1)
    
    return random_lr, random_ud

def main():
    global previous_lr, previous_ud
    
    print("🔧 Cargando modelo y cámara...")
    
    # Configuración mejorada del IMX500
    imx500 = IMX500(RPK_PATH)
    
    # Configurar network intrinsics
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
        buffer_count=12,
        controls={"FrameRate": intrinsics.inference_rate}
    )
    picam2.configure(config)

    # Inicializar servos
    kit.servo[SERVO_CHANNEL_LR].set_pulse_width_range(500, 2500)
    kit.servo[SERVO_CHANNEL_UD].set_pulse_width_range(500, 2500)
    
    # Posición inicial
    initial_lr = (LR_MIN + LR_MAX) // 2
    initial_ud = (UD_MIN + UD_MAX) // 2
    kit.servo[SERVO_CHANNEL_LR].angle = initial_lr
    kit.servo[SERVO_CHANNEL_UD].angle = initial_ud
    
    # Inicializar posiciones anteriores
    previous_lr = initial_lr
    previous_ud = initial_ud
    
    print(f"🔧 [INIT] Servo LR inicializado → {initial_lr}°")
    print(f"🔧 [INIT] Servo UD inicializado → {initial_ud}°")

    # ✅ Inicializar párpados
    initialize_eyelids()

    # Mostrar progreso de carga
    print("🔄 Cargando firmware de IA...")
    imx500.show_network_fw_progress_bar()
    
    picam2.start()
    
    print("⏳ Esperando estabilización de la IA...")
    time.sleep(3)
    
    print("🚀 Seguimiento activado – Ctrl+C para salir")
    print(f"📊 Usando threshold: {intrinsics.threshold}, tasa: {intrinsics.inference_rate} FPS")
    print(f"👁️ Probabilidad de parpadeo: {BLINK_PROBABILITY*100:.1f}%")
    print(f"👀 Probabilidad de mirada curiosa: {RANDOM_LOOK_PROBABILITY*100:.1f}%")
    print(f"😑 Probabilidad de entrecerrar: {SQUINT_PROBABILITY*100:.1f}%")
    print(f"🫁 Respiración en párpados: {'✅' if BREATHING_ENABLED else '❌'}")

    try:
        iteration = 0
        consecutive_no_outputs = 0
        last_breathing_update = time.time()
        
        while True:
            iteration += 1
            
            # ✅ Actualizar respiración en párpados cada cierto tiempo
            if time.time() - last_breathing_update > 0.2:  # Cada 200ms
                update_eyelids_with_breathing()
                last_breathing_update = time.time()
            
            # ✅ Verificar comportamientos aleatorios
            if should_blink():
                blink()
            elif should_squint():
                squint()
            elif should_look_random():
                look_random()
                print("🔄 Volviendo al seguimiento facial...")
            
            # Captura con timeout
            try:
                metadata = picam2.capture_metadata(wait=True)
            except Exception as e:
                print(f"[{iteration}] Error capturando metadata: {e}")
                time.sleep(0.1)
                continue
                
            outputs = imx500.get_outputs(metadata, add_batch=True)

            if outputs is None:
                consecutive_no_outputs += 1
                if consecutive_no_outputs % 10 == 0:
                    print(f"[{iteration}] No outputs (consecutivos: {consecutive_no_outputs})")
                
                if consecutive_no_outputs > 50:
                    print("⚠️ Demasiados fallos, reiniciando...")
                    time.sleep(1)
                    consecutive_no_outputs = 0
                else:
                    time.sleep(0.05)
                continue
            
            consecutive_no_outputs = 0

            detections = process_detections(outputs, threshold=THRESHOLD)
            faces = [d for d in detections if intrinsics.labels[d["class_id"]] == "face"]

            if not faces:
                if iteration % 20 == 0:
                    print(f"[{iteration}] Sin caras detectadas")
                time.sleep(DELAY_BETWEEN_LOOKS)
                continue

            best = max(faces, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["bbox"]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            input_width, input_height = imx500.get_input_size()

            # Calcular ángulos objetivo
            target_lr = map_range(x_center, 0, input_width, LR_MAX, LR_MIN)
            target_lr = max(LR_MIN, min(LR_MAX, target_lr))
            
            target_ud = map_range(y_center, 0, input_height, UD_MAX, UD_MIN)
            target_ud = max(UD_MIN, min(UD_MAX, target_ud))
            
            # ✅ Aplicar suavizado y micro-movimientos
            smooth_lr = smooth_movement(previous_lr, target_lr, SMOOTHING_FACTOR)
            smooth_ud = smooth_movement(previous_ud, target_ud, SMOOTHING_FACTOR)
            
            final_lr = add_micro_movement(smooth_lr)
            final_ud = add_micro_movement(smooth_ud)
            
            # Aplicar límites finales
            final_lr = max(LR_MIN, min(LR_MAX, final_lr))
            final_ud = max(UD_MIN, min(UD_MAX, final_ud))

            # Mover servos
            kit.servo[SERVO_CHANNEL_LR].angle = final_lr
            kit.servo[SERVO_CHANNEL_UD].angle = final_ud
            
            # Actualizar posiciones anteriores
            previous_lr = final_lr
            previous_ud = final_ud

            print(f"[{iteration}] 🎯 X:{int(x_center)} Y:{int(y_center)} → LR:{int(final_lr)}° UD:{int(final_ud)}° (conf:{best['confidence']:.2f})")

            time.sleep(DELAY_BETWEEN_LOOKS)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("👁️ Servo centrado y cámara detenida.")
        kit.servo[SERVO_CHANNEL_LR].angle = (LR_MIN + LR_MAX) // 2
        kit.servo[SERVO_CHANNEL_UD].angle = (UD_MIN + UD_MAX) // 2
        
        # Restaurar párpados
        for channel, servo_num in SERVO_CHANNELS_EYELIDS.items():
            angle = EYELID_POSITIONS[channel]
            kit.servo[servo_num].angle = angle
            print(f"🔧 [SHUTDOWN] Servo {channel} restaurado → {angle}°")
        
        picam2.stop()

if __name__ == "__main__":
    main()