#!/usr/bin/env python3
# xmp_manual.py — Procesamiento manual de detecciones

import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

# Parámetros
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.20

def process_detections_manual(outputs, threshold=0.20):
    """Procesar detecciones manualmente"""
    try:
        # Outputs format: [(1, 300, 4), (1, 300), (1, 300), (1, 1)]
        # [0] = boxes (1, 300, 4) - coordenadas de bounding boxes
        # [1] = scores (1, 300) - scores de confianza  
        # [2] = classes (1, 300) - índices de clases
        # [3] = num_detections (1, 1) - número de detecciones válidas
        
        boxes = outputs[0][0]      # (300, 4)
        scores = outputs[1][0]     # (300,)
        classes = outputs[2][0]    # (300,)
        num_dets = int(outputs[3][0][0])  # número válido de detecciones
        
        print(f"    Raw outputs - boxes: {boxes.shape}, scores: {scores.shape}, classes: {classes.shape}")
        print(f"    Número de detecciones válidas: {num_dets}")
        
        detections = []
        
        # Procesar solo las detecciones válidas
        for i in range(min(num_dets, len(scores))):
            score = float(scores[i])
            class_id = int(classes[i])
            
            if score >= threshold:
                # Coordenadas del bounding box
                x1, y1, x2, y2 = boxes[i]
                
                detections.append({
                    'class_id': class_id,
                    'confidence': score,
                    'bbox': (float(x1), float(y1), float(x2), float(y2))
                })
        
        return detections
        
    except Exception as e:
        print(f"    Error procesando detecciones: {e}")
        return []

def main():
    # Configuración
    imx500 = IMX500(RPK_PATH)
    picam2 = Picamera2(imx500.camera_num)
    
    config = picam2.create_preview_configuration(controls={"FrameRate": 30}, buffer_count=12)
    picam2.configure(config)
    
    # Cargar labels
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        labels = ["face"]  # Por defecto
    
    print(f"Labels cargadas: {labels}")
    
    picam2.start()
    
    try:
        successful_detections = 0
        total_detections = 0
        
        for i in range(30):
            metadata = picam2.capture_metadata()
            
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is not None:
                print(f"\nIteración {i+1}:")
                print(f"  Outputs shapes: {[out.shape for out in np_outputs]}")
                
                # Procesar manualmente
                detections = process_detections_manual(np_outputs, threshold=THRESHOLD)
                
                if detections:
                    successful_detections += 1
                    total_detections += len(detections)
                    
                print(f"  ✅ Detecciones encontradas: {len(detections)}")
                
                for j, det in enumerate(detections):
                    class_id = det['class_id']
                    label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
                    confidence = det['confidence']
                    bbox = det['bbox']
                    
                    print(f"    {j+1}. {label}: {confidence:.3f} @ {bbox}")
                    
            else:
                print(f"Iteración {i+1}: No outputs")
            
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")
    
    finally:
        picam2.stop()
        
        print(f"\n📊 Resumen final:")
        print(f"   Iteraciones con detecciones: {successful_detections}/30")
        print(f"   Total de detecciones: {total_detections}")
        print(f"   Tasa de éxito: {successful_detections/30*100:.1f}%")

if __name__ == "__main__":
    main()