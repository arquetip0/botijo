#!/usr/bin/env python3
# deepseekloco.py - Face detection with IMX500

import sys
import time
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection
from picamera2.previews.qt import QGlPicamera2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import QTimer

# Parámetros
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.20
BBOX_NORMALIZATION = True
BBOX_ORDER = "xy"

class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_camera()
        self.setup_ui()
        self.setup_timer()
        
    def setup_camera(self):
        """Configurar cámara e IMX500"""
        try:
            # Verificar si el archivo RPK existe
            import os
            if not os.path.exists(RPK_PATH):
                raise FileNotFoundError(f"RPK file not found: {RPK_PATH}")
            
            # Configuración IMX500
            self.imx500 = IMX500(RPK_PATH)
            self.picam2 = Picamera2(self.imx500.camera_num)
            
            # Configuración de cámara
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                buffer_count=6,
                controls={"FrameRate": 30}
            )
            self.picam2.configure(config)
            
            # Cargar etiquetas
            try:
                with open(LABELS_PATH) as f:
                    self.labels = [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                self.labels = ["face"]
                print(f"⚠️ Labels file not found, using default: {self.labels}")
                
            print("✅ Cámara configurada correctamente")
            
        except Exception as e:
            print(f"❌ Error configurando cámara: {e}")
            sys.exit(1)
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("Face Detection - IMX500")
        self.setGeometry(100, 100, 800, 600)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Visor de cámara
        self.preview = QGlPicamera2(self.picam2, width=640, height=480)
        layout.addWidget(self.preview)
        
        # Etiqueta de información
        self.info_label = QLabel("Iniciando detección...")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
    def setup_timer(self):
        """Configurar timer para actualización"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_detections)
        self.timer.start(100)  # Actualizar cada 100ms
        
    def update_detections(self):
        """Actualizar detecciones"""
        try:
            # Capturar metadata
            md = self.picam2.capture_metadata()
            
            if md is None:
                return
                
            # Obtener outputs del IMX500
            outputs = self.imx500.get_outputs(md, add_batch=True)
            
            if outputs is not None and len(outputs) > 0:
                # Procesar detecciones
                detections = postprocess_nanodet_detection(
                    outputs,
                    input_size=self.imx500.get_input_size(),
                    metadata=md,
                    picam2=self.picam2,
                    threshold=THRESHOLD,
                    bbox_normalization=BBOX_NORMALIZATION,
                    bbox_order=BBOX_ORDER
                )
                
                # Actualizar información
                num_faces = len(detections) if detections else 0
                self.info_label.setText(f"Rostros detectados: {num_faces}")
                
                # Imprimir detecciones para debug
                if detections:
                    for i, detection in enumerate(detections):
                        label, confidence, bbox = detection
                        print(f"Detección {i+1}: {label} ({confidence:.2f}) - {bbox}")
                
            else:
                self.info_label.setText("Sin detecciones")
                
        except Exception as e:
            print(f"⚠️ Error en update_detections: {e}")
            self.info_label.setText(f"Error: {str(e)}")
    
    def closeEvent(self, event):
        """Limpiar al cerrar"""
        try:
            self.timer.stop()
            if hasattr(self, 'picam2'):
                self.picam2.stop()
            print("🏁 Limpieza completada")
        except Exception as e:
            print(f"⚠️ Error durante limpieza: {e}")
        event.accept()

def main():
    """Función principal"""
    print("🚀 Iniciando aplicación de detección de rostros...")
    
    # Crear aplicación Qt
    app = QApplication(sys.argv)
    
    # Crear ventana principal
    window = FaceDetectionApp()
    
    # Iniciar cámara
    try:
        window.picam2.start()
        print("✅ Cámara iniciada")
        
        # Mostrar ventana
        window.show()
        
        # Ejecutar aplicación
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Error iniciando aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()