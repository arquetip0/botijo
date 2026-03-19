#!/usr/bin/env python3
# stuff_fixed.py — Diagnóstico corregido IMX500

import sys
import time
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection
import numpy as np

# Parámetros - usar exactamente los mismos del ejemplo oficial
RPK_PATH = "/home/jack/botijo/packett/network.rpk"
LABELS_PATH = "/home/jack/botijo/labels.txt"
THRESHOLD = 0.20
BBOX_NORMALIZATION = True
BBOX_ORDER = "xy"

class IMX500Diagnostics:
    def __init__(self):
        self.running = False
        self.detection_count = 0
        
    def setup_camera(self):
        """Configuración exacta del ejemplo oficial"""
        try:
            # Configurar IMX500 exactamente como el ejemplo oficial
            self.imx500 = IMX500(RPK_PATH)
            self.picam2 = Picamera2(self.imx500.camera_num)
            
            # Configuración exacta del ejemplo oficial
            config = self.picam2.create_preview_configuration(
                controls={"FrameRate": 30}, 
                buffer_count=12
            )
            
            self.picam2.configure(config)
            
            # Cargar etiquetas exactamente como el ejemplo oficial
            try:
                with open(LABELS_PATH, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                self.labels = ["face"]
                print(f"⚠️ Labels file not found, using default: {self.labels}")
            
            print(f"✅ Configuración aplicada")
            print(f"   Etiquetas: {self.labels}")
            print(f"   Umbral: {THRESHOLD}")
            print(f"   Bbox normalization: {BBOX_NORMALIZATION}")
            print(f"   Bbox order: {BBOX_ORDER}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error configurando cámara: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_detections_official(self, metadata):
        """Usar exactamente la misma función del ejemplo oficial"""
        try:
            # Copiar exactamente del ejemplo oficial
            np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
            
            if np_outputs is not None:
                # Usar la función oficial de postprocesamiento
                detections = postprocess_nanodet_detection(
                    np_outputs,
                    input_size=self.imx500.get_input_size(),
                    metadata=metadata,
                    picam2=self.picam2,
                    threshold=THRESHOLD,
                    bbox_normalization=BBOX_NORMALIZATION,
                    bbox_order=BBOX_ORDER
                )
                
                # Convertir a formato consistente
                results = []
                if detections:
                    for detection in detections:
                        if len(detection) >= 3:
                            # detection format: [label_idx, confidence, bbox]
                            label_idx, confidence, bbox = detection[0], detection[1], detection[2:]
                            
                            # Convertir índice a etiqueta
                            if isinstance(label_idx, (int, float)) and int(label_idx) < len(self.labels):
                                label = self.labels[int(label_idx)]
                            else:
                                label = str(label_idx)
                            
                            results.append((
                                label,
                                float(confidence),
                                tuple(bbox)
                            ))
                
                return results
            
            return []
            
        except Exception as e:
            print(f"⚠️ Error en parse_detections_official: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_diagnostics(self, iterations=20):
        """Ejecutar diagnósticos exactamente como el ejemplo oficial"""
        if not self.setup_camera():
            return False
        
        try:
            print("🚀 Iniciando cámara...")
            self.picam2.start()
            
            # Esperar estabilización
            print("⏳ Esperando estabilización...")
            time.sleep(2)
            
            print(f"\n🔍 Iniciando {iterations} iteraciones de diagnóstico...\n")
            
            self.running = True
            successful_detections = 0
            total_detections = 0
            
            for i in range(iterations):
                if not self.running:
                    break
                
                try:
                    # Capturar exactamente como el ejemplo oficial
                    metadata = self.picam2.capture_metadata()
                    
                    # Debug: mostrar info de metadatos
                    print(f"🌀 Iteración {i+1}/{iterations}")
                    print(f"   📊 Metadatos disponibles: {list(metadata.keys())[:5]}...")
                    
                    # Verificar si hay outputs disponibles
                    np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
                    print(f"   🔢 Outputs disponibles: {np_outputs is not None}")
                    
                    if np_outputs is not None:
                        print(f"   📐 Shapes de outputs: {[out.shape if hasattr(out, 'shape') else type(out) for out in np_outputs]}")
                    
                    # Procesar detecciones usando la función oficial
                    detections = self.parse_detections_official(metadata)
                    
                    total_detections += len(detections)
                    
                    # Estadísticas
                    if detections:
                        successful_detections += 1
                    
                    print(f"   🎯 Detecciones: {len(detections)}")
                    
                    if detections:
                        for j, (label, conf, bbox) in enumerate(detections):
                            print(f"     {j+1}. {label}: {conf:.3f} @ {bbox}")
                    else:
                        print("     — Sin detecciones —")
                    
                    print("-" * 50)
                    
                    # Pausa
                    time.sleep(0.2)
                    
                except KeyboardInterrupt:
                    print("\n🛑 Interrumpido por usuario")
                    break
                except Exception as e:
                    print(f"⚠️ Error en iteración {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Resumen final
            print(f"\n📈 Resumen:")
            print(f"   Iteraciones completadas: {i+1}")
            print(f"   Iteraciones con detecciones: {successful_detections}")
            print(f"   Total de detecciones: {total_detections}")
            print(f"   Tasa de éxito: {successful_detections/(i+1)*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpieza de recursos"""
        self.running = False
        try:
            if hasattr(self, 'picam2'):
                self.picam2.stop()
                print("✅ Cámara detenida")
        except Exception as e:
            print(f"⚠️ Error en limpieza: {e}")

def main():
    print("🔧 Diagnóstico IMX500 - Versión oficial")
    print("=" * 50)
    
    diagnostics = IMX500Diagnostics()
    
    try:
        success = diagnostics.run_diagnostics(iterations=30)
        if success:
            print("\n✅ Diagnóstico completado")
        else:
            print("\n❌ Diagnóstico falló")
            
    except KeyboardInterrupt:
        print("\n🛑 Programa interrumpido")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        diagnostics.cleanup()

if __name__ == "__main__":
    main()