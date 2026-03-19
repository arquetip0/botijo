#!/usr/bin/env python3
import time
from picamera2 import Picamera2, Preview
from libcamera import Transform

# ── Iniciar cámara ──
picam2 = Picamera2()

# ── Configuración con ROTACIÓN 180° (ajustable) ──
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},
    transform=Transform(hflip=1, vflip=1)  # ↺ rota 180° (ajustar según orientación final)
)
picam2.configure(config)

# ── Iniciar preview DRM a pantalla completa ──
picam2.start_preview(Preview.DRM)  # No uses argumentos aquí
picam2.start()

print("📷  Preview activa (Ctrl‑C para salir)")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
