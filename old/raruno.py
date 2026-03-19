from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import time, threading
from adafruit_servokit import ServoKit

# Servo LR
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500,2500)
kit.servo[0].angle = 90

# Carga modelo e intrinsics
imx = IMX500("/home/jack/botijo/packett/network.rpk")
intr = imx.network_intrinsics or NetworkIntrinsics()
intr.task = "object detection"
intr.threshold = 0.2  # baja si no funciona
intr.max_detections = 5
intr.labels = open("/home/jack/botijo/labels.txt").read().splitlines()
intr.update_with_defaults()

# Configura la cámara igual que el demo
picam2 = Picamera2(imx.camera_num)
cfg = picam2.create_preview_configuration(
    buffer_count=12,
    controls={"FrameRate": intr.inference_rate}
)
picam2.configure(cfg)
print("FrameRate:", intr.inference_rate, "Threshold:", intr.threshold)
imx.show_network_fw_progress_bar()
picam2.start(show_preview=False)

# Hilo básico que imprime detecciones
def track():
    while True:
        md = picam2.capture_metadata()
        objs = md.get("object", [])
        print("DEBUG:", objs)
        time.sleep(0.5)

threading.Thread(target=track, daemon=True).start()
print("CORRIENDO: observa la consola, mueve tu cara...")
time.sleep(60)
