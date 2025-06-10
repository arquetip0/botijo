from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import time

imx = IMX500("/home/jack/botijo/packett/network.rpk")

intr = imx.network_intrinsics or NetworkIntrinsics()
intr.task = "object detection"
intr.threshold = 0.2
intr.max_detections = 5
intr.labels = open("/home/jack/botijo/labels.txt").read().splitlines()
intr.update_with_defaults()

picam2 = Picamera2(imx.camera_num)
cfg = picam2.create_preview_configuration(
    buffer_count=12,
    controls={"FrameRate": intr.inference_rate}
)

picam2.configure(cfg)
print("FrameRate:", intr.inference_rate)
print("Threshold:", intr.threshold)

imx.show_network_fw_progress_bar()
picam2.start(show_preview=False)

try:
    for _ in range(60):
        md = picam2.capture_metadata()
        print("DEBUG:", md.get("object"))
        time.sleep(0.5)
finally:
    print("FIN")
