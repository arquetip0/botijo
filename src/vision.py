"""Vision module — IMX500 face detection.

Drives the Raspberry Pi AI Camera (IMX500, CSI) for real-time face detection.
A daemon thread captures metadata, runs the on-chip neural network, and
maintains a thread-safe list of detected faces as normalized rectangles.

All constants come from config.HARDWARE["vision"].
Graceful degradation: if picamera2/IMX500 are unavailable (e.g. on a dev
laptop), init() returns False and get_faces() always returns [].
"""

import logging
import threading
import time
from dataclasses import dataclass

from config import HARDWARE

log = logging.getLogger("botijo.vision")

# ---------------------------------------------------------------------------
# Hardware abstraction
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    _HAS_HARDWARE = True
except ImportError:
    _HAS_HARDWARE = False
    log.info("picamera2/IMX500 not available — vision module will run in stub mode")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = HARDWARE["vision"]

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_picam2 = None
_imx500 = None
_shutdown = threading.Event()
_detect_thread = None
_faces = []
_faces_lock = threading.Lock()
_initialized = False


@dataclass
class FaceRect:
    """A detected face as a normalized rectangle."""
    x: float  # center x, normalized 0-1
    y: float  # center y, normalized 0-1
    w: float  # width, normalized 0-1
    h: float  # height, normalized 0-1
    confidence: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_detections(outputs, threshold=0.2):
    """Parse raw IMX500 outputs into a list of detection dicts."""
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
                    "class_id": class_id,
                    "confidence": score,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                })
        return detections
    except Exception as e:
        log.debug("Error processing detections: %s", e)
        return []


def _detection_loop():
    """Daemon thread: capture metadata → run detection → update _faces list."""
    global _faces

    log.info("Detection thread started")
    threshold = _cfg.get("detection_threshold", 0.2)
    consecutive_no_outputs = 0

    # Grab labels once
    intrinsics = _imx500.network_intrinsics or NetworkIntrinsics()
    if not hasattr(intrinsics, "labels") or not intrinsics.labels:
        intrinsics.labels = ["face"]

    while not _shutdown.is_set():
        try:
            if _picam2 is None or _imx500 is None:
                _shutdown.wait(0.1)
                continue

            # Capture metadata from the camera
            try:
                metadata = _picam2.capture_metadata(wait=True)
            except Exception as e:
                log.debug("Error capturing metadata: %s", e)
                _shutdown.wait(0.1)
                continue

            # Get neural network outputs
            outputs = _imx500.get_outputs(metadata, add_batch=True)
            if outputs is None:
                consecutive_no_outputs += 1
                if consecutive_no_outputs > 100:
                    log.warning("Too many consecutive no-output frames, pausing 1s")
                    _shutdown.wait(1)
                    consecutive_no_outputs = 0
                else:
                    _shutdown.wait(0.05)
                continue

            consecutive_no_outputs = 0

            # Process detections, filter for faces
            detections = _process_detections(outputs, threshold=threshold)
            faces = [d for d in detections
                     if d["class_id"] < len(intrinsics.labels)
                     and intrinsics.labels[d["class_id"]] == "face"]

            if not faces:
                with _faces_lock:
                    _faces = []
                _shutdown.wait(0.05)
                continue

            # Convert to normalized FaceRect list
            input_w, input_h = _imx500.get_input_size()
            result = []
            for det in faces:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2.0 / input_w
                cy = (y1 + y2) / 2.0 / input_h
                w = abs(x2 - x1) / input_w
                h = abs(y2 - y1) / input_h
                result.append(FaceRect(
                    x=max(0.0, min(1.0, cx)),
                    y=max(0.0, min(1.0, cy)),
                    w=max(0.0, min(1.0, w)),
                    h=max(0.0, min(1.0, h)),
                    confidence=det["confidence"],
                ))

            # Sort by confidence descending (best face first)
            result.sort(key=lambda f: f.confidence, reverse=True)

            with _faces_lock:
                _faces = result

            # ~20 FPS
            _shutdown.wait(0.05)

        except Exception as e:
            log.warning("Detection loop error: %s", e)
            _shutdown.wait(0.5)

    log.info("Detection thread stopped")


# ---------------------------------------------------------------------------
# Camera init / shutdown helpers
# ---------------------------------------------------------------------------

def _init_camera():
    """Initialize IMX500 + Picamera2. Returns True on success."""
    global _picam2, _imx500

    rpk_path = _cfg.get("rpk_path", "/home/jack/botijo/packett/network.rpk")
    labels_path = _cfg.get("labels_path", "/home/jack/botijo/labels.txt")
    threshold = _cfg.get("detection_threshold", 0.2)

    try:
        log.info("Loading IMX500 model from %s", rpk_path)
        _imx500 = IMX500(rpk_path)

        # Configure network intrinsics
        intrinsics = _imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.threshold = threshold
        intrinsics.iou = 0.5
        intrinsics.max_detections = 5

        # Load labels
        try:
            with open(labels_path) as f:
                intrinsics.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            intrinsics.labels = ["face"]
            log.warning("Labels file not found at %s, defaulting to ['face']", labels_path)

        intrinsics.update_with_defaults()

        # Configure camera
        _picam2 = Picamera2(_imx500.camera_num)
        config = _picam2.create_preview_configuration(
            buffer_count=12,
            controls={"FrameRate": intrinsics.inference_rate},
        )
        _picam2.configure(config)

        log.info("Loading AI firmware...")
        _imx500.show_network_fw_progress_bar()

        _picam2.start()

        # Allow AI to stabilize
        log.info("Waiting for AI stabilization (3s)...")
        time.sleep(3)

        log.info("Camera system initialized")
        return True

    except Exception as e:
        log.error("Failed to initialize camera: %s", e)
        _picam2 = None
        _imx500 = None
        return False


def _shutdown_camera():
    """Stop camera and release resources."""
    global _picam2, _imx500

    try:
        if _picam2 is not None:
            _picam2.stop()
            _picam2 = None
        _imx500 = None
        log.info("Camera system shut down")
    except Exception as e:
        log.warning("Error shutting down camera: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init() -> bool:
    """Initialize IMX500 + Picamera2 and start the detection thread.

    Returns True if hardware is available and initialized, False otherwise
    (stub mode — get_faces() will always return []).
    """
    global _initialized, _detect_thread

    if _initialized:
        log.warning("vision.init() called twice")
        return _picam2 is not None

    if not _HAS_HARDWARE:
        log.warning("No vision hardware — running in stub mode")
        _initialized = True
        return False

    if not _init_camera():
        _initialized = True
        return False

    # Start detection thread
    _shutdown.clear()
    _detect_thread = threading.Thread(target=_detection_loop, daemon=True, name="vision-detect")
    _detect_thread.start()

    _initialized = True
    log.info("Vision module initialized — face detection active")
    return True


def get_faces() -> list[FaceRect]:
    """Return the current list of detected faces (thread-safe).

    Faces are sorted by confidence (highest first). Each FaceRect has
    normalized coordinates (0.0-1.0).
    """
    with _faces_lock:
        return list(_faces)


def cleanup():
    """Stop the detection thread and release the camera."""
    global _initialized

    if not _initialized:
        return

    log.info("Vision cleanup starting...")

    # Signal thread to stop
    _shutdown.set()

    # Wait for thread
    if _detect_thread is not None and _detect_thread.is_alive():
        _detect_thread.join(timeout=3)

    # Shut down camera
    _shutdown_camera()

    _initialized = False
    log.info("Vision cleanup complete")
