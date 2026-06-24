"""Vision pipeline orchestrator with live display.

Captures frames from a camera, runs bottle classification, draws
bounding boxes on a live preview window, and dispatches results over
USB serial to the ESP32.

Two classifier modes (selectable via ``--tf``):

- **OpenCV DNN** (default): MobileNet SSD Caffe model, VOC class 5,
  bounding-box based. Falls back gracefully if model files are absent.
- **TensorFlow** (``--tf``): custom 3-class MobileNetV2 classifier,
  no bounding box, type-aware labels (Pool Verde / Hatsu Morado).

Usage::

    python -m src.vision.main                             # OpenCV DNN
    python -m src.vision.main --tf                         # TF classifier
    python -m src.vision.main --tf --model path/to/model.h5
    python -m src.vision.main --test                       # no serial, no display
    python -m src.vision.main --tf --test                  # TF test mode

Press ``q`` or ``ESC`` in the preview window, or ``Ctrl+C`` in the
terminal to stop.
"""

import argparse
import logging
import time

import cv2
import serial
import serial.tools.list_ports

from src.protocol.message import encode, BottleType
from src.vision.capture import CameraCapture
from src.vision.classifier import BottleClassifier

_LOG = logging.getLogger("bottle-detection")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

_SERIAL_RETRY_DELAY = 2.0  # seconds between reconnection attempts
_SERIAL_TIMEOUT = 1.0
_MODEL_DIR = "models"
_CAFFEMODEL = f"{_MODEL_DIR}/MobileNetSSD_deploy.caffemodel"
_PROTOTXT = f"{_MODEL_DIR}/MobileNetSSD_deploy.prototxt"
_TF_DEFAULT_MODEL = f"{_MODEL_DIR}/bottle_classifier.h5"

_WINDOW_NAME = "Bottle Detection"


# ---------------------------------------------------------------------------
# Detect whether OpenCV was built with GUI support (GTK, Cocoa, …)
# ---------------------------------------------------------------------------
_HAS_GUI: bool = False
try:
    cv2.namedWindow("__probe__")
    cv2.destroyWindow("__probe__")
    _HAS_GUI = True
except cv2.error:
    pass


# ---------------------------------------------------------------------------
# Colour palette for the overlay (BGR)
# ---------------------------------------------------------------------------
_COLOR_GREEN  = (0, 230, 120)
_COLOR_RED    = (50, 50, 230)
_COLOR_WHITE  = (255, 255, 255)
_COLOR_BG     = (40, 40, 40)     # dark background behind text

# ---------------------------------------------------------------------------
# Log throttle: only print to terminal when state changes or once per second
# ---------------------------------------------------------------------------
_LOG_INTERVAL = 1.0              # seconds between periodic log entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bottle detection with live camera preview"
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for ESP32 (auto-detect if omitted)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (0..1, default 0.7)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: no serial (add --display to show window)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Force-enable live preview (on by default unless --test is set)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview window (headless mode)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default 0)",
    )
    parser.add_argument(
        "--inference-skip",
        type=int,
        default=2,
        metavar="N",
        help="Run inference every N frames (default 2 = ~2× display FPS)",
    )
    parser.add_argument(
        "--tf",
        action="store_true",
        help="Use TensorFlow classifier instead of OpenCV DNN",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to TF .h5 model (default: {})".format(_TF_DEFAULT_MODEL),
    )
    return parser


def _draw_overlay(
    frame: cv2.Mat,
    is_bottle: bool,
    confidence: float,
    box: tuple[int, int, int, int] | None,
    fps: float = 0.0,
    class_name: str | None = None,
) -> cv2.Mat:
    """Annotate *frame* with bounding box, label, confidence and FPS.

    Args:
        frame: Original BGR frame (modified in-place).
        is_bottle: Whether a bottle was detected.
        confidence: Confidence score.
        box: ``(x1, y1, x2, y2)`` bounding box or ``None``.
        fps: Current frames-per-second for the overlay.
        class_name: Display label (e.g. "Pool Verde"). If ``None``,
            falls back to "BOTTLE" / "NO BOTTLE".

    Returns:
        The annotated frame (same object as *frame*).

    """
    color = _COLOR_GREEN if is_bottle else _COLOR_RED
    label = class_name or ("BOTTLE" if is_bottle else "NO BOTTLE")
    text = f"{label}  {confidence:.1%}"

    # --- Bounding box ---
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - th - 12),
            (x1 + tw + 12, y1 + 6),
            color,
            -1,
        )
        cv2.putText(
            frame, text, (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, _COLOR_WHITE, 2,
        )

    # --- Bottom status bar ---
    h, w = frame.shape[:2]
    bar_text = f"{label}  ({confidence:.1%})  |  Servo: {'90°' if is_bottle else '180°'}"
    cv2.rectangle(frame, (0, h - 36), (w, h), _COLOR_BG, -1)
    cv2.putText(
        frame, bar_text, (12, h - 10),
        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1,
    )

    # --- Top-right FPS counter ---
    fps_text = f"{fps:.0f} FPS"
    cv2.putText(
        frame, fps_text, (w - 118, 28),
        cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1,
    )

    return frame


def _auto_detect_port() -> str | None:
    """Attempt to find an ESP32 serial port automatically."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if any(kw in (p.description or "").lower() for kw in ("cp210", "ch340", "silab", "esp32")):
            return p.device
    for p in ports:
        if "usb" in (p.description or "").lower() or "serial" in (p.description or "").lower():
            return p.device
    return None


def _open_serial(port: str | None) -> serial.Serial | None:
    """Open a serial connection, retrying if the port is not available."""
    if port is None:
        port = _auto_detect_port()
    if port is None:
        return None
    try:
        ser = serial.Serial(
            port=port, baudrate=9600,
            timeout=_SERIAL_TIMEOUT, write_timeout=_SERIAL_TIMEOUT,
        )
        _LOG.info("Connected to ESP32 on %s", port)
        return ser
    except (serial.SerialException, OSError) as exc:
        _LOG.warning("Cannot open port %s: %s", port, exc)
        return None


def run(args: argparse.Namespace) -> None:
    """Main classification loop with live preview & throttled logging.

    Supports two classifier modes selected via ``--tf``:

    - **OpenCV DNN** (default): returns ``(is_bottle, confidence, box)``.
    - **TensorFlow** (``--tf``): returns ``BottlePrediction`` with class
      name and type info.

    Inference is only run every *inference_skip* frames (default 2) to
    keep the display smooth.  The last inference result is shown on
    intermediate frames.
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    show_display = _HAS_GUI and (not args.no_display) and (args.display or not args.test)
    if args.display and not _HAS_GUI:
        _LOG.warning(
            "--display was requested but OpenCV has no GUI support. "
            "Install GTK (apt install opencv-python) or use --no-display."
        )

    skip = max(1, args.inference_skip)
    if skip > 1:
        _LOG.info("Inference skip=%d — display runs faster, inference every %d frames", skip, skip)

    # ------------------------------------------------------------------
    # Classifier initialisation
    # ------------------------------------------------------------------
    use_tf = args.tf

    if use_tf:
        model_path = args.model or _TF_DEFAULT_MODEL
        _LOG.info("Loading TF model (threshold=%.2f) from %s ...", args.threshold, model_path)
        from src.vision.classifier_tf import BottleTFClassifier  # noqa: late import
        classifier = BottleTFClassifier(model_path=model_path, threshold=args.threshold)
        _LOG.info("TF model loaded successfully.")
        classifier_type = "tf"
    else:
        _LOG.info("Loading OpenCV DNN model (threshold=%.2f) ...", args.threshold)
        classifier = BottleClassifier(threshold=args.threshold)
        classifier.load_model(_CAFFEMODEL, _PROTOTXT)
        _LOG.info("OpenCV DNN model loaded successfully.")
        classifier_type = "dnn"

    serial_port: serial.Serial | None = None if args.test else _open_serial(args.port)
    if serial_port is None and not args.test:
        _LOG.info("No serial port available — will retry every %.0fs", _SERIAL_RETRY_DELAY)

    if show_display:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- FPS & inference state ---
    frame_index = 0
    frame_count = 0
    fps_start = time.perf_counter()
    fps_ema = 30.0
    last_bottle_state: bool | None = None
    last_log_time = 0.0

    # Last classification result (reused on skipped frames)
    # Type varies by classifier — initialise with a safe "no detection" value
    if use_tf:
        from src.vision.classifier_tf import BottlePrediction  # noqa: late import
        infer_result = BottlePrediction(class_id=0, confidence=0.0, class_name="No bottle")
    else:
        infer_result = (False, 0.0, None)

    with CameraCapture(source=args.camera) as camera:
        _LOG.info(
            "Camera opened (640x480). Using %s classifier. %s",
            classifier_type,
            "Live preview active — press Q or ESC to quit."
            if show_display else "Running headless. Press Ctrl+C to stop.",
        )

        while True:
            tick = time.perf_counter()

            # --- FPS tracking ---
            frame_count += 1
            frame_index += 1
            if tick - fps_start >= 1.0:
                fps_ema = 0.9 * fps_ema + 0.1 * frame_count / (tick - fps_start)
                fps_start = tick
                frame_count = 0

            success, frame = camera.read()
            if not success:
                _LOG.warning("Failed to read frame — skipping")
                time.sleep(0.1)
                continue

            # --- Run inference every N frames ---
            run_inference = (frame_index % skip == 0)
            if run_inference:
                infer_result = classifier.predict(frame)

            # --- Unpack result by classifier type ---
            if classifier_type == "tf":
                pred = infer_result
                is_bottle = pred.class_id != 0
                confidence = pred.confidence
                bottle_type = pred.class_id  # class_id matches BottleType
                class_name = pred.class_name
                box = None  # TF classifier has no bounding box
            else:
                is_bottle, confidence, box = infer_result
                bottle_type = int(is_bottle)  # 1 for any bottle, 0 for none
                class_name = None

            servo_angle = 90 if is_bottle else 180

            # --- Throttled logging ---
            now = tick
            state_changed = is_bottle != last_bottle_state
            if state_changed or (now - last_log_time) >= _LOG_INTERVAL:
                label = class_name or ("BOTTLE" if is_bottle else "NOT BOTTLE")
                _LOG.info(
                    "%s (confidence=%.3f) servo=%d° type=%d%s",
                    label, confidence, servo_angle, bottle_type,
                    f" box={box}" if box else "",
                )
                last_log_time = now
            last_bottle_state = is_bottle

            # --- Live preview ---
            if show_display:
                display = _draw_overlay(
                    frame, is_bottle, confidence, box, fps_ema,
                    class_name=class_name,
                )
                cv2.imshow(_WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    _LOG.info("Preview window closed by user")
                    break

            # --- Serial dispatch (only on inference frames) ---
            if run_inference and not args.test:
                message = encode(is_bottle, bottle_type=bottle_type, servo_angle=servo_angle)
                if serial_port is not None:
                    try:
                        serial_port.write(message)
                    except (serial.SerialException, OSError) as exc:
                        _LOG.error("Serial write failed: %s", exc)
                        serial_port.close()
                        serial_port = None

                if serial_port is None:
                    serial_port = _open_serial(args.port)
                    if serial_port is not None:
                        _LOG.info("Reconnected to ESP32")
                    else:
                        time.sleep(_SERIAL_RETRY_DELAY)


def main() -> None:
    """Entry point for ``python -m src.vision.main``."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        _LOG.info("Shutting down by user request")
    finally:
        if _HAS_GUI:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
