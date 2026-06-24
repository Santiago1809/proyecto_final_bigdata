"""Vision pipeline orchestrator.

Captures frames from a camera, runs bottle classification, and
dispatches results over USB serial to the ESP32.

Usage:
    python -m src.vision.main [--port /dev/ttyUSB0] [--threshold 0.7]

Press Ctrl+C to stop.
"""

import argparse
import logging
import time

import serial
import serial.tools.list_ports

from src.protocol.message import encode
from src.vision.capture import CameraCapture
from src.vision.classifier import BottleClassifier

_LOG = logging.getLogger("bottle-detection")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

_SERIAL_RETRY_DELAY = 2.0  # seconds between reconnection attempts
_SERIAL_TIMEOUT = 1.0
_MODEL_DIR = "models"
_CAFFEMODEL = f"{_MODEL_DIR}/MobileNetSSD_deploy.caffemodel"
_PROTOTXT = f"{_MODEL_DIR}/MobileNetSSD_deploy.prototxt"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bottle detection servo scan")
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
        help="Run in test mode (no serial, log only)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default 0)",
    )
    return parser


def _auto_detect_port() -> str | None:
    """Attempt to find an ESP32 serial port automatically."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # Common ESP32 VID/PID or description keywords
        if any(kw in (p.description or "").lower() for kw in ("cp210", "ch340", "silab", "esp32")):
            return p.device
    # Fallback: first available USB-serial device
    for p in ports:
        if "usb" in (p.description or "").lower() or "serial" in (p.description or "").lower():
            return p.device
    return None


def _open_serial(port: str | None) -> serial.Serial | None:
    """Open a serial connection, retrying if the port is not available.

    Args:
        port: Device path or None for auto-detect.

    Returns:
        Open Serial instance, or None if no port is found.

    """
    if port is None:
        port = _auto_detect_port()
    if port is None:
        return None

    try:
        ser = serial.Serial(
            port=port,
            baudrate=9600,
            timeout=_SERIAL_TIMEOUT,
            write_timeout=_SERIAL_TIMEOUT,
        )
        _LOG.info("Connected to ESP32 on %s", port)
        return ser
    except (serial.SerialException, OSError) as exc:
        _LOG.warning("Cannot open port %s: %s", port, exc)
        return None


def run(args: argparse.Namespace) -> None:
    """Main classification loop.

    Captures frames, runs inference, and optionally dispatches results
    over serial. Handles serial disconnection with automatic retry.

    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    _LOG.info("Loading model (threshold=%.2f) ...", args.threshold)
    classifier = BottleClassifier(threshold=args.threshold)
    classifier.load_model(_CAFFEMODEL, _PROTOTXT)
    _LOG.info("Model loaded successfully.")

    serial_port: serial.Serial | None = None if args.test else _open_serial(args.port)

    if serial_port is None and not args.test:
        _LOG.info("No serial port available — will retry every %.0fs", _SERIAL_RETRY_DELAY)

    with CameraCapture(source=args.camera) as camera:
        _LOG.info("Camera opened (640x480). Running inference loop. Press Ctrl+C to stop.")

        while True:
            success, frame = camera.read()
            if not success:
                _LOG.warning("Failed to read frame — skipping")
                time.sleep(0.1)
                continue

            is_bottle, confidence = classifier.predict(frame)
            label = "BOTTLE" if is_bottle else "NOT BOTTLE"
            _LOG.info("%s (confidence=%.3f)", label, confidence)

            # Serial dispatch
            if not args.test:
                message = encode(is_bottle)
                if serial_port is not None:
                    try:
                        serial_port.write(message)
                    except (serial.SerialException, OSError) as exc:
                        _LOG.error("Serial write failed: %s", exc)
                        serial_port.close()
                        serial_port = None

                # Retry connection if disconnected
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


if __name__ == "__main__":
    main()
