#!/usr/bin/env python3
"""Dataset capture tool — grab training images from the PC webcam.

Usage::

    python -m tools.capture_dataset
    python -m tools.capture_dataset --camera 2
    python -m tools.capture_dataset --roi-size 224

Opens the PC webcam, shows a live preview with a centred green rectangle.
Press a class key to save the ROI to the corresponding dataset directory:

    Key  │  Class
    ─────┼─────────────────────
     1   │  no_bottle
     2   │  pool_verde
     3   │  hatsu_morado
     r   │  Delete ALL captures from this session
     q   │  Quit

The crop inside the green rectangle is resized to 224×224 and saved with a
timestamp filename under ``training/data/<class>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Make the project root importable when running as `python -m tools.capture_dataset`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.vision.capture import CameraCapture  # noqa: E402

_LOG = logging.getLogger("capture-dataset")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(_project_root) / "training" / "data"
CLASS_MAP: dict[int, str] = {
    ord("1"): "no_bottle",
    ord("2"): "pool_verde",
    ord("3"): "hatsu_morado",
}
CLASS_NAMES_SHORT = {ord("1"): "no_bottle", ord("2"): "Pool", ord("3"): "Hatsu"}
WINDOW_NAME = "Captura de dataset"

# Colours (BGR)
_COL_GREEN = (0, 230, 120)
_COL_WHITE = (255, 255, 255)
_COL_YELLOW = (0, 255, 255)
_COL_BG = (40, 40, 40)
_COL_RED = (50, 50, 230)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create class subdirectories if they don't exist."""
    for class_name in CLASS_MAP.values():
        (DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)
    _LOG.info("Dataset directory: %s", DATA_DIR)


def _make_filename(class_name: str, count: int) -> str:
    """Generate a timestamped, sortable filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{class_name}_{count:04d}.jpg"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture training images from PC webcam",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default 0)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=300,
        help="Side length of the green capture rectangle in pixels (default 300)",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=224,
        help="Final saved image size in pixels (default 224)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width (default 640)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Open camera, show preview, save captures on keypress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    roi_size = args.roi_size
    resize_to = args.resize_to
    cam_width = args.width
    cam_height = int(cam_width * 3 / 4)  # 4:3 aspect

    _ensure_dirs()

    # Counters per class
    counters: dict[str, int] = {name: 0 for name in CLASS_MAP.values()}
    # Count how many frames we've already saved in this session (for filenames)
    session_count = 0
    # Track saved file paths so [R] can roll them back
    saved_paths: list[str] = []

    last_saved_msg = ""
    last_saved_time = 0.0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        _LOG.error("Cannot open camera index %d", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.read()  # warm-up

    # Centre the ROI on the frame
    x_start = (cam_width - roi_size) // 2
    y_start = (cam_height - roi_size) // 2

    _LOG.info(
        "Camera %d opened (%dx%d). ROI: %d×%d at (%d, %d). Press 1/2/3 to capture, r to reject, q to quit.",
        args.camera, cam_width, cam_height, roi_size, roi_size, x_start, y_start,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            _LOG.warning("Lost camera feed — retrying…")
            time.sleep(0.5)
            continue

        display = frame.copy()

        # ── Draw the ROI rectangle ──
        cv2.rectangle(
            display,
            (x_start, y_start),
            (x_start + roi_size, y_start + roi_size),
            _COL_GREEN,
            2,
        )

        # ── Top-left instructions ──
        lines = [
            ("[1] No bottle", _COL_WHITE),
            ("[2] Pool Verde", _COL_WHITE),
            ("[3] Hatsu Morado", _COL_WHITE),
            ("[R] Borrar TODAS   [Q] Salir", _COL_RED),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                display, text, (10, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )

        # ── Top-right counters ──
        cx = cam_width - 180
        cv2.putText(
            display, "Captures", (cx, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_YELLOW, 1,
        )
        for i, (cls_name, short) in enumerate(CLASS_NAMES_SHORT.items()):
            count = counters[CLASS_MAP[cls_name]]
            text = f"{short}: {count}"
            cv2.putText(
                display, text, (cx, 58 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL_WHITE, 1,
            )

        # ── Last-saved toast ──
        if last_saved_msg and time.time() - last_saved_time < 2.0:
            cv2.putText(
                display, last_saved_msg, (10, cam_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL_GREEN, 1,
            )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        # ── Handle keypress ──
        if key == ord("q"):
            _LOG.info("Capture session ended by user.")
            break

        if key == ord("r"):
            if saved_paths:
                deleted = 0
                for p in saved_paths:
                    try:
                        Path(p).unlink()
                        deleted += 1
                    except OSError:
                        pass
                _LOG.warning("Borradas %d capturas de esta sesión", deleted)
                saved_paths.clear()
                for k in counters:
                    counters[k] = 0
                session_count = 0
                last_saved_msg = f"BORRADAS {deleted} capturas"
                last_saved_time = time.time()
            else:
                _LOG.debug("Nada que borrar.")
            continue

        if key in CLASS_MAP:
            class_name = CLASS_MAP[key]
            # Crop and save
            roi = frame[y_start: y_start + roi_size, x_start: x_start + roi_size]
            roi_resized = cv2.resize(roi, (resize_to, resize_to))
            filename = _make_filename(class_name, counters[class_name])
            save_path = DATA_DIR / class_name / filename
            cv2.imwrite(str(save_path), roi_resized)

            saved_paths.append(str(save_path))

            counters[class_name] += 1
            session_count += 1

            last_saved_msg = f"Saved {class_name}/{filename}"
            last_saved_time = time.time()

            _LOG.info("[%d/%d] %s", counters[class_name], session_count, last_saved_msg)

    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ──
    total = sum(counters.values())
    _LOG.info("=" * 50)
    _LOG.info("Session complete — %d total captures", total)
    for cls_name in CLASS_MAP.values():
        _LOG.info("  %s: %d", cls_name, counters[cls_name])
    _LOG.info("=" * 50)


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        _LOG.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
