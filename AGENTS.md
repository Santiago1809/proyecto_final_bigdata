# AGENTS.md — Bottle Detection Servo Scan

## Entry point

```bash
python -m src.vision.main                        # OpenCV DNN (default)
python -m src.vision.main --tf                    # TF classifier
python -m src.vision.main --tf --model path/to/model.h5
python -m src.vision.main --test                  # no serial — logs only
python -m src.vision.main --tf --test             # TF test mode
python -m src.vision.main --camera 2 --threshold 0.75
```

Invoke via `-m` (not as a script path). Standard `argparse`, no dotenv.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model files:
- **OpenCV DNN**: `models/MobileNetSSD_deploy.caffemodel` + `models/MobileNetSSD_deploy.prototxt` — NOT in the repo. Download from OpenCV model zoo.
- **TF classifier** (`--tf` flag): `models/bottle_classifier.h5` — produced by `training/train.py`. NOT in the repo.

Tests mock the DNN net and TF model — no model files needed.

## Tests

Uses stdlib `unittest` (no pytest).

```bash
python -m unittest discover tests -v       # all tests
python -m unittest tests.test_message       # single file
python -m unittest tests.test_classifier
```

Classifier tests mock `cv2.dnn.Net` and `tf.keras.models.load_model` — no camera or model files required.

## Architecture

Python host + ESP32 firmware living side-by-side in the same repo.

```
src/
├── vision/          # host-side Python vision pipeline
│   ├── main.py      # → orchestrator entry point
│   ├── capture.py   # → OpenCV VideoCapture wrapper
│   ├── preprocess.py# → resize + DNN blob + TF preprocess
│   ├── classifier.py# → MobileNet SSD (VOC class 5, bottle) [legacy]
│   └── classifier_tf.py# → 3-class TF classifier (Pool/Hatsu/No bottle) [new]
├── protocol/        # shared serial protocol
│   └── message.py   # → {"b":1,"t":1,"s":90}\n encode/decode
└── hardware/esp32/  # Arduino sketch
    ├── firmware.ino
    ├── led_control.h
    └── servo_sweep.h

models/              # place .caffemodel + .prototxt here (not committed)
training/            # TF training pipeline (standalone)
├── config.py
├── dataset.py
├── model.py
├── train.py
└── requirements-train.txt
```

All Python imports use absolute `from src.xxx` — always run commands from the repo root.

## ESP32 firmware

Arduino sketch in `src/hardware/esp32/`. Open `firmware.ino` in the Arduino IDE, select your ESP32 board, and flash via USB. 9600 baud.

No ArduinoJson dependency — the parser hand-walks `strstr` for `"b"` and reads the value after the colon. JSON format: `{"b":1}\n` (green) / `{"b":0}\n` (red). The ``"t"`` (bottle type) and ``"s"`` (servo angle) fields are present in Python output but safely ignored by the C parser.

## Hardware wiring

| Component | ESP32 pin | Note |
|-----------|-----------|------|
| Green LED | GPIO 26 + 220Ω | to GND |
| Red LED   | GPIO 27 + 220Ω | to GND |
| Servo SG90| GPIO 13 signal | external 5V supply required |

## Quirks & constraints

- **Serial**: auto-detects ESP32 by VID/PID keywords (CP210, CH340, SiLabs). Falls back to first USB-serial device. Retries every 2 s if disconnected.
- **Servo power**: DO NOT draw servo current from the ESP32 USB port — use an external 5V supply.
- **Autonomous fallback**: ESP32 keeps sweeping if no serial command for ≥5 s; both LEDs turn off.
- **.gitignore**: only ignores `.atl/`. Model files and `.venv/` should not be committed.
- **No CI, no linter, no formatter config** in the repo. No `setup.py` or `pyproject.toml`.
