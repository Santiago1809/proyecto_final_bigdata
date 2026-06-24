# Bottle Detection Servo Scan

Vision-based system that detects plastic bottles in a camera feed and signals an ESP32 to drive green/red LEDs while running a continuous 180° servo sweep.

## System Overview

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Camera      │     │  Vision Host       │     │  ESP32            │
│  (USB)       │     │  (Python)          │     │                   │
│              │     │                    │     │  ┌──────────────┐ │
│  ── frame ──>│     │  capture.py        │     │  │ Timer ISR    │ │
│              │     │  → preprocess.py    │     │  │ → sweep +1°  │ │
│              │     │  → classifier.py    │     │  │   per 167ms  │ │
│              │     │  → conf ≥ 0.7?      │     │  └──────────────┘ │
│              │     │  → message.encode   │     │                   │
│              │     │  → serial write     │     │  ┌──────────────┐ │
│              │     │         │           │     │  │ Serial       │ │
│              │     │         │ USB       │     │  │ → parse JSON │ │
│              │     │         v           │     │  │ → set LED    │ │
│              │     │  {"b":1} ──────────>│     │  └──────────────┘ │
│              │     │  {"b":0} ──────────>│     │                   │
└──────────────┘     └───────────────────┘     └──────────────────┘
```

## Components

### Vision Pipeline (Python / Host)

| Module | File | Purpose |
|--------|------|---------|
| Capture | `src/vision/capture.py` | OpenCV VideoCapture wrapper, 640×480 BGR frames |
| Preprocess | `src/vision/preprocess.py` | 300×300 resize + DNN blob conversion |
| Classifier | `src/vision/classifier.py` | MobileNet SSD inference, COCO class 39 (bottle), configurable threshold |
| Orchestrator | `src/vision/main.py` | Capture → classify → serial dispatch loop |

### ESP32 Firmware (C++)

| Module | File | Purpose |
|--------|------|---------|
| LED Control | `src/hardware/esp32/led_control.h` | Green/red GPIO helpers, standby blink pattern |
| Servo Sweep | `src/hardware/esp32/servo_sweep.h` | LEDC PWM (50 Hz), hardware timer ISR, 1°/167ms sweep |
| Main | `src/hardware/esp32/firmware.ino` | Serial JSON parser, LED dispatch, autonomous fallback |

### Protocol

Compact JSON over USB serial at **9600 baud**:

| Event | Message |
|-------|---------|
| Bottle detected | `{"b":1}\n` |
| No bottle | `{"b":0}\n` |

## Wiring Diagram (ASCII)

```
USB Host ──── USB Cable ──── ESP32 Dev Board
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
           GPIO 26           GPIO 27           GPIO 13
              │                │                 │
           ╔══╧══╗          ╔══╧══╗          ╔══╧════╗
           ║ 220Ω ║          ║ 220Ω ║          ║ SG90  ║
           ╚══╤══╝          ╚══╤══╝          ║ Servo ║
              │                │              ╚══╤════╝
          ┌───┴───┐        ┌───┴───┐             │
          │ Green │        │  Red  │        Signal│(Orange)
          │  LED  │        │  LED  │             │
          └───┬───┘        └───┬───┘         ┌───┴────┐
              │                │             │ 5V GND │
             GND              GND            │(Brown/ │
                                             │ Red)   │
                                             └────────┘
```

**Servo power**: Use an external 5V supply for the SG90. Do NOT draw servo current from the ESP32 USB port.

**LED resistors**: 220Ω current-limiting resistors are required for each LED.

## Setup

### Prerequisites

- Python 3.10+
- ESP32 dev board (ESP32-WROOM-32 or similar)
- USB camera
- SG90 (or equivalent) servomotor
- Green LED, Red LED, 220Ω resistors, breadboard, jumper wires

### Host Installation

```bash
# Clone the repository
git clone <repo-url> && cd bottle-detection-servo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place model files in models/
# Download MobileNetSSD_deploy.caffemodel and MobileNetSSD_deploy.prototxt
# into the models/ directory
```

### Model Files

This system uses MobileNet SSD via OpenCV DNN. You need:

- `models/MobileNetSSD_deploy.caffemodel` (weights)
- `models/MobileNetSSD_deploy.prototxt` (architecture)

These can be obtained from the [OpenCV Model Zoo](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn) or other MobileNet SSD v1 COCO repositories.

## Usage

### Vision Pipeline (with ESP32)

```bash
# Auto-detect ESP32 serial port
python -m src.vision.main

# Specify serial port and custom threshold
python -m src.vision.main --port /dev/ttyUSB0 --threshold 0.75

# Use camera index 2
python -m src.vision.main --camera 2
```

### Test Mode (no serial required)

```bash
python -m src.vision.main --test
```

Runs the full vision pipeline without connecting to an ESP32. Classification results are logged to the console.

### Running Tests

```bash
# Unit tests
python -m unittest discover tests -v

# Specific test file
python -m unittest tests.test_message
python -m unittest tests.test_classifier
```

### ESP32 Firmware

Open `src/hardware/esp32/firmware.ino` in the Arduino IDE (or platformio):

1. Select your ESP32 board.
2. Flash the firmware via USB.
3. Open the Serial Monitor at 9600 baud to verify.

The servo sweep starts automatically on boot. LEDs respond to incoming JSON commands.

## Autonomous Fallback

If the ESP32 receives no serial command for **≥5 seconds**:

- The servo sweep continues running (independent of host).
- Both LEDs turn off until a new command arrives.

This ensures the mechanical system keeps moving even if the host disconnects or crashes.

## Project Structure

```
src/
├── protocol/
│   └── message.py          # JSON encode/decode for serial
├── vision/
│   ├── capture.py           # Camera frame acquisition
│   ├── preprocess.py        # Resize + DNN blob conversion
│   ├── classifier.py        # Bottle detection model wrapper
│   └── main.py              # Orchestration entry point
└── hardware/
    └── esp32/
        ├── led_control.h     # LED GPIO helpers
        ├── servo_sweep.h     # Servo sweep via LEDC + timer
        └── firmware.ino      # Main ESP32 firmware

data/
└── samples/                  # Labeled test images (bottle_/nobottle_)

tests/
├── test_message.py           # Protocol roundtrip tests
└── test_classifier.py        # Classifier threshold boundary tests

models/                       # Place .caffemodel + .prototxt here
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | MobileNet SSD v1 (COCO class 39) | Zero extra ML dep — OpenCV DNN built-in |
| Protocol | Compact JSON `{"b":N}\n` | Human-readable, debuggable, extensible |
| Servo PWM | ESP32 LEDC hardware (50 Hz, 16-bit) | Jitter-free, no library needed |
| Sweep timing | 1°/167ms (~30s full cycle) | Smooth sweep, mechanical stability |
| Confidence threshold | 0.7 (configurable) | Balances precision vs recall (spec default) |
