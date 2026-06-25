# Bottle Detection — Clasificador de Envases

Sistema de visión artificial que detecta y clasifica botellas plásticas en tiempo real usando **TensorFlow + MobileNetV2** con una webcam del PC, y controla un **ESP32** con LEDs, servomotor y zumbador.

Clasifica en 3 clases: `no_bottle` (fondo), `pool_verde`, `hatsu_morado`.

> 📖 Para documentación **exhaustiva** de cada módulo, flag y configuración, ver [`DOCUMENTACION_PROYECTO.md`](DOCUMENTACION_PROYECTO.md).

---

## Arquitectura

```
Webcam PC ──> Python Host ──serial──> ESP32
                  │                    │
            ┌─────┴─────┐        ┌─────┴─────┐
            │MobileNetV2 │        │LEDs G+R   │
            │3 clases    │        │Servo SG90 │
            │Feature     │        │Buzzer     │
            │rejection   │        └───────────┘
            └────────────┘
```

**Host (Python):** captura frames → preprocesa → clasifica con MobileNetV2 → aplica rechazo por distancia de características → envía JSON por serial → registra detecciones en Excel.

**ESP32 (C++):** recibe JSON → enciende LED verde/rojo → posiciona servo → emite pitido.

---

## Setup Rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El modelo entrenado va en `models/bottle_classifier_latest.keras`. Se produce con el pipeline de entrenamiento (ver más abajo).

---

## Uso en Producción

```bash
# Ejecutar con cámara 0, threshold 0.7, auto-detecta ESP32
python -m src.vision.main

# Cámara específica y threshold más exigente
python -m src.vision.main --camera 2 --threshold 0.85

# Modo test (sin ESP32)
python -m src.vision.main --test --display

# Ajustar sensibilidad del rechazo OOD
python -m src.vision.main --rejection-sigma 4.0
```

### Flags principales

| Flag | Default | Descripción |
|------|---------|-------------|
| `--camera` | `0` | Índice de webcam del PC |
| `--threshold` | `0.7` | Confianza mínima [0-1] |
| `--rejection-sigma` | `6.0` | Sensibilidad del rechazo OOD (0 = desactivado) |
| `--test` | — | Sin conexión serial |
| `--display` | — | Forzar ventana de preview |
| `--model` | `models/bottle_classifier_latest.keras` | Ruta al modelo |

Tecla `q` o `ESC` para salir.

---

## Entrenamiento

El pipeline está en `training/`. Requiere dependencias adicionales:

```bash
pip install -r training/requirements-train.txt
```

### Dataset

Colocar imágenes en `training/data/<clase>/`:

```
training/data/
├── no_bottle/       # ~400 imágenes de fondo
├── pool_verde/      # ~750 imágenes Pool Verde
└── hatsu_morado/    # ~750 imágenes Hatsu Morado
```

### Capturar dataset desde la webcam

```bash
python -m tools.capture_dataset
```

Teclas: `1`=no_bottle, `2`=pool_verde, `3`=hatsu_morado, `r`=borrar todas, `q`=salir.

### Entrenar modelo

```bash
python training/train.py --data-dir training/data
```

Entrenamiento en dos fases: backbone congelado (20 épocas, LR 5e-4) → fine-tuning últimas 30 capas (15 épocas, LR 1e-5).

El modelo se guarda en `models/bottle_classifier_latest.keras`.

### Extraer centroides (rechazo OOD)

```bash
python -m tools.extract_centroids \
    --model models/bottle_classifier_latest.keras \
    --data-dir training/data
```

Produce `models/bottle_classifier_latest_centroids.json` para el sistema de rechazo por distancia de características.

---

## Tests

```bash
# Todos
python -m unittest discover tests -v

# Individuales
python -m unittest tests.test_message
python -m unittest tests.test_classifier
python -m unittest tests.test_excel_logger
```

---

## ESP32 — Firmware

Abrir `src/hardware/esp32/esp32.ino` en Arduino IDE, seleccionar placa ESP32 y flashear.

### Pines

| Componente | GPIO | Nota |
|------------|------|------|
| LED Verde | 2 | Con resistencia 220Ω a GND |
| LED Rojo | 5 | Con resistencia 220Ω a GND |
| Servo SG90 | 14 | Señal naranja; **alimentación 5V externa** |
| Buzzer activo | 27 | Señal; GND al ESP32 |

**⚠️ No alimentar el servo desde el USB del ESP32** — usar fuente externa de 5V.

### Protocolo serial (9600 baud)

```json
{"b":1,"t":1,"s":90}   // Pool Verde detectado → LED verde, servo 90°
{"b":1,"t":2,"s":90}   // Hatsu Morado detectado → LED verde, servo 90°
{"b":0,"t":0,"s":180}  // Sin botella → LED rojo, servo 180°
```

Si no recibe comandos por ≥5 segundos, los LEDs se apagan (fallback autónomo).

---

## Estructura del Proyecto

```
src/
├── protocol/
│   └── message.py              # JSON encode/decode serial
├── vision/
│   ├── main.py                 # Orquestador (captura → clasifica → envía)
│   ├── capture.py              # Wrapper de cámara OpenCV
│   ├── classifier_tf.py        # Clasificador MobileNetV2 + rechazo OOD
│   ├── preprocess.py           # Preprocesamiento 224×224 RGB
│   └── excel_logger.py         # Registro de detecciones en Excel
└── hardware/esp32/
    ├── esp32.ino               # Firmware principal
    ├── led_control.h           # Control de LEDs
    ├── servo_control.h         # Control de servomotor (PWM LEDC)
    └── buzzer_control.h        # Control de zumbador no bloqueante

training/
├── train.py                    # Entrenamiento en dos fases
├── model.py                    # Arquitectura MobileNetV2
├── dataset.py                  # Carga + aumento de datos (9 transformaciones)
├── config.py                   # Hiperparámetros
└── data/                       # Dataset (no commiteado)

tools/
├── capture_dataset.py          # Captura de dataset desde webcam
└── extract_centroids.py        # Extracción de centroides para rechazo

tests/
├── test_message.py             # Tests del protocolo serial
├── test_classifier.py          # Tests del clasificador
└── test_excel_logger.py        # Tests del logger Excel
```

---

## Librerías

### Producción (`requirements.txt`)

| Librería | Versión | Uso |
|----------|---------|-----|
| `opencv-python` | ≥4.8 | Captura y procesamiento de imágenes |
| `numpy` | ≥1.24 | Operaciones numéricas |
| `pyserial` | ≥3.5 | Comunicación serial con ESP32 |
| `tensorflow-cpu` | ≥2.13 | Inferencia MobileNetV2 |
| `pandas` | ≥2.0 | Logging a Excel |
| `openpyxl` | ≥3.1 | Escritura de archivos Excel |
| `psutil` | ≥5.9 | Detección de RAM (batch size automático) |

### Entrenamiento (`training/requirements-train.txt`)

Además agrega `scikit-learn` (split estratificado) y `pillow`.
