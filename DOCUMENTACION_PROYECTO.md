# Documentación Completa — Sistema de Detección de Envases

## Índice

1. [Resumen del Proyecto](#1-resumen-del-proyecto)
2. [Arquitectura General](#2-arquitectura-general)
3. [Estructura del Repositorio](#3-estructura-del-repositorio)
4. [Pipeline de Visión (Python)](#4-pipeline-de-visión-python)
   - 4.1 [main.py — Orquestador](#41-mainpy--orquestador)
   - 4.2 [classifier_tf.py — Clasificador MobileNetV2](#42-classifier_tfpy--clasificador-mobilenetv2)
   - 4.3 [capture.py — Captura de Cámara](#43-capturepy--captura-de-cámara)
   - 4.4 [preprocess.py — Preprocesamiento de Imágenes](#44-preprocesspy--preprocesamiento-de-imágenes)
   - 4.5 [excel_logger.py — Registro en Excel](#45-excel_loggerpy--registro-en-excel)
5. [Protocolo Serial](#5-protocolo-serial)
6. [Pipeline de Entrenamiento](#6-pipeline-de-entrenamiento)
   - 6.1 [train.py — Punto de Entrada](#61-trainpy--punto-de-entrada)
   - 6.2 [model.py — Arquitectura del Modelo](#62-modelpy--arquitectura-del-modelo)
   - 6.3 [dataset.py — Carga y Aumento de Datos](#63-datasetpy--carga-y-aumento-de-datos)
   - 6.4 [config.py — Configuración e Hiperparámetros](#64-configpy--configuración-e-hiperparámetros)
7. [Herramientas](#7-herramientas)
   - 7.1 [capture_dataset.py — Captura de Dataset](#71-capture_datasetpy--captura-de-dataset)
   - 7.2 [extract_centroids.py — Extracción de Centroides](#72-extract_centroidspy--extracción-de-centroides)
8. [Firmware ESP32 (C++)](#8-firmware-esp32-c)
   - 8.1 [esp32.ino — Firmware Principal](#81-esp32ino--firmware-principal)
   - 8.2 [led_control.h — Control de LEDs](#82-led_controlh--control-de-leds)
   - 8.3 [servo_control.h — Control del Servomotor](#83-servo_controlh--control-del-servomotor)
   - 8.4 [buzzer_control.h — Control del Zumbador](#84-buzzer_controlh--control-del-zumbador)
9. [Tests](#9-tests)
10. [Hardware: Conexiones Eléctricas](#10-hardware-conexiones-eléctricas)
11. [Glosario de Flags y Parámetros](#11-glosario-de-flags-y-parámetros)
12. [Requerimientos del Sistema](#12-requerimientos-del-sistema)

---

## 1. Resumen del Proyecto

Sistema de **visión por computadora** que detecta envases plásticos en tiempo real usando la webcam del PC y un clasificador basado en **MobileNetV2** (TensorFlow). Cuando detecta una botella, envía comandos JSON por puerto serie a un **ESP32** que controla:

- **LED verde** cuando hay una botella detectada
- **LED rojo** cuando no hay botella
- **Servomotor** que se posiciona según el tipo de botella
- **Zumbador** que emite un pitido breve al detectar

El sistema clasifica en **3 clases**:

| ID  | Clase          | Descripción                          |
|-----|----------------|--------------------------------------|
| 0   | `no_bottle`    | No hay botella (fondo, manos, etc.) |
| 1   | `pool_verde`   | Botella Pool Verde                   |
| 2   | `hatsu_morado` | Botella Hatsu Morado                 |

Incluye un pipeline completo de **entrenamiento** con transfer learning (MobileNetV2 pre-entrenado en ImageNet), aumento de datos, fine-tuning, y un sistema de **rechazo por espacio de características** para reducir falsos positivos.

---

## 2. Arquitectura General

```
┌─────────────────────────────────────────────────────────┐
│                     PC (Linux)                           │
│                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Webcam   │   │  main.py     │   │  protocol/       │  │
│  │ del PC   │──>│  (orquesta)  │──>│  message.py      │  │
│  │          │   │              │   │  encode → JSON   │  │
│  │ 640×480  │   │  ┌─────────┐ │   └────────┬────────┘  │
│  │ BGR      │   │  │classifier│ │            │           │
│  └──────────┘   │  │_tf.py   │ │           USB          │
│                 │  └─────────┘ │          Serie          │
│                 │  ┌─────────┐ │            │           │
│                 │  │excel    │ │            v           │
│                 │  │_logger  │ │   ┌────────────────┐   │
│                 │  └─────────┘ │   │    ESP32        │   │
│                 └──────────────┘   │                │   │
│                                    │  ┌──────────┐  │   │
│  ┌──────────────────────┐          │  │ LEDs     │  │   │
│  │  training/           │          │  │ (G+R)    │  │   │
│  │  train.py            │          │  ├──────────┤  │   │
│  │  model.py            │          │  │ Servo    │  │   │
│  │  dataset.py          │          │  │ SG90     │  │   │
│  │  config.py           │          │  ├──────────┤  │   │
│  └──────────────────────┘          │  │ Buzzer   │  │   │
│                                    │  └──────────┘  │   │
│  ┌──────────────────────┐          └────────────────┘   │
│  │  tools/              │                                │
│  │  capture_dataset.py  │                                │
│  │  extract_centroids.py│                                │
│  └──────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

**Flujo de datos en producción:**

1. `CameraCapture` abre la webcam del PC y entrega frames 640×480 BGR
2. `BottleTFClassifier.predict(frame)` preprocesa la imagen (224×224, RGB), la pasa por MobileNetV2, obtiene probabilidades softmax (3 clases)
3. Si la confianza supera el threshold (ej: 0.7) y no es rechazada por distancia de características, se considera detección positiva
4. `encode()` en `message.py` arma un JSON compacto: `{"b":1,"t":1,"s":90}\n`
5. Se envía por serial al ESP32 a 9600 baud
6. El ESP32 parsea el JSON, enciende LED verde, posiciona el servo a 90°, suena el buzzer 200ms
7. `ExcelLogger` registra cada detección en un archivo Excel (`registro_envases.xlsx`)

---

## 3. Estructura del Repositorio

```
/
├── .gitignore                        # Archivos ignorados por git
├── AGENTS.md                         # Instrucciones para el asistente AI
├── README.md                         # Documentación de inicio rápido
├── requirements.txt                  # Dependencias de producción
├── registro_envases.xlsx             # Bitácora de detecciones (se genera automáticamente)
│
├── src/                              # Código fuente principal
│   ├── protocol/
│   │   └── message.py                # Protocolo serial JSON
│   ├── vision/
│   │   ├── __init__.py               # Marca el directorio como paquete Python
│   │   ├── main.py                   # Orquestador del pipeline
│   │   ├── capture.py                # Wrapper de OpenCV VideoCapture
│   │   ├── classifier_tf.py          # Clasificador MobileNetV2
│   │   ├── preprocess.py             # Preprocesamiento de frames
│   │   └── excel_logger.py           # Logger de detecciones a Excel
│   └── hardware/
│       └── esp32/                    # Firmware del ESP32 (Arduino C++)
│           ├── esp32.ino
│           ├── led_control.h
│           ├── servo_control.h
│           └── buzzer_control.h
│
├── training/                         # Pipeline de entrenamiento
│   ├── __init__.py
│   ├── train.py                      # Entrenamiento en dos fases
│   ├── model.py                      # Constructor del modelo MobileNetV2
│   ├── dataset.py                    # Carga, split y aumento de datos
│   ├── config.py                     # Hiperparámetros
│   ├── requirements-train.txt        # Dependencias de entrenamiento
│   └── data/                         # Dataset de imágenes
│       ├── no_bottle/                # ~398 imágenes sin botella
│       ├── pool_verde/               # ~750 imágenes Pool Verde
│       └── hatsu_morado/             # ~772 imágenes Hatsu Morado
│
├── tools/                            # Herramientas auxiliares
│   ├── __init__.py
│   ├── capture_dataset.py            # Captura de dataset desde webcam
│   └── extract_centroids.py          # Extracción de centroides para rechazo
│
├── tests/                            # Tests unitarios
│   ├── __init__.py
│   ├── test_message.py               # Tests del protocolo serial
│   ├── test_classifier.py            # Tests del clasificador
│   └── test_excel_logger.py          # Tests del logger Excel
│
└── models/                           # Modelos entrenados (no commiteados)
    └── *.keras / *.h5                # Archivos de modelo
```

---

## 4. Pipeline de Visión (Python)

### 4.1 `main.py` — Orquestador

**Archivo:** `src/vision/main.py` (392 líneas)

Es el punto de entrada del sistema en producción. Orquesta: cámara → clasificador → serial → Excel.

#### Cómo se ejecuta

```bash
python -m src.vision.main                          # Modo normal, cámara 0, threshold 0.7
python -m src.vision.main --camera 2               # Cámara índice 2
python -m src.vision.main --threshold 0.75          # Threshold de confianza más alto
python -m src.vision.main --model path/model.keras  # Modelo personalizado
python -m src.vision.main --test                    # Modo test (sin serial, sin display)
python -m src.vision.main --test --display          # Modo test con ventana de previsualización
python -m src.vision.main --no-display              # Headless (sin ventana)
python -m src.vision.main --port /dev/ttyUSB0       # Puerto serial específico
python -m src.vision.main --rejection-sigma 4.0     # Ajusta sensibilidad del rechazo
python -m src.vision.main --inference-skip 1        # Inferencia en cada frame (por defecto cada 2)
```

#### Flags (argumentos CLI)

| Flag | Tipo | Default | Descripción |
|------|------|---------|-------------|
| `--port` | str | `None` (auto-detect) | Puerto serie del ESP32. Auto-detecta buscando VID/PID de CP210, CH340, SiLabs. |
| `--threshold` | float | `0.7` | Umbral de confianza [0,1]. Por debajo de este valor, la predicción se degrada a "no bottle". |
| `--rejection-sigma` | float | `6.0` | Sigma para rechazo por espacio de características. Más alto = más permisivo. 0 = deshabilita rechazo. |
| `--test` | flag | `False` | Modo test: no conecta serial, no muestra ventana (a menos que se combine con `--display`). |
| `--display` | flag | `False` | Fuerza la ventana de previsualización (por defecto activa en modo normal, inactiva en `--test`). |
| `--no-display` | flag | `False` | Deshabilita la ventana de previsualización explícitamente. |
| `--camera` | int | `0` | Índice del dispositivo de cámara (0 = primera webcam del PC). |
| `--inference-skip` | int | `2` | Ejecuta inferencia cada N frames. `2` = la inferencia corre a la mitad de FPS que el display. |
| `--model` | str | `models/bottle_classifier_latest.keras` | Ruta al modelo `.keras` entrenado. |

#### Funcionamiento interno

1. **Detección de GUI**: Al inicio, intenta crear una ventana con `cv2.namedWindow`. Si falla (OpenCV sin soporte GTK), desactiva el display automáticamente. Esto permite correr en entornos headless (servidores, Raspberry sin monitor).

2. **Carga del clasificador**: Instancia `BottleTFClassifier` con el modelo, threshold y rejection-sigma indicados.

3. **Conexión serial**: Busca el ESP32 por VID/PID (CP210, CH340, SiLabs) o por descripción "USB"/"serial". Si no encuentra, reintenta cada 2 segundos.

4. **Bucle principal**:
   - Cada `inference_skip` frames ejecuta `classifier.predict(frame)`
   - En los frames intermedios reusa la última predicción (esto da un display más suave)
   - Calcula FPS con EMA (media móvil exponencial) para el overlay
   - Dibuja el overlay con `_draw_overlay()`: bounding box (aunque TF no da bbox, el código está preparado), barra de estado inferior, contador de FPS
   - Envía comando serial con `encode()`
   - Si hay logger Excel, registra las detecciones con rate limiting de 2 segundos
   - Loggea al terminal cada 1 segundo o en cada cambio de estado

5. **Salida**: Tecla `q` o `ESC` en la ventana, o `Ctrl+C` en terminal.

#### `_draw_overlay()`

Función que dibuja la información visual sobre el frame:

- Color **verde** si hay botella detectada, **rojo** si no
- Etiqueta con nombre de clase + confianza porcentual
- Si hay rejection OOD: muestra `[OOD]` y la distancia al centroide
- Barra de estado inferior con: clase, confianza, ángulo del servo
- FPS en esquina superior derecha

#### `_auto_detect_port()`

Busca el puerto serie del ESP32:

1. Primero busca por palabras clave en la descripción: `cp210`, `ch340`, `silab`, `esp32`
2. Si no encuentra, busca cualquier puerto que contenga "usb" o "serial"
3. Si no encuentra nada, retorna `None`

#### `_open_serial()`

Intenta abrir el puerto serie. Si falla, retorna `None` sin lanzar excepción. En el bucle principal se reintenta la conexión cada `_SERIAL_RETRY_DELAY` (2 segundos).

---

### 4.2 `classifier_tf.py` — Clasificador MobileNetV2

**Archivo:** `src/vision/classifier_tf.py` (411 líneas)

Es el cerebro del sistema. Usa **MobileNetV2** pre-entrenado en ImageNet con una cabeza personalizada para clasificar 3 tipos de envases.

#### Clases públicas

**`BottleType(IntEnum)`** — Enum que mapea IDs de clase a nombres:

```python
class BottleType(IntEnum):
    NONE = 0
    POOL_VERDE = 1
    HATSU_MORADO = 2
```

Se usa tanto en el clasificador como en el protocolo serial, garantizando que los valores sean consistentes.

**`BottlePrediction(dataclass)`** — Resultado de una predicción. Es **inmutable** (frozen=True):

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `class_id` | int | Índice de clase (0, 1, 2) |
| `confidence` | float | Probabilidad softmax de la clase predicha |
| `class_name` | str | Etiqueta legible ("No bottle", "Pool Verde", "Hatsu Morado") |
| `rejected` | bool | Si fue rechazado por distancia de características (OOD) |
| `feature_distance` | float | Distancia euclidiana al centroide de la clase (0 si rechazo desactivado) |

**`BottleTFClassifier`** — Clase principal.

#### Constructor: `__init__(model_path, threshold, centroids_path, rejection_sigma)`

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `model_path` | str | (obligatorio) | Ruta al archivo `.keras` del modelo entrenado |
| `threshold` | float | `0.5` | Umbral de confianza. Predicciones por debajo → clase 0 |
| `centroids_path` | str\|None | `None` | Ruta al JSON de centroides. `None` = auto-detecta. `""` = deshabilita rechazo |
| `rejection_sigma` | float | `3.0` | Sigmas para el threshold de rechazo (solo referencia; el runtime usa su propio sigma) |

**Proceso de inicialización:**

1. Carga el modelo con `tf.keras.models.load_model(model_path)`
2. Crea `_forward()` usando `@tf.function(jit_compile=True)` para activar XLA (compilación acelerada). Si XLA no está disponible, cae a `tf.function` normal sin JIT.
3. **Feature-space rejection**: si encuentra el archivo de centroides (auto-detectado o explícito), construye un modelo combinado que produce **dos salidas** en una sola pasada: el vector de características (128-d de la penúltima capa) + la salida softmax. Esto evita hacer dos inferencias.

#### Método `predict(frame)`

```python
pred = classifier.predict(frame_bgr)  # → BottlePrediction
```

1. Toma un frame BGR de cualquier tamaño
2. Lo preprocesa con `to_tf_input()` → 224×224 RGB float32
3. Agrega dimensión batch: `tensor[None, ...]`
4. Si el rechazo está activo: ejecuta `_feature_forward` → obtiene (features, probs)
5. Si no: ejecuta `_forward` → obtiene solo probs
6. Convierte a `BottlePrediction` con `_parse_prediction()`

#### `_parse_prediction(probs, features=None)`

1. Toma `argmax` de las probabilidades → clase predicha
2. Si hay features y rechazo activo: llama a `_reject_by_features()`. Si la distancia excede el threshold, rechaza y degrada a clase 0 con `rejected=True`
3. Si la confianza está por debajo del threshold: degrada a clase 0

#### Rechazo por espacio de características (`_reject_by_features`)

Este es el mecanismo para reducir falsos positivos con objetos nunca vistos:

```python
threshold = mean_dist + self._rejection_sigma * std_dist
distance = np.linalg.norm(features - centroid)
return distance > threshold  # True = rechazar
```

- **Centroide**: vector promedio de características (128-d) de todas las imágenes de entrenamiento de esa clase
- **mean_dist**: distancia promedio desde el centroide en entrenamiento
- **std_dist**: desviación estándar de esas distancias
- **threshold**: si la distancia de la nueva imagen supera `mean_dist + sigma * std_dist`, se considera "fuera de distribución" (OOD) y se rechaza

**Sigma ajustable en runtime**: el threshold se recalcula en cada predicción usando el `rejection_sigma` actual, NO el que estaba guardado en el JSON. Esto permite ajustar la sensibilidad sin reextraer centroides.

#### Método `_find_feature_layer(model)`

Encuentra la capa penúltima Dense (la de 128 neuronas antes del softmax). Busca por nombre `dense_1` primero (nombre estándar del `model.py`), y si no lo encuentra, auto-detecta buscando la anteúltima capa Dense.

#### Método `_build_forward(model)`

Envuelve la inferencia con `@tf.function(jit_compile=True)`:

- **XLA (Accelerated Linear Algebra)**: compila el grafo de TensorFlow a código máquina optimizado. Acelera la inferencia significativamente
- Si XLA falla (entornos sin soporte), cae a `tf.function` normal
- La función envuelta corre con `training=False` (desactiva Dropout y BatchNorm en inferencia)

#### `_load_model_safe(model_path)`

Carga el modelo con `tf.keras.models.load_model()`. Separado como método estático para facilitar el mockeo en tests.

---

### 4.3 `capture.py` — Captura de Cámara

**Archivo:** `src/vision/capture.py` (56 líneas)

Wrapper limpio de `cv2.VideoCapture` implementado como **context manager**:

```python
with CameraCapture(source=0) as camera:
    success, frame = camera.read()
```

#### Clase `CameraCapture`

| Método | Descripción |
|--------|-------------|
| `__init__(source)` | Recibe el índice de cámara (int) o ruta de video (str) |
| `__enter__()` | Abre la cámara, configura 640×480, lee un frame de calentamiento para estabilizar auto-exposición |
| `__exit__()` | Libera la cámara |
| `read()` | Retorna `(success, frame)` — frame BGR de 640×480 |

**Por qué 640×480**: es una resolución estándar que balancea velocidad de captura (FPS) con suficiente detalle para clasificar botellas a 224×224 después del resize.

**Frame de calentamiento**: se lee un frame inmediatamente después de abrir la cámara para que el auto-exposure se estabilice antes de la primera inferencia real.

---

### 4.4 `preprocess.py` — Preprocesamiento de Imágenes

**Archivo:** `src/vision/preprocess.py` (29 líneas)

```python
def to_tf_input(frame: np.ndarray) -> np.ndarray:
```

Convierte un frame BGR de cualquier tamaño en un tensor listo para MobileNetV2:

1. **Redimensiona** a 224×224 con interpolación bilineal (`cv2.INTER_LINEAR`)
2. **Convierte BGR→RGB**: OpenCV usa BGR por defecto, pero MobileNetV2 espera RGB
3. **Convierte a float32**: mantiene valores en [0, 255]

**NO normaliza a [0,1]**: el modelo tiene incorporado `tf.keras.applications.mobilenet_v2.preprocess_input()` que hace `x / 127.5 - 1` para escalar a [-1, 1]. Esto está en `model.py` línea 56. Si se normalizara dos veces, el rendimiento caería.

---

### 4.5 `excel_logger.py` — Registro en Excel

**Archivo:** `src/vision/excel_logger.py` (64 líneas)

Registra cada detección positiva en un archivo Excel para trazabilidad.

#### Clase `ExcelLogger`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `filepath` | `registro_envases.xlsx` | Ruta del archivo Excel |
| `cooldown` | `2.0` segundos | Tiempo mínimo entre registros consecutivos |

#### Método `log(class_name) → bool`

1. Verifica rate limiting: si pasaron menos de `cooldown` segundos desde el último log, retorna `False` y no guarda
2. Obtiene fecha (`YYYY-MM-DD`) y hora (`HH:MM:SS`) actuales
3. Si el archivo ya existe, lo lee y concatena la nueva fila
4. Si no existe, crea el archivo con headers: `Fecha | Hora | Objeto`
5. Guarda con `pandas.to_excel()` sin incluir el índice

**Columnas del Excel**:

| Fecha | Hora | Objeto |
|-------|------|--------|
| 2026-06-25 | 14:30:15 | pool_verde |

**Rate limiting**: evita que el mismo envase se registre múltiples veces mientras está frente a la cámara. Como la inferencia corre cada 2 frames (~15-30 veces por segundo), sin rate limiting se llenaría el Excel de duplicados.

---

## 5. Protocolo Serial

**Archivo:** `src/protocol/message.py` (89 líneas)

Define el protocolo de comunicación entre el host Python y el ESP32 usando **JSON compacto** sobre USB serial a **9600 baud**.

#### Formato del mensaje

| Evento | Mensaje | LED | Servo |
|--------|---------|-----|-------|
| Pool Verde detectado | `{"b":1,"t":1,"s":90}\n` | Verde | 90° |
| Hatsu Morado detectado | `{"b":1,"t":2,"s":90}\n` | Verde | 90° |
| Sin botella | `{"b":0,"t":0,"s":180}\n` | Rojo | 180° |

**Campos del JSON:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `b` | int (0/1) | **Bottle detected**. 1 = hay botella, 0 = no |
| `t` | int (0/1/2) | **Type**. Tipo de botella (0=ninguna, 1=Pool, 2=Hatsu) |
| `s` | int (0-180) | **Servo angle**. Ángulo del servomotor |

#### `encode(bottle_detected, bottle_type, servo_angle) → bytes`

Convierte los parámetros a JSON compacto:

- Usa `separators=(",", ":")` para eliminar espacios → `{"b":1,"t":1,"s":90}` en lugar de `{"b": 1, "t": 1, "s": 90}`
- Agrega `\n` como delimitador de mensaje
- Si `servo_angle` es `None`, usa 90° si hay botella, 180° si no

#### `decode(data) → dict`

Parsea los bytes recibidos:

1. Decodifica UTF-8 y elimina whitespace
2. Parsea JSON con `json.loads()`
3. Valida que exista la clave `"b"` — si falta, lanza `KeyError`
4. **Backward compatibility**: si falta `"t"`, asume 0. Si falta `"s"`, infiere el ángulo desde `"b"`
5. Convierte todos los valores a `int`

#### `BottleType(IntEnum)` (duplicado)

El mismo enum del clasificador está también aquí para mantener el protocolo autocontenido y garantizar que los valores sean consistentes entre el clasificador y el mensaje serial.

---

## 6. Pipeline de Entrenamiento

### 6.1 `train.py` — Punto de Entrada

**Archivo:** `training/train.py` (265 líneas)

Script de entrenamiento en dos fases: **frozen backbone** → **fine-tuning**.

#### Cómo se ejecuta

```bash
python training/train.py --data-dir ./training/data
python training/train.py --data-dir ./training/data --epochs-frozen 15 --epochs-finetune 10
python training/train.py --data-dir ./training/data --lr-frozen 1e-3 --lr-finetune 1e-5
python training/train.py --data-dir ./training/data --model-dir ./models
```

#### Flags

| Flag | Default | Descripción |
|------|---------|-------------|
| `--data-dir` | (obligatorio) | Directorio raíz del dataset con subcarpetas por clase |
| `--model-dir` | `models` | Directorio donde guardar los modelos entrenados |
| `--epochs-frozen` | `20` | Épocas máximas entrenando solo la cabeza (backbone congelado) |
| `--epochs-finetune` | `15` | Épocas máximas de fine-tuning |
| `--lr-frozen` | `5e-4` | Learning rate para la fase congelada |
| `--lr-finetune` | `1e-5` | Learning rate para fine-tuning (más bajo) |
| `--img-size` | `224` | Tamaño de imagen de entrada (cuadrado) |

#### Flujo de entrenamiento

**Fase 0 — Carga de datos:**
1. Escanea subdirectorios en `--data-dir` buscando imágenes
2. Usa `sklearn.model_selection.train_test_split` con **estratificación** (mantiene proporción de clases en train/val)
3. Cuenta imágenes por clase y calcula **class weights** para balancear la pérdida

**Fase 1 — Frozen backbone (entrenamiento de la cabeza):**
1. MobileNetV2 pre-entrenado en ImageNet se mantiene **congelado** (no se entrena)
2. Solo se entrenan las capas nuevas: GAP → Dense(256) → Dropout → Dense(128) → Softmax(3)
3. Learning rate: 5e-4
4. Callbacks: ModelCheckpoint (guarda mejor modelo), EarlyStopping (paciencia 5), ReduceLROnPlateau (reduce LR si val_loss se estanca)

**Fase 2 — Fine-tuning:**
1. Descongela las últimas **30 capas** del backbone MobileNetV2 (el ~20% superior)
2. Las capas tempranas (bordes, texturas genéricas) se mantienen congeladas
3. Learning rate 20× más bajo: 1e-5 (para no destruir los pesos pre-entrenados)
4. Repite los mismos callbacks

**Guardado:**
1. Guarda con timestamp: `models/bottle_classifier_20260625_143000.keras`
2. Guarda copia como: `models/bottle_classifier_latest.keras` (siempre sobrescribe)

#### Callbacks

| Callback | Monitorea | Paciencia | Acción |
|----------|-----------|-----------|--------|
| `ModelCheckpoint` | val_loss | — | Guarda el mejor modelo |
| `EarlyStopping` | val_loss | 5 épocas | Detiene entrenamiento si no mejora |
| `ReduceLROnPlateau` | val_loss | 2 épocas | Reduce LR ×0.3 si val_loss se estanca |

#### Class weights

```python
class_weight = {i: total / (n_classes * count) for i, count in enumerate(class_counts)}
```

Si `no_bottle` tiene 398 imágenes y `pool_verde` tiene 750, la clase minoritaria recibe un peso mayor en la función de pérdida. Esto evita que el modelo se sesgue hacia las clases con más datos.

---

### 6.2 `model.py` — Arquitectura del Modelo

**Archivo:** `training/model.py` (108 líneas)

#### `build_mobilenetv2(img_size, num_classes, dropout_rate, dense_units, dense_units_2, backbone_trainable)`

Construye el modelo completo:

```
Entrada: (224, 224, 3) RGB, valores [0, 255]
    │
    ▼
preprocess_input (escala a [-1, 1])
    │
    ▼
MobileNetV2 backbone (sin top, pesos ImageNet, congelado)
    │  output: (7, 7, 1280)
    ▼
GlobalAveragePooling2D  →  (1280,)
    │
    ▼
Dense(256, ReLU)  →  256 neuronas
    │
    ▼
Dropout(0.3)  →  apaga 30% de las neuronas aleatoriamente
    │             (regularización para evitar overfitting)
    ▼
Dense(128, ReLU)  →  128 neuronas (capa de características)
    │
    ▼
Dense(num_classes, Softmax)  →  probabilidades por clase
```

**Detalles importantes:**

- **preprocess_input**: MobileNetV2 fue entrenado con inputs en [-1, 1]. El `preprocess_input` de Keras hace `x / 127.5 - 1`. Esto convierte [0, 255] → [-1, 1].
- **backbone(training=False)**: Forza el modo inferencia en el backbone durante la fase congelada para que las capas BatchNorm NO actualicen sus estadísticas con los batches pequeños del nuevo dataset.
- **Dropout(0.3)**: Regularización. En entrenamiento apaga aleatoriamente el 30% de las neuronas, forzando al modelo a no depender de una sola ruta de activación.
- **Dense(128)**: Esta es la **capa de características** (feature layer) de 128 dimensiones que se usa para el rechazo por espacio de características.

#### `unfreeze_top_layers(model, num_layers=30)`

Descongela las últimas `num_layers` capas del backbone:

1. Hace `backbone.trainable = True`
2. Las primeras `len(backbone.layers) - num_layers` se vuelven a congelar
3. Las últimas `num_layers` quedan entrenables

¿Por qué 30 capas? MobileNetV2 tiene ~154 capas internas. Las primeras ~124 (80%) detectan características genéricas (bordes, texturas, colores) que sirven para cualquier tarea. Solo las últimas 30 (20%) necesitan ajustarse para reconocer botellas específicas.

---

### 6.3 `dataset.py` — Carga y Aumento de Datos

**Archivo:** `training/dataset.py` (204 líneas)

#### `load_dataset(data_dir, config) → (train_ds, val_ds, class_names, class_counts)`

Retorna datasets de TensorFlow listos para entrenar.

**Estructura esperada del directorio:**

```
training/data/
├── no_bottle/
│   ├── img001.jpg
│   └── ...
├── pool_verde/
│   ├── img001.jpg
│   └── ...
└── hatsu_morado/
    ├── img001.jpg
    └── ...
```

**Proceso:**

1. **`_discover_classes()`**: Escanea subdirectorios y los ordena según `Config.CLASS_NAMES` = `("no_bottle", "pool_verde", "hatsu_morado")`. Esto garantiza que el orden de clases sea **siempre el mismo**, alineado con `BottleType`.
2. **`_collect_files()`**: Recolecta todos los archivos de imagen con extensiones: `.jpg`, `.jpeg`, `.png`, `.bmp`.
3. **Stratified split**: `train_test_split(test_size=0.2, stratify=labels)` asegura que la proporción de clases sea idéntica en train (80%) y validation (20%).
4. **Construcción de datasets tf.data**:
   - Crea un `tf.data.Dataset` desde los paths
   - Para training: hace **shuffle** (buffer = tamaño del dataset)
   - Decodifica imágenes con `tf.image.decode_jpeg`/`decode_png`
   - Redimensiona a 224×224 con `tf.image.resize`
   - **Cache**: las imágenes decodificadas se guardan en RAM (`ds.cache()`) para no releer del disco en cada época

#### Aumento de datos (solo training set)

**`_build_augmentation()`** — pipeline secuencial de Keras:

| Transformación | Rango | Propósito |
|---------------|-------|-----------|
| `RandomFlip("horizontal")` | Siempre | Inversión horizontal — simula botellas mirando a izquierda/derecha |
| `RandomRotation(0.1)` | ±10% de círculo | Rotación leve — simula ángulos ligeros |
| `RandomZoom(0.1)` | ±10% | Zoom in/out — simula distancia variable |
| `RandomShear(0.15)` | ±15% | Distorsión de corte — simula perspectiva oblicua |
| `RandomTranslation(0.10)` | ±10% en X/Y | Desplazamiento — simula botella fuera del centro |
| `RandomBrightness(0.2)` | ±20% | Variación de brillo — simula cambios de iluminación |
| `RandomContrast(0.2)` | ±20% | Variación de contraste |
| `RandomGrayscale(0.2)` | 20% de probabilidad | Convierte a escala de grises — reduce dependencia del color |

Además, una transformación adicional `_color_jitter()`:

| Transformación | Rango | Propósito |
|---------------|-------|-----------|
| `random_hue` | ±15% | Cambia el tono de color (simula diferentes tintes de plástico) |
| `random_saturation` | 50-150% | Cambia saturación |

**Por qué este nivel de aumento**: las botellas pueden aparecer en diferentes condiciones de iluminación, ángulos, distancias, y el plástico puede tener variaciones de color. El aumento fuerza al modelo a aprender características robustas (forma, textura) en lugar de memorizar condiciones específicas de captura.

##### Flujo completo de aumento (training):

```
Imagen original → Decode → Resize 224×224 → Cache (RAM)
  → RandomFlip → RandomRotation → RandomZoom → RandomShear
  → RandomTranslation → RandomBrightness → RandomContrast
  → RandomGrayscale → Clip [0, 255]
  → ColorJitter (hue + saturación)
  → Batch → Prefetch
```

**Clip**: después del brightness/contrast los valores pueden salirse de [0, 255]; se recortan para mantener el rango válido.

**Prefetch**: `ds.prefetch(tf.data.AUTOTUNE)` prepara el siguiente batch mientras el GPU entrena el actual.

---

### 6.4 `config.py` — Configuración e Hiperparámetros

**Archivo:** `training/config.py` (109 líneas)

Dataclass inmutable con todos los hiperparámetros.

| Campo | Valor | Descripción |
|-------|-------|-------------|
| `IMG_SIZE` | `224` | Tamaño de entrada (MobileNetV2 espera 224×224) |
| `BACKBONE` | `"mobilenetv2"` | Arquitectura CNN base |
| `NUM_CLASSES` | `3` | no_bottle, pool_verde, hatsu_morado |
| `CLASS_NAMES` | `("no_bottle", "pool_verde", "hatsu_morado")` | Orden de clases (crítico — debe coincidir con `BottleType`) |
| `DENSE_UNITS` | `256` | Neuronas en primera capa densa del clasificador |
| `DENSE_UNITS_2` | `128` | Neuronas en segunda capa densa (capa de características) |
| `DROPOUT_RATE` | `0.3` | Dropout para regularización |
| `BATCH_SIZE` | auto (según RAM) | Batch size calculado automáticamente |
| `VALIDATION_SPLIT` | `0.2` | 20% de datos para validación |
| `RANDOM_SEED` | `42` | Semilla para reproducibilidad |
| `FROZEN_EPOCHS` | `20` | Épocas máximas en fase congelada |
| `FROZEN_LR` | `5e-4` | Learning rate fase congelada |
| `FINETUNE_EPOCHS` | `15` | Épocas máximas fine-tuning |
| `FINETUNE_LR` | `1e-5` | Learning rate fine-tuning (más bajo) |
| `FINETUNE_TOP_LAYERS` | `30` | Capas a descongelar del backbone |
| `PATIENCE` | `5` | Paciencia del EarlyStopping |
| `MODEL_SAVE_DIR` | `"models"` | Directorio de modelos |

#### `auto_batch_size(max_batch=32, safety_mb=512) → int`

Calcula el batch size máximo seguro según la RAM disponible:

1. Intenta con `psutil.virtual_memory()` (librería `psutil`)
2. Si no está instalada, lee `/proc/meminfo` en Linux
3. Si falla, asume 4 GB libres
4. Fórmula: `batch = free_mb / safety_mb` (ej: 8 GB libres / 512 MB = 16 de batch)
5. Limita a `max_batch` (32)

---

## 7. Herramientas

### 7.1 `capture_dataset.py` — Captura de Dataset

**Archivo:** `tools/capture_dataset.py` (~295 líneas)

Herramienta interactiva para capturar imágenes de entrenamiento desde la webcam de la PC. Diseñada para reemplazar las fotos de celular con imágenes capturadas en las condiciones reales de producción.

#### Cómo se ejecuta

```bash
python -m tools.capture_dataset
python -m tools.capture_dataset --camera 2
python -m tools.capture_dataset --roi-size 224
python -m tools.capture_dataset --resize-to 224 --width 1280
```

#### Flags

| Flag | Default | Descripción |
|------|---------|-------------|
| `--camera` | `0` | Índice del dispositivo de cámara |
| `--roi-size` | `300` | Tamaño del recuadro de captura verde en píxeles |
| `--resize-to` | `224` | Tamaño final al que se redimensiona la imagen guardada |
| `--width` | `640` | Ancho de captura de la cámara (alto se calcula 4:3) |

#### Teclas

| Tecla | Acción |
|-------|--------|
| `1` | Guarda ROI como `no_bottle` en `training/data/no_bottle/` |
| `2` | Guarda ROI como `pool_verde` en `training/data/pool_verde/` |
| `3` | Guarda ROI como `hatsu_morado` en `training/data/hatsu_morado/` |
| `r` | **Borra TODAS las capturas de esta sesión** (archivos + contadores a cero) |
| `q` | Sale de la aplicación |

#### Funcionamiento

1. Abre la cámara con `cv2.VideoCapture`
2. Muestra un recuadro verde centrado de `roi_size` × `roi_size`
3. Al presionar 1/2/3: corta el ROI, lo redimensiona a `resize_to`×`resize_to` (default 224), lo guarda con nombre `timestamp_clase_numero.jpg`
4. Muestra contadores en vivo: cuántas capturas lleva de cada clase
5. **Tecla R**: elimina físicamente todos los archivos guardados en esta sesión y resetea contadores. Útil para empezar de nuevo sin salir del programa.
6. Al salir, muestra resumen de la sesión

**Nombre de archivo**: `20260625_143000_no_bottle_0001.jpg` — ordenable por timestamp.

**Toast visual**: breve mensaje verde confirmando cada captura (desaparece a los 2 segundos).

---

### 7.2 `extract_centroids.py` — Extracción de Centroides

**Archivo:** `tools/extract_centroids.py` (287 líneas)

Extrae los centroides de características del dataset para el sistema de rechazo OOD (Out-of-Distribution).

#### Cómo se ejecuta

```bash
python -m tools.extract_centroids \
    --model models/bottle_classifier_latest.keras \
    --data-dir training/data \
    --sigma 3.0
```

#### Flags

| Flag | Default | Descripción |
|------|---------|-------------|
| `--model` | (obligatorio) | Ruta al modelo `.keras` entrenado |
| `--data-dir` | (obligatorio) | Directorio raíz del dataset |
| `--sigma` | `3.0` | Sigmas para threshold de rechazo (solo referencia) |
| `--batch-size` | `32` | Batch size para extracción de features |
| `--output` | auto | Ruta de salida JSON (default: junto al modelo, con sufijo `_centroids.json`) |

#### Cómo funciona

1. Carga el modelo entrenado
2. Encuentra la capa de características (penúltima Dense, 128-d)
3. Construye un modelo que solo produce el vector de características (sin softmax)
4. Pasa **todas** las imágenes del dataset por ese modelo en batches
5. Por cada clase, calcula:
   - **Centroide** (`mean`): vector promedio de características (128 floats)
   - **mean_dist**: distancia euclidiana promedio desde el centroide
   - **std_dist**: desviación estándar de las distancias
   - **threshold**: `mean_dist + sigma * std_dist`
6. Guarda todo en un archivo JSON

#### Archivo de salida (`bottle_classifier_latest_centroids.json`)

```json
{
  "0": {
    "mean": [0.123, -0.456, ...],    // 128 valores
    "mean_dist": 2.3456,
    "std_dist": 0.7890,
    "threshold": 4.7126,
    "class_name": "no_bottle",
    "num_samples": 398
  },
  "1": { ... },   // pool_verde
  "2": { ... }    // hatsu_morado
}
```

**Importante**: el clasificador en runtime IGNORA el `threshold` guardado y lo recalcula con `mean_dist + rejection_sigma * std_dist`. El sigma de `extract_centroids` es solo informativo/referencia.

---

## 8. Firmware ESP32 (C++)

### 8.1 `esp32.ino` — Firmware Principal

**Archivo:** `src/hardware/esp32/esp32.ino` (140 líneas)

Firmware del ESP32 que recibe comandos JSON por serial y controla LEDs, servomotor y zumbador.

#### Pines (versión actual del código)

| Componente | Pin GPIO | Nota |
|------------|----------|------|
| LED Verde | 02 | La documentación dice GPIO 26, el código usa 02 |
| LED Rojo | 05 | La documentación dice GPIO 27, el código usa 05 |
| Servo | 14 | La documentación dice GPIO 13, el código usa 14 |
| Buzzer | 27 | La documentación dice GPIO 12, el código usa 27 |

*Nota: hay discrepancia entre los pines en el código y en los comentarios. Los pines reales son los del código.*

#### `setup()`

1. Inicia serial a 9600 baud con timeout de 50ms
2. Inicializa LEDs, servo y buzzer
3. Parpadeo breve de verificación

#### `loop()`

1. Lee byte a byte del serial, acumulando en buffer hasta recibir `\n`
2. Cuando recibe `\n`: llama `processCommand(buffer)`
3. Verifica timeout de LEDs: si pasaron ≥5s sin comando, apaga LEDs (standby autónomo)
4. Si no hay timeout, ejecuta `standbyBlink()` (parpadeo cada 5s)
5. Actualiza buzzer (lo apaga cuando termina el beep)
6. Delay 5ms

#### `processCommand(json)`

1. Busca la clave `"b"` con `strstr()`, luego `strchr()` para encontrar los `:`, y lee el valor
2. Si `b == 1`: LED verde + beep de 200ms
3. Si `b == 0`: LED rojo
4. Busca la clave `"s"`, parsea el ángulo con `atoi()`, y si está en [0, 180], mueve el servo
5. **No usa ArduinoJson** — el parsing es manual con `strstr/strchr/atoi` para evitar dependencias

#### Modo autónomo de fallback

Si el ESP32 no recibe comandos por ≥5 segundos:
- Los LEDs se apagan
- El servo mantiene su última posición
- Cuando vuelve un comando, retoma la operación normal

---

### 8.2 `led_control.h` — Control de LEDs

**Archivo:** `src/hardware/esp32/led_control.h` (100 líneas)

Clase `LEDControl` que maneja los dos LEDs.

| Método | Acción |
|--------|--------|
| `begin()` | Configura pines como OUTPUT, apaga ambos |
| `green()` | LED verde ON, rojo OFF, actualiza timestamp |
| `red()` | LED rojo ON, verde OFF, actualiza timestamp |
| `off()` | Apaga ambos |
| `standbyBlink()` | Parpadeo de ambos cada 5s si no hay comando reciente |
| `isTimedOut()` | `true` si pasaron ≥5s sin comando |
| `refresh()` | Resetea el contador de timeout |

**Stanford Blink**: cuando el sistema está funcionando pero no hay cambios de estado, ambos LEDs parpadean juntos por 100ms cada 5 segundos para indicar que el sistema está vivo.

---

### 8.3 `servo_control.h` — Control del Servomotor

**Archivo:** `src/hardware/esp32/servo_control.h` (80 líneas)

Clase `ServoControl` que maneja el servomotor SG90 usando PWM por hardware (LEDC).

#### Cálculo de PWM

El SG90 usa PWM a **50 Hz** (período de 20 ms). Los pulsos estándar son:
- **0°** → 544 µs
- **180°** → 2400 µs

Con el LEDC de ESP32 a **16 bits** de resolución (valores 0-65535):

```
duty = pulse_us / 20000 * 65536

0°   → (544 / 20000) * 65536 ≈ 1782
180° → (2400 / 20000) * 65536 ≈ 7864
```

Para un ángulo intermedio:

```python
duty = DUTY_0_DEG + (DUTY_RANGE * angle) / MAX_ANGLE
     = 1782 + (6082 * angle) / 180
```

| Método | Acción |
|--------|--------|
| `begin(pin)` | Adjunta PWM al pin, servo a 90° (centro) |
| `setAngle(angle)` | Mueve a ángulo [0, 180] con `constrain()` |
| `getCurrentAngle()` | Retorna último ángulo |
| `detach()` | Suelta el pin PWM |

---

### 8.4 `buzzer_control.h` — Control del Zumbador

**Archivo:** `src/hardware/esp32/buzzer_control.h` (56 líneas)

Clase `BuzzerControl` para un zumbador activo (se enciende con HIGH).

| Método | Acción |
|--------|--------|
| `begin()` | Configura pin como OUTPUT, apaga |
| `beep(durationMs)` | Enciende buzzer por `durationMs` milisegundos (no bloqueante) |
| `update()` | Apaga buzzer cuando expira el tiempo (llamar desde loop) |

**No bloqueante**: usa `millis()` para programar el apagado. El beep de 200ms no detiene el loop principal.

---

## 9. Tests

**Ubicación:** `tests/`

### `test_message.py` (176 líneas)

Tests del protocolo serial. Verifica:

- **`BottleType` enum**: valores enteros correctos (0, 1, 2) y nombres
- **`encode()`**:
  - Botella detectada → `{"b":1,"t":0,"s":90}\n`
  - Sin botella → `{"b":0,"t":0,"s":180}\n`
  - Con `bottle_type` 1 y 2 → `"t"` correcto
  - Ángulo personalizado
  - JSON compacto sin espacios
- **`decode()`**:
  - Parseo correcto de mensajes completos
  - **Backward compatibility**: mensajes sin `"t"` o sin `"s"` se parsean correctamente
  - Manejo de newlines y whitespace
  - Ángulos arbitrarios
- **Roundtrips**: encode → decode recupera el dict original
- **Errores**: JSON mal formado → `JSONDecodeError`, falta clave `"b"` → `KeyError`, vacío → `JSONDecodeError`

### `test_classifier.py` (217 líneas)

Tests del `BottleTFClassifier`. Mockea `tf.keras.models.load_model` para no necesitar modelo real.

- **`BottleType` enum**: valores y mapeo desde class_id
- **`BottlePrediction`**: construcción, inmutabilidad (frozen dataclass)
- **Predicciones**: cada clase (0, 1, 2) retorna class_id y class_name correctos
- **Threshold**: confianza bajo threshold → degrada a clase 0; en el threshold exacto o arriba → mantiene clase
- **Threshold personalizado**: 0.7 acepta 0.85, rechaza 0.65
- **Clase dominante**: cuando dos clases están sobre threshold, gana la de mayor confianza
- **Errores**: threshold fuera de [0, 1] → `ValueError`

### `test_excel_logger.py` (89 líneas)

Tests del `ExcelLogger`. Usa directorio temporal.

- **Creación**: primer log crea el archivo con headers correctos
- **Append**: segundo log agrega fila preservando la anterior
- **Rate limiting**: cooldown de 10s bloquea el segundo log; cooldown corto expira
- **Formato de fecha/hora**: `YYYY-MM-DD` y `HH:MM:SS`

### Cómo ejecutar tests

```bash
python -m unittest discover tests -v       # Todos los tests
python -m unittest tests.test_message       # Solo protocolo
python -m unittest tests.test_classifier    # Solo clasificador
```

---

## 10. Hardware: Conexiones Eléctricas

### Diagrama de conexión

```
                ESP32
        ┌───────────────────┐
        │                   │
        │  GPIO 02 ────────┼──────┬── LED Verde (ánodo)
        │                   │      │
        │                   │     ═══ 220Ω
        │                   │      │
        │                   │      └── GND
        │                   │
        │  GPIO 05 ────────┼──────┬── LED Rojo (ánodo)
        │                   │      │
        │                   │     ═══ 220Ω
        │                   │      │
        │                   │      └── GND
        │                   │
        │  GPIO 14 ────────┼────── Naranja (señal) ──┐
        │                   │                          │
        │                   │                    ╔═════╧════╗
        │                   │                    ║  SG90    ║
        │  GND    ────────┼─────────────────────╨┤  Servo   ║
        │                   │                    ║          ║
        │                   │                    ╚══════════╝
        │                   │                    (5V externo)
        │  GPIO 27 ────────┼────── Señal ── Buzzer activo (+)
        │                   │                    │
        │                   │                    └── GND (ESP32)
        │                   │
        │  USB ─────────────┼── PC (alimentación + datos)
        └───────────────────┘
```

### Especificaciones

| Componente | Especificación |
|------------|----------------|
| **ESP32** | ESP32-WROOM-32 o similar, 3.3V lógico |
| **LEDs** | Verde y Rojo, 5mm, 20mA, con resistencias 220Ω limitadoras de corriente |
| **Servomotor** | SG90 (micro servo), 5V, 0-180°, señal PWM 50 Hz |
| **Buzzer** | Zumbador activo (genera tono propio con HIGH), 3.3-5V |
| **Cámara** | Webcam del PC compatible con V4L2 (Linux) |
| **Conexión** | USB entre PC y ESP32 (alimentación + datos serial) |

### Advertencias

1. **NO alimentar el servo desde el USB del ESP32**: el SG90 puede consumir picos de ~750mA al moverse, muy por encima de lo que el regulador USB del ESP32 puede entregar. Usar una **fuente externa de 5V** para el servo (compartir GND con ESP32).
2. **Resistencias 220Ω obligatorias**: sin ellas, los LEDs drenarán corriente ilimitada y pueden dañar los GPIO del ESP32.
3. **Buzzer activo vs pasivo**: este código usa un buzzer **activo** (suena solo con HIGH). Un buzzer pasivo necesita señal PWM para generar tono.

---

## 11. Glosario de Flags y Parámetros

### `src/vision/main.py`

| Flag | Tipo | Default | Descripción |
|------|------|---------|-------------|
| `--port` | str | auto | Puerto serial ESP32 |
| `--threshold` | float | 0.7 | Umbral de confianza mínimo [0-1] |
| `--rejection-sigma` | float | 6.0 | Sigma para rechazo OOD (0 = desactivado) |
| `--test` | flag | — | Modo test sin serial |
| `--display` | flag | — | Forzar ventana preview |
| `--no-display` | flag | — | Deshabilitar ventana preview |
| `--camera` | int | 0 | Índice de cámara |
| `--inference-skip` | int | 2 | Inferencia cada N frames |
| `--model` | str | `models/bottle_classifier_latest.keras` | Ruta al modelo |

### `training/train.py`

| Flag | Default | Descripción |
|------|---------|-------------|
| `--data-dir` | (obligatorio) | Dataset con subcarpetas por clase |
| `--model-dir` | `models` | Directorio de salida de modelos |
| `--epochs-frozen` | 20 | Épocas backbone congelado |
| `--epochs-finetune` | 15 | Épocas fine-tuning |
| `--lr-frozen` | 5e-4 | Learning rate fase 1 |
| `--lr-finetune` | 1e-5 | Learning rate fase 2 |
| `--img-size` | 224 | Tamaño de imagen |

### `tools/capture_dataset.py`

| Flag | Default | Descripción |
|------|---------|-------------|
| `--camera` | 0 | Índice de cámara |
| `--roi-size` | 300 | Tamaño del recuadro de captura |
| `--resize-to` | 224 | Tamaño de imagen guardada |
| `--width` | 640 | Resolución horizontal de captura |

### `tools/extract_centroids.py`

| Flag | Default | Descripción |
|------|---------|-------------|
| `--model` | (obligatorio) | Modelo entrenado `.keras` |
| `--data-dir` | (obligatorio) | Dataset de entrenamiento |
| `--sigma` | 3.0 | Sigmas para threshold (referencia) |
| `--batch-size` | 32 | Batch para extracción |
| `--output` | auto | Ruta JSON de salida |

---

## 12. Requerimientos del Sistema

### Producción (`requirements.txt`)

| Librería | Versión Mínima | Propósito |
|----------|---------------|-----------|
| `opencv-python` | ≥4.8.0 | Captura de cámara, procesamiento de imágenes, visualización |
| `numpy` | ≥1.24.0 | Manipulación de arrays, operaciones matemáticas |
| `pyserial` | ≥3.5 | Comunicación serial con ESP32 |
| `tensorflow-cpu` | ≥2.13.0 | MobileNetV2, inferencia del clasificador |
| `openpyxl` | ≥3.1.0 | Escritura de archivos Excel |
| `pandas` | ≥2.0.0 | Manejo de datos tabulares para logging Excel |
| `psutil` | ≥5.9.0 | Detección de RAM disponible para batch size |

### Entrenamiento (`training/requirements-train.txt`)

| Librería | Versión Mínima | Propósito |
|----------|---------------|-----------|
| `tensorflow` | ≥2.13 | Entrenamiento del modelo (con GPU si está disponible) |
| `scikit-learn` | ≥1.2 | Stratified train/val split |
| `pillow` | ≥9.0 | Carga de imágenes (usado internamente por TF) |
| `numpy` | ≥1.24 | Operaciones numéricas |

### Python

- Versión **3.10+** requerida
- Usar `python -m` (module) para ejecutar, no como script directo

### ESP32

- Framework: **Arduino** (IDE o CLI)
- Board: **ESP32-WROOM-32** o compatible
- Baudrate: **9600** (configurable en código)

### Cámara

- Webcam del PC compatible con **V4L2** en Linux
- Resolución mínima: 640×480 (la usa el sistema)
- Sin requisitos especiales de driver

---

*Documentación generada el 25 de junio de 2026. Contiene la descripción completa de todos los módulos, flags, configuraciones y componentes del proyecto.*
