"""Serial protocol for host-to-ESP32 communication.

Uses compact JSON over USB serial at 9600 baud.
Format: {"b":1,"s":90}\n for bottle (servo at 90°),
        {"b":0,"s":180}\n for not bottle (servo at 180°).
"""

import json

_DELIMITER = b"\n"

# Servo angle mapping
_ANGLE_BOTTLE = 90
_ANGLE_NO_BOTTLE = 180


def encode(bottle_detected: bool, servo_angle: int | None = None) -> bytes:
    """Encode a detection result into a serial-ready JSON byte string.

    Args:
        bottle_detected: True if a bottle was detected, False otherwise.
        servo_angle: Servo target angle (0-180). Defaults to 90 for
            bottle, 180 for not bottle.

    Returns:
        UTF-8 encoded JSON bytes with newline delimiter,
        e.g. b'{"b":1,"s":90}\\n'.

    """
    if servo_angle is None:
        servo_angle = _ANGLE_BOTTLE if bottle_detected else _ANGLE_NO_BOTTLE
    payload = {"b": 1 if bottle_detected else 0, "s": servo_angle}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + _DELIMITER


def decode(data: bytes) -> dict:
    """Decode a serial byte string into a Python dictionary.

    Args:
        data: Raw bytes received from the serial port. May include
              leading/trailing whitespace or the newline delimiter.

    Returns:
        Parsed dictionary, e.g. ``{"b": 1, "s": 90}`` or ``{"b": 0, "s": 180}``.
        The ``"s"`` key defaults to the standard angle if not present
        in the message (backward-compatible).

    Raises:
        json.JSONDecodeError: If the data is not valid JSON.
        KeyError: If the ``"b"`` key is missing.

    """
    decoded = data.decode("utf-8").strip()
    if not decoded:
        raise json.JSONDecodeError("Empty data", "", 0)
    result = json.loads(decoded)
    if "b" not in result:
        raise KeyError(f"Missing required key 'b' in message: {decoded!r}")
    result["b"] = int(result["b"])
    if "s" in result:
        result["s"] = int(result["s"])
    else:
        # Backward compatibility: infer angle from detection
        result["s"] = _ANGLE_BOTTLE if result["b"] else _ANGLE_NO_BOTTLE
    return result
