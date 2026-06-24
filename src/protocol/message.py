"""Serial protocol for host-to-ESP32 communication.

Uses compact JSON over USB serial at 9600 baud.
Format: {"b":1,"t":1,"s":90}\n for Pool Verde,
        {"b":1,"t":2,"s":90}\n for Hatsu Morado,
        {"b":0,"t":0,"s":180}\n for no bottle.
"""

import json
from enum import IntEnum

_DELIMITER = b"\n"

# Servo angle mapping
_ANGLE_BOTTLE = 90
_ANGLE_NO_BOTTLE = 180


class BottleType(IntEnum):
    """Bottle type labels for the serial protocol (matches classifier)."""

    NONE = 0
    POOL_VERDE = 1
    HATSU_MORADO = 2


def encode(
    bottle_detected: bool,
    bottle_type: int = 0,
    servo_angle: int | None = None,
) -> bytes:
    """Encode a detection result into a serial-ready JSON byte string.

    Args:
        bottle_detected: True if a bottle was detected, False otherwise.
        bottle_type: Type of bottle (0=none, 1=Pool Verde, 2=Hatsu Morado).
            Defaults to 0 for backward compatibility.
        servo_angle: Servo target angle (0-180). Defaults to 90 for
            bottle, 180 for not bottle.

    Returns:
        UTF-8 encoded JSON bytes with newline delimiter,
        e.g. b'{"b":1,"t":1,"s":90}\\n'.

    """
    if servo_angle is None:
        servo_angle = _ANGLE_BOTTLE if bottle_detected else _ANGLE_NO_BOTTLE
    payload = {
        "b": 1 if bottle_detected else 0,
        "t": bottle_type,
        "s": servo_angle,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + _DELIMITER


def decode(data: bytes) -> dict:
    """Decode a serial byte string into a Python dictionary.

    Args:
        data: Raw bytes received from the serial port. May include
              leading/trailing whitespace or the newline delimiter.

    Returns:
        Parsed dictionary, e.g. ``{"b": 1, "t": 1, "s": 90}``.
        The ``"t"`` key defaults to 0 if not present (backward-compatible
        with messages produced before the type field was added).
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
    # Backward compatibility: "t" is optional
    result["t"] = int(result["t"]) if "t" in result else 0
    if "s" in result:
        result["s"] = int(result["s"])
    else:
        # Backward compatibility: infer angle from detection
        result["s"] = _ANGLE_BOTTLE if result["b"] else _ANGLE_NO_BOTTLE
    return result
