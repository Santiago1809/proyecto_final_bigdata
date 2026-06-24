"""Serial protocol for host-to-ESP32 communication.

Uses compact JSON over USB serial at 9600 baud.
Format: {"b":1}\\n for bottle, {"b":0}\\n for not bottle.
"""

import json

_DELIMITER = b"\n"


def encode(bottle_detected: bool) -> bytes:
    """Encode a detection result into a serial-ready JSON byte string.

    Args:
        bottle_detected: True if a bottle was detected, False otherwise.

    Returns:
        UTF-8 encoded JSON bytes with newline delimiter, e.g. b'{"b":1}\\n'.

    """
    payload = {"b": 1 if bottle_detected else 0}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + _DELIMITER


def decode(data: bytes) -> dict:
    """Decode a serial byte string into a Python dictionary.

    Args:
        data: Raw bytes received from the serial port. May include
              leading/trailing whitespace or the newline delimiter.

    Returns:
        Parsed dictionary, e.g. {"b": 1} or {"b": 0}.

    Raises:
        json.JSONDecodeError: If the data is not valid JSON.
        KeyError: If the "b" key is missing.

    """
    decoded = data.decode("utf-8").strip()
    if not decoded:
        raise json.JSONDecodeError("Empty data", "", 0)
    result = json.loads(decoded)
    if "b" not in result:
        raise KeyError(f"Missing required key 'b' in message: {decoded!r}")
    result["b"] = int(result["b"])
    return result
