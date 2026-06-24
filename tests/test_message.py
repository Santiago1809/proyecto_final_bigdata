"""Tests for src.protocol.message encode/decode roundtrip.

Verifies:
  - encode(bottle=True)  produces b'{"b":1,"t":0,"s":90}\\n'
  - encode(bottle=False) produces b'{"b":0,"t":0,"s":180}\\n'
  - encode with bottle_type 1/2 produces correct "t" field
  - custom servo angles
  - decode recovers the original dict with "t" and "s" keys
  - backward-compatible decode of old messages without "t" or "s"
  - malformed JSON raises JSONDecodeError
  - missing "b" key raises KeyError
  - empty bytes raises JSONDecodeError
  - BottleType enum values
"""

import json
import unittest

from src.protocol.message import BottleType, decode, encode


class TestBottleType(unittest.TestCase):
    """BottleType enum values."""

    def test_bottle_type_values(self):
        """BottleType has correct int values."""
        self.assertEqual(BottleType.NONE, 0)
        self.assertEqual(BottleType.POOL_VERDE, 1)
        self.assertEqual(BottleType.HATSU_MORADO, 2)

    def test_bottle_type_names(self):
        """BottleType names match expectations."""
        self.assertEqual(BottleType(0).name, "NONE")
        self.assertEqual(BottleType(1).name, "POOL_VERDE")
        self.assertEqual(BottleType(2).name, "HATSU_MORADO")


class TestMessageEncodeDecode(unittest.TestCase):
    """Message encoding and decoding roundtrips."""

    # --- encode ---

    def test_encode_bottle_default_type(self):
        """encode(True) yields b'{"b":1,"t":0,"s":90}\\n'."""
        result = encode(True)
        self.assertEqual(result, b'{"b":1,"t":0,"s":90}\n')

    def test_encode_no_bottle_default_type(self):
        """encode(False) yields b'{"b":0,"t":0,"s":180}\\n'."""
        result = encode(False)
        self.assertEqual(result, b'{"b":0,"t":0,"s":180}\n')

    def test_encode_bottle_with_type_1(self):
        """encode(True, bottle_type=1) yields b'{"b":1,"t":1,"s":90}\\n'."""
        result = encode(True, bottle_type=1)
        self.assertEqual(result, b'{"b":1,"t":1,"s":90}\n')

    def test_encode_bottle_with_type_2(self):
        """encode(True, bottle_type=2) yields b'{"b":1,"t":2,"s":90}\\n'."""
        result = encode(True, bottle_type=2)
        self.assertEqual(result, b'{"b":1,"t":2,"s":90}\n')

    def test_encode_bottle_custom_angle(self):
        """encode(True, servo_angle=45) uses the provided angle."""
        result = encode(True, servo_angle=45)
        self.assertEqual(result, b'{"b":1,"t":0,"s":45}\n')

    def test_encode_no_bottle_custom_angle(self):
        """encode(False, servo_angle=0) uses the provided angle."""
        result = encode(False, servo_angle=0)
        self.assertEqual(result, b'{"b":0,"t":0,"s":0}\n')

    def test_encode_compact_no_spaces(self):
        """JSON output has no spaces for compactness."""
        for args in [(True,), (False,), (True, 1), (False, 2, 120)]:
            data = encode(*args)
            self.assertNotIn(b" ", data, f"spaces in encode{args}")

    # --- decode ---

    def test_decode_bottle_with_type_and_angle(self):
        """decode(b'{"b":1,"t":1,"s":90}') returns all keys."""
        result = decode(b'{"b":1,"t":1,"s":90}')
        self.assertEqual(result, {"b": 1, "t": 1, "s": 90})

    def test_decode_no_bottle_with_type_and_angle(self):
        """decode(b'{"b":0,"t":0,"s":180}') returns all keys."""
        result = decode(b'{"b":0,"t":0,"s":180}')
        self.assertEqual(result, {"b": 0, "t": 0, "s": 180})

    def test_decode_backward_compatible_no_type(self):
        """decode old message without "t" defaults t=0."""
        result = decode(b'{"b":1}')
        self.assertEqual(result["b"], 1)
        self.assertEqual(result["t"], 0)
        self.assertEqual(result["s"], 90)

    def test_decode_backward_compatible_no_type_no_bottle(self):
        """decode old message without "t" defaults t=0 for no bottle."""
        result = decode(b'{"b":0}')
        self.assertEqual(result["b"], 0)
        self.assertEqual(result["t"], 0)
        self.assertEqual(result["s"], 180)

    def test_decode_backward_compatible_no_angle(self):
        """decode message without "s" infers angle from b."""
        result = decode(b'{"b":1,"t":2}')
        self.assertEqual(result["b"], 1)
        self.assertEqual(result["t"], 2)
        self.assertEqual(result["s"], 90)

    def test_decode_with_newline(self):
        """decode handles trailing newline."""
        result = decode(b'{"b":1,"t":0,"s":90}\n')
        self.assertEqual(result, {"b": 1, "t": 0, "s": 90})

    def test_decode_with_whitespace(self):
        """decode handles leading/trailing whitespace."""
        result = decode(b'  {"b":0,"t":0,"s":180}  ')
        self.assertEqual(result, {"b": 0, "t": 0, "s": 180})

    def test_decode_custom_angle(self):
        """decode parses any valid angle."""
        result = decode(b'{"b":1,"t":1,"s":45}')
        self.assertEqual(result, {"b": 1, "t": 1, "s": 45})
        result = decode(b'{"b":0,"t":0,"s":15}')
        self.assertEqual(result, {"b": 0, "t": 0, "s": 15})

    # --- roundtrips ---

    def test_roundtrip_bottle(self):
        """encode(True) -> bytes -> decode -> {"b": 1, "t": 0, "s": 90}."""
        data = encode(True)
        result = decode(data)
        self.assertEqual(result, {"b": 1, "t": 0, "s": 90})

    def test_roundtrip_no_bottle(self):
        """encode(False) -> bytes -> decode -> {"b": 0, "t": 0, "s": 180}."""
        data = encode(False)
        result = decode(data)
        self.assertEqual(result, {"b": 0, "t": 0, "s": 180})

    def test_roundtrip_with_type(self):
        """roundtrip with bottle_type preserves type."""
        data = encode(True, bottle_type=2)
        result = decode(data)
        self.assertEqual(result, {"b": 1, "t": 2, "s": 90})

    def test_roundtrip_custom_angle(self):
        """roundtrip with custom angle is preserved."""
        data = encode(False, servo_angle=42)
        result = decode(data)
        self.assertEqual(result, {"b": 0, "t": 0, "s": 42})

    # --- error handling ---

    def test_malformed_json_raises(self):
        """Malformed input raises JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError):
            decode(b"not-json")

    def test_empty_bytes_raises(self):
        """Empty bytes raises JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError):
            decode(b"")

    def test_missing_key_raises(self):
        """Payload without 'b' key raises KeyError."""
        with self.assertRaises(KeyError):
            decode(b'{"x": 1}')
        data = encode(False)
        self.assertNotIn(b" ", data)


if __name__ == "__main__":
    unittest.main()
