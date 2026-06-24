"""Tests for src.protocol.message encode/decycle roundtrip.

Verifies:
  - encode(bottle=True)  produces b'{"b":1}\\n'
  - encode(bottle=False) produces b'{"b":0}\\n'
  - decode recovers the original dict
  - malformed JSON raises JSONDecodeError
  - missing "b" key raises KeyError
  - empty bytes raises JSONDecodeError
"""

import json
import unittest

from src.protocol.message import decode, encode


class TestMessageEncodeDecode(unittest.TestCase):
    """Message encoding and decoding roundtrips."""

    def test_encode_bottle(self):
        """encode(True) yields b'{"b":1}\\n'."""
        result = encode(True)
        self.assertEqual(result, b'{"b":1}\n')

    def test_encode_no_bottle(self):
        """encode(False) yields b'{"b":0}\\n'."""
        result = encode(False)
        self.assertEqual(result, b'{"b":0}\n')

    def test_decode_bottle(self):
        """decode(b'{"b":1}') returns dict with b=1."""
        result = decode(b'{"b":1}')
        self.assertEqual(result, {"b": 1})

    def test_decode_no_bottle(self):
        """decode(b'{"b":0}') returns dict with b=0."""
        result = decode(b'{"b":0}')
        self.assertEqual(result, {"b": 0})

    def test_decode_with_newline(self):
        """decode handles trailing newline."""
        result = decode(b'{"b":1}\n')
        self.assertEqual(result, {"b": 1})

    def test_decode_with_whitespace(self):
        """decode handles leading/trailing whitespace."""
        result = decode(b'  {"b":0}  ')
        self.assertEqual(result, {"b": 0})

    def test_roundtrip_bottle(self):
        """encode(True) -> bytes -> decode -> {"b": 1}."""
        data = encode(True)
        result = decode(data)
        self.assertEqual(result, {"b": 1})

    def test_roundtrip_no_bottle(self):
        """encode(False) -> bytes -> decode -> {"b": 0}."""
        data = encode(False)
        result = decode(data)
        self.assertEqual(result, {"b": 0})

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

    def test_encode_compact(self):
        """Verify no spaces in JSON output for compactness."""
        data = encode(True)
        self.assertNotIn(b" ", data)
        data = encode(False)
        self.assertNotIn(b" ", data)


if __name__ == "__main__":
    unittest.main()
