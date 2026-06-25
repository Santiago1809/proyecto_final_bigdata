"""Tests for the ExcelLogger detection event logger.

Uses a temporary directory to avoid cluttering the workspace.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

import pandas as pd

from src.vision.excel_logger import ExcelLogger


class TestExcelLogger(unittest.TestCase):
    """ExcelLogger file creation, append, cooldown, and data integrity."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self.tmpdir.name, "registro_envases.xlsx")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creates_file_on_first_log(self):
        """First log() call creates the Excel file with header row."""
        logger = ExcelLogger(filepath=self.filepath)
        result = logger.log("pool_verde")
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.filepath))

        df = pd.read_excel(self.filepath)
        self.assertListEqual(list(df.columns), ["Fecha", "Hora", "Objeto"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Objeto"], "pool_verde")

    def test_appends_on_second_log(self):
        """Second log() call appends a new row, preserving previous data."""
        logger = ExcelLogger(filepath=self.filepath, cooldown=0.0)
        logger.log("pool_verde")
        logger.log("hatsu_morado")

        df = pd.read_excel(self.filepath)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Objeto"], "pool_verde")
        self.assertEqual(df.iloc[1]["Objeto"], "hatsu_morado")

    def test_cooldown_rate_limiting(self):
        """log() returns False if called within cooldown window."""
        logger = ExcelLogger(filepath=self.filepath, cooldown=10.0)
        first = logger.log("pool_verde")
        self.assertTrue(first)

        second = logger.log("hatsu_morado")
        self.assertFalse(second)

        # Only the first entry should exist
        df = pd.read_excel(self.filepath)
        self.assertEqual(len(df), 1)

    def test_cooldown_expires(self):
        """log() returns True again after the cooldown window passes."""
        logger = ExcelLogger(filepath=self.filepath, cooldown=0.01)
        logger.log("pool_verde")
        time.sleep(0.02)
        result = logger.log("hatsu_morado")
        self.assertTrue(result)

        df = pd.read_excel(self.filepath)
        self.assertEqual(len(df), 2)

    def test_timestamp_format(self):
        """Log entry has valid Fecha (date) and Hora (time) strings."""
        logger = ExcelLogger(filepath=self.filepath, cooldown=0.0)
        logger.log("pool_verde")

        df = pd.read_excel(self.filepath)
        fecha = str(df.iloc[0]["Fecha"])
        hora = str(df.iloc[0]["Hora"])
        # Basic format checks: YYYY-MM-DD and HH:MM:SS
        self.assertRegex(fecha, r"\d{4}-\d{2}-\d{2}")
        self.assertRegex(hora, r"\d{2}:\d{2}:\d{2}")


if __name__ == "__main__":
    unittest.main()
