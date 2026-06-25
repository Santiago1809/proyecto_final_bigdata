"""Detection event logger that writes to an Excel spreadsheet.

Records each positive bottle detection with timestamp and class name
for traceability and offline analysis.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd


_COOLDOWN = 2.0       # minimum seconds between log entries
_FILEPATH = "registro_envases.xlsx"
_COLUMNS = ["Fecha", "Hora", "Objeto"]


class ExcelLogger:
    """Logs detection events to an Excel file with rate limiting.

    Args:
        filepath: Path to the Excel file (default ``registro_envases.xlsx``).
        cooldown: Minimum seconds between consecutive log entries.
    """

    def __init__(
        self,
        filepath: str = _FILEPATH,
        cooldown: float = _COOLDOWN,
    ) -> None:
        self.filepath = filepath
        self.cooldown = cooldown
        self._last_log_time = 0.0

    def log(self, class_name: str) -> bool:
        """Write a detection row to the Excel file.

        Args:
            class_name: Detected class label (e.g. ``"pool_verde"``).

        Returns:
            ``True`` if the entry was written, ``False`` if skipped
            due to the cooldown window.
        """
        now = time.time()
        if now - self._last_log_time < self.cooldown:
            return False
        self._last_log_time = now

        fecha = datetime.now().strftime("%Y-%m-%d")
        hora = datetime.now().strftime("%H:%M:%S")
        row = pd.DataFrame([[fecha, hora, class_name]], columns=_COLUMNS)

        if os.path.exists(self.filepath):
            existing = pd.read_excel(self.filepath)
            df = pd.concat([existing, row], ignore_index=True)
        else:
            df = row

        df.to_excel(self.filepath, index=False)
        return True
