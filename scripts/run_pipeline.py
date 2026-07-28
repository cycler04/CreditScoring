#!/usr/bin/env python3
"""Run the full local checklist pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_scoring.pipeline import run_pipeline


if __name__ == "__main__":
    print(json.dumps(run_pipeline(PROJECT_ROOT), indent=2))
