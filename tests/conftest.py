"""Pytest configuration for stable local imports from the project root."""

from __future__ import annotations

import sys
from pathlib import Path


# Keep imports simple during the scaffold stage by adding the repo root once.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))