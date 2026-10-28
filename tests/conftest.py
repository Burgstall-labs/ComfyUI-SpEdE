"""Make the pack importable however pytest is started.

The tests import ``spede_core`` and the pack's root modules by name; both the
development repository and the exported node pack keep them one level above
this folder, so that directory goes first on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
