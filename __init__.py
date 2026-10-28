"""Local SpEdE competition custom nodes: no AccGS or external APIs.

For standalone export, place this package __init__.py at the root of the
custom-node folder along with spede_core and spede_assets.py.
"""
from __future__ import annotations
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parent
# The pack's absolute imports (spede_core, spede_assets) resolve from the
# directory that holds them: this one in the exported pack, its parent in the
# development repository. Appended, not inserted first, so this directory's
# nodes.py never shadows another top-level ``nodes`` module (the repository's
# adapter pack) already importable earlier on the path.
for _p in (_ROOT, _ROOT.parent):
    if (_p/"spede_core").is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__=["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS"]
