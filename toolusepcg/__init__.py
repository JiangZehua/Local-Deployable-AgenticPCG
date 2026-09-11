"""ToolUsePCG - LLM agent system for procedural content generation using tool calling."""

__version__ = "0.1.0"

from pathlib import Path as _Path
import sys as _sys
_vendor = str(_Path(__file__).resolve().parents[1] / "vendor")
if _vendor not in _sys.path:
    _sys.path.insert(0, _vendor)
