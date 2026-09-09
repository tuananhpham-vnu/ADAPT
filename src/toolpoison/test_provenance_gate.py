"""Compatibility alias for src.adapt.test_provenance_gate."""
import sys
from importlib import import_module
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_impl = import_module("src.adapt.test_provenance_gate")
if __name__ == "__main__":
    if hasattr(_impl, "main"):
        _impl.main()
    else:
        import runpy
        runpy.run_module("src.adapt.test_provenance_gate", run_name="__main__")
else:
    sys.modules[__name__] = _impl
