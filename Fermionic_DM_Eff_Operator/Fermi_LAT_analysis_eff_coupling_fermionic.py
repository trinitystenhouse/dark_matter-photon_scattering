import importlib.util
import os
import sys


_HERE = os.path.dirname(__file__)
_TARGET = os.path.join(_HERE, "Fermi-LAT_analysis_eff_coupling_fermionic.py")

_spec = importlib.util.spec_from_file_location("_fermionic_hyphenated", _TARGET)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_TARGET}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _mod
_spec.loader.exec_module(_mod)
