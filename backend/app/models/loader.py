import sys
import os

# Add parent directory to path (go up 3 levels: loader.py -> models -> app -> backend -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import load_model, load_scalers

_models_cache = {}
_scalers_cache = {}

def get_model(capacity: str):
    """Get cached model or load if not cached."""
    if capacity not in _models_cache:
        from core.config.config import configs
        cfg = configs["models_config"][capacity]
        _models_cache[capacity] = load_model(cfg['model_path'], capacity)
        _scalers_cache[capacity] = load_scalers(cfg['metrics_dir'])

    return _models_cache[capacity], _scalers_cache[capacity]

def preload_all_models():
    """Preload all three capacity models at startup."""
    for capacity in ["900", "1300", "2200"]:
        get_model(capacity)
    return _models_cache, _scalers_cache
