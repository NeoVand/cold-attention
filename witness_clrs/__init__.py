"""A looped pair block that counts and compares witnesses, trained on CLRS-30 (see README.md).

Light at import: the data generator and the scorer work without jax; the model's names load on first use."""
from .tasks import TASKS, GRAPH_TASKS, SEQ_TASKS, OUTPUTS, NAMES

__version__ = '0.1.0'
_LAZY = {'init_params': 'model', 'forward': 'model', 'make_rest_read': 'model', 'count_params': 'model',
         'registered_read': 'evaluate', 'score': 'scoring', 'save': 'weights', 'load': 'weights', 'load_hub': 'weights', 'load_shared': 'weights', 'load_shared_hub': 'weights'}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f'.{_LAZY[name]}', __name__), name)
    raise AttributeError(name)
