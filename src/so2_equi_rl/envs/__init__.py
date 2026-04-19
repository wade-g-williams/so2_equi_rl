"""EnvWrapper lives in wrapper.py. The hhe __file__ patch below fires on any entry
into envs/*, before wrapper.py's chained imports.
"""

import os

# hhe ships an empty __init__.py, so __file__ is None and random_object.py crashes on
# os.path.dirname(hhe.__file__). Patch it before any downstream import hits the chain.
import helping_hands_rl_envs as _hhe

if _hhe.__file__ is None:
    # __path__ is a _NamespacePath (iterable, not subscriptable), so list() it.
    _hhe.__file__ = os.path.join(list(_hhe.__path__)[0], "__init__.py")

del os, _hhe
