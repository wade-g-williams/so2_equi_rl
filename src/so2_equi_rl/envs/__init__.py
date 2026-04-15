"""Env wrappers. `EnvWrapper` around BulletArm close-loop tasks lives in
wrapper.py. The hhe __file__ patch below runs on any entry into envs/* so
it fires before wrapper.py's chained imports.
"""

import os

# helping_hands_rl_envs ships an empty __init__.py, so __file__ is None under
# Py3.7 and random_object.py crashes on os.path.dirname(hhe.__file__). Patch
# it before any downstream import hits the chain.
import helping_hands_rl_envs as _hhe

if _hhe.__file__ is None:
    # __path__ is a _NamespacePath (iterable, not subscriptable); list() it.
    _hhe.__file__ = os.path.join(list(_hhe.__path__)[0], "__init__.py")

del os, _hhe
