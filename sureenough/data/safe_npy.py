import pickle

import numpy as np

_NUMPY_MULTIARRAY = {"numpy.core.multiarray", "numpy._core.multiarray"}
_ALLOWED = {
    **{(m, n): None for m in _NUMPY_MULTIARRAY for n in ("_reconstruct", "scalar")},
    ("numpy", "ndarray"): None,
    ("numpy", "dtype"): None,
    **{(b, n): None for b in ("builtins", "__builtin__") for n in ("dict", "list", "str")},
}


class _AllowListUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        if module in _NUMPY_MULTIARRAY:
            module = "numpy._core.multiarray"
        if module == "__builtin__":
            module = "builtins"
        return super().find_class(module, name)


def load_pickled_npy(fileobj) -> dict:
    version = np.lib.format.read_magic(fileobj)
    if version == (1, 0):
        _, _, dtype = np.lib.format.read_array_header_1_0(fileobj)
    elif version == (2, 0):
        _, _, dtype = np.lib.format.read_array_header_2_0(fileobj)
    else:
        raise ValueError(f"unsupported .npy version {version}")
    if dtype != np.dtype(object):
        raise ValueError(f"expected a pickled object array, got dtype {dtype}")
    obj = _AllowListUnpickler(fileobj).load()
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"expected a dict payload, got {type(obj).__name__}")
    return obj
