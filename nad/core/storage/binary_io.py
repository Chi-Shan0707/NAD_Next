
import numpy as np, os

def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def create_memmap(path: str, dtype, shape, mode="w+"):
    ensure_parent(path)
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def open_memmap(path: str, dtype, shape=None, mode="r"):
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def write_array_atomic(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    ensure_parent(path)
    with open(tmp, "wb") as f:
        arr.tofile(f)
    os.replace(tmp, path)

def mmap_from_file(path: str, dtype):
    file_size = os.path.getsize(path)
    itemsize = np.dtype(dtype).itemsize
    n = file_size // itemsize
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,))
