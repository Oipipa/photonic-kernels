import numpy as np
import random

_GLOBAL_RNG = None

def set_global_seed(seed: int) -> None:
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)
    random.seed(seed)

def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = np.ones_like(d, dtype=np.complex128)
    mask = np.abs(d) > 0
    ph[mask] = d[mask] / np.abs(d[mask])
    u = q @ np.diag(np.conj(ph))
    return u

def psd_sqrt(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    h = (mat + mat.conj().T) * 0.5
    w, v = np.linalg.eigh(h)
    w_clipped = np.clip(w, eps, None)
    s = np.sqrt(w_clipped)
    return v @ (np.diag(s) @ v.conj().T)

def permanent2x2(a11, a12, a21, a22) -> complex:
    return a11 * a22 + a12 * a21

def permanent_ryser(A: np.ndarray) -> complex:
    A = np.asarray(A)
    n = A.shape[1]
    total = 0.0 + 0.0j
    for mask in range(1, 1 << n):
        bits = [(mask >> j) & 1 for j in range(n)]
        k = sum(bits)
        sums = A[:, bits == np.array(1)]
        row_sums = sums.sum(axis=1) if sums.size else np.zeros(A.shape[0], dtype=A.dtype)
        prod = np.prod(row_sums)
        total += ((-1) ** (n - k)) * prod
    return total