import numpy as np

try:
    from .utils import set_global_seed, haar_unitary
except Exception:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from photonic_kernel.utils import set_global_seed, haar_unitary

def build_mixers(seed: int = 2025) -> list:
    rng = np.random.default_rng(seed)
    return [haar_unitary(6, rng) for _ in range(5)]

def encode_unitary(x: np.ndarray, mixers: list) -> np.ndarray:
    theta = np.asarray(x, dtype=float).ravel()
    if theta.size < 30:
        theta = np.pad(theta, (0, 30 - theta.size))
    theta = theta[:30]
    blocks = [theta[i*6:(i+1)*6] for i in range(5)]
    Ds = [np.diag(np.exp(1j * 2.0 * np.pi * b)) for b in blocks]
    U = mixers[4] @ Ds[4] @ mixers[3] @ Ds[3] @ mixers[2] @ Ds[2] @ mixers[1] @ Ds[1] @ mixers[0] @ Ds[0]
    return U

def make_unitary_product(xi: np.ndarray, xj: np.ndarray, mixers: list) -> np.ndarray:
    Ui = encode_unitary(xi, mixers)
    Uj = encode_unitary(xj, mixers)
    return Uj.conj().T @ Ui

if __name__ == "__main__":
    set_global_seed(777)
    rng = np.random.default_rng(777)
    xi = rng.random(27)
    xj = rng.random(27)
    mixers = build_mixers()
    U = encode_unitary(xi, mixers)
    W = make_unitary_product(xi, xj, mixers)
    err_U = np.linalg.norm(U.conj().T @ U - np.eye(6), ord="fro")
    err_W = np.linalg.norm(W.conj().T @ W - np.eye(6), ord="fro")
    print(f"unitary_U_error={err_U}")
    print(f"unitary_W_error={err_W}")
