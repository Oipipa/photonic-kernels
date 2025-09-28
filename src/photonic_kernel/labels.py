import numpy as np

try:
    from .utils import set_global_seed, psd_sqrt
    from .unitary import build_mixers
    from .kernels import compute_kernels
except Exception:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from photonic_kernel.utils import set_global_seed, psd_sqrt
    from photonic_kernel.unitary import build_mixers
    from photonic_kernel.kernels import compute_kernels

def geometric_difference_labels(KQ: np.ndarray, KC: np.ndarray) -> np.ndarray:
    SQ = psd_sqrt(np.asarray(KQ, dtype=float))
    KC = np.asarray(KC, dtype=float)
    KC_reg = KC + 1e-8 * np.eye(KC.shape[0])
    KC_inv = np.linalg.inv(KC_reg)
    M = SQ @ KC_inv @ SQ
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    v = V[:, -1]
    y = SQ @ v
    labels = np.sign(np.real(y))
    labels[labels == 0] = 1
    return labels.astype(int)

if __name__ == "__main__":
    set_global_seed(2027)
    rng = np.random.default_rng(2027)
    X = rng.random((10, 27))
    mixers = build_mixers()
    KQ, KC = compute_kernels(X, mixers, 0, 1)
    y = geometric_difference_labels(KQ, KC)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == -1))
    print(f"n_pos={n_pos}")
    print(f"n_neg={n_neg}")