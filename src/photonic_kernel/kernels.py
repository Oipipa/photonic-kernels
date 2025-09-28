import numpy as np

try:
    from .utils import set_global_seed
    from .unitary import build_mixers, make_unitary_product
    from .twophoton import kernel_projection_same
except Exception:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from photonic_kernel.utils import set_global_seed
    from photonic_kernel.unitary import build_mixers, make_unitary_product
    from photonic_kernel.twophoton import kernel_projection_same

def compute_kernels(X: np.ndarray, mixers, p: int = 0, q: int = 1):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    KQ = np.zeros((n, n), dtype=float)
    KC = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = X[i]
        for j in range(n):
            xj = X[j]
            W = make_unitary_product(xi, xj, mixers)
            a_q = float(kernel_projection_same(W, p, q, True))
            b_q = float(kernel_projection_same(W.conj().T, p, q, True))
            KQ[i, j] = np.sqrt(a_q * b_q)
            a_c = float(kernel_projection_same(W, p, q, False))
            b_c = float(kernel_projection_same(W.conj().T, p, q, False))
            KC[i, j] = np.sqrt(a_c * b_c)
    return KQ, KC

if __name__ == "__main__":
    set_global_seed(2026)
    rng = np.random.default_rng(2026)
    X = rng.random((12, 27))
    mixers = build_mixers()
    KQ, KC = compute_kernels(X, mixers, 0, 1)
    print(f"shape_KQ={KQ.shape}")
    print(f"shape_KC={KC.shape}")
    sym_err_KQ = float(np.linalg.norm(KQ - KQ.T, ord="fro"))
    sym_err_KC = float(np.linalg.norm(KC - KC.T, ord="fro"))
    print(f"sym_err_KQ={sym_err_KQ}")
    print(f"sym_err_KC={sym_err_KC}")
    print(f"range_KQ={float(np.min(KQ))},{float(np.max(KQ))}")
    print(f"range_KC={float(np.min(KC))},{float(np.max(KC))}")