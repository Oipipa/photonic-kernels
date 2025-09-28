import numpy as np

try:
    from .utils import set_global_seed, haar_unitary
except Exception:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from photonic_kernel.utils import set_global_seed, haar_unitary

def collision_free_pairs(m: int = 6):
    return [(r, s) for r in range(m) for s in range(r + 1, m)]

def two_photon_probs(W: np.ndarray, p: int, q: int, indist: bool):
    W = np.asarray(W)
    m = W.shape[0]
    out = {}
    for r, s in collision_free_pairs(m):
        if indist:
            amp = W[r, p] * W[s, q] + W[r, q] * W[s, p]
            prob = np.abs(amp) ** 2
        else:
            prob = (np.abs(W[r, p]) ** 2) * (np.abs(W[s, q]) ** 2) + (np.abs(W[r, q]) ** 2) * (np.abs(W[s, p]) ** 2)
        out[(r, s)] = float(np.real(prob))
    return out

def kernel_projection_same(W: np.ndarray, p: int, q: int, indist: bool) -> float:
    probs = two_photon_probs(W, p, q, indist)
    total = float(sum(probs.values()))
    key = (min(p, q), max(p, q))
    numer = float(probs.get(key, 0.0))
    return 0.0 if total == 0.0 else numer / total

if __name__ == "__main__":
    set_global_seed(2025)
    rng = np.random.default_rng(2025)
    W = haar_unitary(6, rng)
    p, q = 0, 1
    cf_ind = two_photon_probs(W, p, q, True)
    cf_cls = two_photon_probs(W, p, q, False)
    cf_mass_indist = float(sum(cf_ind.values()))
    cf_mass_class = float(sum(cf_cls.values()))
    print(f"cf_mass_indist={cf_mass_indist}")
    print(f"cf_mass_class={cf_mass_class}")
    kq = kernel_projection_same(W, p, q, True)
    kc = kernel_projection_same(W, p, q, False)
    print(f"kq={kq}")
    print(f"kc={kc}")
