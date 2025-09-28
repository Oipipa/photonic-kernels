import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from .utils import set_global_seed
    from .unitary import build_mixers
    from .kernels import compute_kernels
    from .labels import geometric_difference_labels
except Exception:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from photonic_kernel.utils import set_global_seed
    from photonic_kernel.unitary import build_mixers
    from photonic_kernel.kernels import compute_kernels
    from photonic_kernel.labels import geometric_difference_labels

def evaluate_all(X: np.ndarray, y: np.ndarray, KQ: np.ndarray, KC: np.ndarray, seed: int = 1337) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = y.shape[0]
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(idx, train_size=2/3, test_size=1/3, stratify=y, random_state=seed)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    KQ_train = KQ[np.ix_(idx_train, idx_train)]
    KQ_test = KQ[np.ix_(idx_test, idx_train)]
    KC_train = KC[np.ix_(idx_train, idx_train)]
    KC_test = KC[np.ix_(idx_test, idx_train)]

    svm_q = SVC(kernel="precomputed", random_state=seed)
    svm_q.fit(KQ_train, y_train)
    yq = svm_q.predict(KQ_test)
    acc_q = float(accuracy_score(y_test, yq))

    svm_c = SVC(kernel="precomputed", random_state=seed)
    svm_c.fit(KC_train, y_train)
    yc = svm_c.predict(KC_test)
    acc_c = float(accuracy_score(y_test, yc))

    svm_lin = SVC(kernel="linear", random_state=seed)
    svm_lin.fit(X_train, y_train)
    yl = svm_lin.predict(X_test)
    acc_lin = float(accuracy_score(y_test, yl))

    svm_poly = SVC(kernel="poly", degree=3, random_state=seed)
    svm_poly.fit(X_train, y_train)
    yp = svm_poly.predict(X_test)
    acc_poly = float(accuracy_score(y_test, yp))

    svm_rbf = SVC(kernel="rbf", random_state=seed)
    svm_rbf.fit(X_train, y_train)
    yr = svm_rbf.predict(X_test)
    acc_rbf = float(accuracy_score(y_test, yr))

    return {"quantum": acc_q, "classical": acc_c, "linear": acc_lin, "poly": acc_poly, "rbf": acc_rbf}

if __name__ == "__main__":
    set_global_seed(1337)
    rng = np.random.default_rng(1337)
    X = rng.random((30, 27))
    mixers = build_mixers()
    KQ, KC = compute_kernels(X, mixers, 0, 1)
    y = geometric_difference_labels(KQ, KC)
    accs = evaluate_all(X, y, KQ, KC, seed=1337)
    print(f"ACCURACIES: {accs}")
