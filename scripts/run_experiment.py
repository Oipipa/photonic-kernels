import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from photonic_kernel.unitary import build_mixers
from photonic_kernel.kernels import compute_kernels
from photonic_kernel.labels import geometric_difference_labels
from photonic_kernel.eval import evaluate_all

def main():
    sizes = [40, 60, 80, 100]
    seeds = [1001, 1002, 1003, 1004, 1005]
    methods = ["quantum", "classical", "linear", "poly", "rbf"]
    mixers = build_mixers()
    records = []
    for N in sizes:
        accs_by_method = {m: [] for m in methods}
        for s in seeds:
            rng = np.random.default_rng(s)
            X = rng.random((N, 27))
            KQ, KC = compute_kernels(X, mixers, 0, 1)
            y = geometric_difference_labels(KQ, KC)
            acc = evaluate_all(X, y, KQ, KC, seed=s)
            for m in methods:
                accs_by_method[m].append(acc[m])
        for m in methods:
            vals = np.array(accs_by_method[m], dtype=float)
            records.append({"size": N, "method": m, "mean": float(vals.mean()), "std": float(vals.std(ddof=0))})
    df = pd.DataFrame(records, columns=["size", "method", "mean", "std"])
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "accuracies.csv")
    df.to_csv(out_path, index=False)
    print("size,summary")
    for N in sizes:
        row = df[df["size"] == N]
        parts = []
        for m in methods:
            r = row[row["method"] == m].iloc[0]
            parts.append(f"{m}={r['mean']:.3f}±{r['std']:.3f}")
        print(f"{N}," + " ".join(parts))
    print("UPDATE_LOG")
    print("Built fixed mixers once, then evaluated across dataset sizes [40,60,80,100] and seeds [1001..1005].")
    print("For each (size,seed), generated X∈[0,1)^{N×27}, computed KQ and KC, derived labels via geometric-difference, and measured SVM accuracies for quantum, classical, and standard baselines.")
    print("Aggregated mean and standard deviation per method and size; wrote results/accuracies.csv with columns size,method,mean,std (20 rows).")
    print("Printed a compact summary table for quick inspection.")
    print("RUN_EXPERIMENT_DONE")

if __name__ == "__main__":
    main()
