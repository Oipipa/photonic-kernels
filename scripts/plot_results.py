import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES_DIR = os.path.join(ROOT, "results")
CSV_PATH = os.path.join(RES_DIR, "accuracies.csv")
OUT_PATH = os.path.join(RES_DIR, "figure4c_like.png")

df = pd.read_csv(CSV_PATH)
methods_order = ["quantum", "classical", "rbf", "poly", "linear"]
methods_present = [m for m in methods_order if m in df["method"].unique()]

plt.figure()
for m in methods_present:
    sub = df[df["method"] == m].sort_values("size")
    sizes = sub["size"].to_numpy()
    means = sub["mean"].to_numpy()
    stds = sub["std"].to_numpy()
    plt.errorbar(sizes, means, yerr=stds, marker="o", linestyle="-", capsize=3, label=m)

plt.xlabel("Dataset size N")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)
plt.xticks(sorted(df["size"].unique()))
plt.grid(True, alpha=0.3)
plt.legend(title="Method")
plt.tight_layout()
os.makedirs(RES_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150)
print(f"FIGURE_SAVED: {os.path.relpath(OUT_PATH, ROOT)}")
print("UPDATE_LOG")
print("Loaded aggregated accuracies and plotted mean±std versus dataset size for methods: quantum, classical, rbf, poly, and linear.")
print("Used a single figure with error bars to mirror the paper-style summary; y-axis constrained to [0,1] for comparability.")
print("Saved the figure to results/figure4c_like.png for inclusion in reports and README.")
