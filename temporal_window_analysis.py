# temporal_window_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulated performance results (replace with real outputs)
tau = [1, 2, 3, 4]
heuristic = {
    "Precision": [0.71, 0.79, 0.86, 0.83],
    "Recall":    [0.67, 0.76, 0.83, 0.85],
    "F1":        [0.69, 0.77, 0.84, 0.84]
}
ml = {
    "Precision": [0.75, 0.82, 0.88, 0.85],
    "Recall":    [0.70, 0.79, 0.87, 0.86],
    "F1":        [0.72, 0.80, 0.87, 0.85]
}

heuristic_df = pd.DataFrame(heuristic, index=tau)
ml_df = pd.DataFrame(ml, index=tau)

plt.figure(figsize=(7,5))
for metric in ["Precision", "Recall", "F1"]:
    plt.plot(tau, heuristic_df[metric], marker='o', linestyle='--', label=f"Heuristic {metric}")
    plt.plot(tau, ml_df[metric], marker='s', linestyle='-', label=f"ML {metric}")

plt.xlabel("Temporal Window τ (seconds)")
plt.ylabel("Score")
plt.title("Temporal Window Sensitivity: Heuristic vs ML")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig("../results/window_sensitivity_plot.png", dpi=300)
plt.show()
