# temporal_window_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os

# Load dataset
data_path = os.path.join("..", "data", "clips_annotations.csv")
df = pd.read_csv(data_path)

# Define base features and labels
base_features = ["pause_duration", "lateral_offset", "speed_change", "angle_change"]
y = df["label"]

# Temporal windows to evaluate (τ in seconds)
tau_values = [1, 2, 3, 4]

results = {"τ": [], "Precision": [], "Recall": [], "F1": []}

for tau in tau_values:
    # Simulate temporal aggregation: scale features slightly with τ
    X = df[base_features] * (tau / 3.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute metrics
    p = precision_score(y_test, y_pred, average="macro")
    r = recall_score(y_test, y_pred, average="macro")
    f = f1_score(y_test, y_pred, average="macro")

    results["τ"].append(tau)
    results["Precision"].append(p)
    results["Recall"].append(r)
    results["F1"].append(f)

results_df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(results_df["τ"], results_df["Precision"], "o--", label="Precision (ML)")
plt.plot(results_df["τ"], results_df["Recall"], "s--", label="Recall (ML)")
plt.plot(results_df["τ"], results_df["F1"], "d-", label="F1 (ML)")

plt.xlabel("Temporal Window τ (seconds)")
plt.ylabel("Score")
plt.title("Temporal Window Sensitivity (Decision Tree Classifier)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/window_sensitivity_plot.png", dpi=300)
plt.show()

# Save metrics for reference
results_df.to_csv("../results/temporal_sensitivity_results.csv", index=False)
print(results_df)
