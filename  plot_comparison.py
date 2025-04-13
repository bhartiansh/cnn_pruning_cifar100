# plot_comparison.py

import json
import matplotlib.pyplot as plt

def plot_metrics(file_path, label):
    with open(file_path, "r") as f:
        history = json.load(f)

    plt.plot(history["val_accuracy"], label=f"{label} Val Acc")
    plt.plot(history["accuracy"], label=f"{label} Train Acc")

plot_metrics("results/baseline_metrics.json", "MobileNetV2")
plt.legend()
plt.title("Accuracy Comparison")
plt.show()