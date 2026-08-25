"""Visualization utilities for SAE experiments."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_ablation_comparison(results_dict: Dict[str, Dict], save_path: str) -> None:
    labels = ["baseline", "binding", "random"]
    values = [
        results_dict.get("baseline", {}).get("baseline_accuracy", 0.0),
        results_dict.get("binding", {}).get("ablated_accuracy", 0.0),
        results_dict.get("random", {}).get("ablated_accuracy", 0.0),
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Ablation comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
