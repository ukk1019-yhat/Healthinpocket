import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


CLASS_LABELS = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def plot_training_history(history: Dict, output_path: str | Path = "outputs/plots/training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss Curves")
    acc = [m["accuracy"] for m in history["val_metrics"]]
    kappa = [m["kappa"] for m in history["val_metrics"]]
    axes[1].plot(epochs, acc, label="Accuracy")
    axes[1].plot(epochs, kappa, label="Kappa")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].set_title("Validation Metrics")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], output_path: str | Path = "outputs/plots/confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path = "outputs/plots/label_distribution.png"):
    plt.figure(figsize=(8, 5))
    counts = df["diagnosis"].value_counts().sort_index()
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    bars = plt.bar([CLASS_LABELS[i] for i in counts.index], counts.values, color=colors)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", fontsize=10)
    plt.xlabel("DR Severity")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    output_path: str | Path = "outputs/plots/sample_predictions.png",
    max_samples: int = 16,
):
    n = min(len(images), max_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(images[i])
        title = f"True: {CLASS_LABELS[true_labels[i]]}\nPred: {CLASS_LABELS[pred_labels[i]]}"
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
