import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dirs(*dirs: str) -> None:
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def write_training_log(rows, output_path: str) -> None:
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows, output_dir: str, method: str = "") -> None:
    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]

    # Use val if test not recorded, otherwise test
    has_test = any(row.get("test_loss") not in (None, "") for row in rows)
    eval_loss_key = "test_loss" if has_test else "val_loss"
    eval_acc_key = "test_acc" if has_test else "val_acc"
    eval_label = "Test" if has_test else "Val"

    eval_loss = [row.get(eval_loss_key) or float("nan") for row in rows]
    eval_acc = [row.get(eval_acc_key) or float("nan") for row in rows]

    title_suffix = f" - {method}" if method else ""

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, eval_loss, marker="o", label=f"{eval_label} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CIFAR-10 Loss{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, eval_acc, marker="o", label=f"{eval_label} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"CIFAR-10 Accuracy{title_suffix}")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(9, 8))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CIFAR10_CLASSES))
    plt.xticks(tick_marks, CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CIFAR10_CLASSES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            color = "white" if confusion_matrix[i, j] > threshold else "black"
            plt.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
