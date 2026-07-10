import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "comparison"

def build_model_specs() -> list[dict]:
    legacy_simple_log = (
        ROOT.parent / "deep_learning" / "cifar10_cnn_baseline" / "results" / "training_log.csv"
    )
    current_simple_log = ROOT / "results" / "simple_cnn" / "training_log.csv"

    if legacy_simple_log.exists():
        simple_spec = {
            "name": "SimpleCNN",
            "log": legacy_simple_log,
            "source": "legacy_baseline_test",
            "params": "0.62M",
            "note": "Legacy 10-epoch baseline.",
        }
    else:
        simple_spec = {
            "name": "SimpleCNN",
            "log": current_simple_log,
            "source": "test",
            "params": "0.62M",
            "note": "10-epoch baseline evaluated on the CIFAR-10 test split.",
        }

    return [
        simple_spec,
        {
            "name": "ResNet20",
            "log": ROOT / "results" / "resnet20" / "training_log.csv",
            "source": "val",
            "params": "0.27M",
            "note": "Best validation checkpoint; includes 5-fold summary.",
        },
        {
            "name": "VGG16-BN",
            "log": ROOT / "results" / "vgg16_bn" / "training_log.csv",
            "source": "val",
            "params": "15.2M",
            "note": "Best validation accuracy among available methods.",
        },
    ]


def read_log(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def as_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def metric_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    epochs = []
    values = []
    for row in rows:
        value = as_float(row, key)
        if value is not None:
            epochs.append(as_int(row, "epoch"))
            values.append(value)
    return epochs, values


def best_metric(rows: list[dict], preferred_key: str):
    candidates = []
    for row in rows:
        value = as_float(row, preferred_key)
        if value is not None:
            candidates.append((value, as_int(row, "epoch"), row))
    if not candidates:
        return None, None, None
    return max(candidates, key=lambda item: item[0])


def format_pct(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def collect_model_summaries():
    summaries = []
    logs = {}

    for spec in build_model_specs():
        rows = read_log(spec["log"])
        logs[spec["name"]] = rows

        if spec["source"] in ("legacy_baseline_test", "test"):
            best_value, best_epoch, best_row = best_metric(rows, "test_acc")
            metric_name = "test_acc"
            best_val = None
            best_test = best_value
        else:
            best_value, best_epoch, best_row = best_metric(rows, "val_acc")
            metric_name = "val_acc"
            best_val = best_value
            best_test = None

        summaries.append(
            {
                "model": spec["name"],
                "params": spec["params"],
                "epochs_recorded": len(rows),
                "best_epoch": best_epoch,
                "best_metric": metric_name,
                "best_val_acc": best_val,
                "best_test_acc": best_test,
                "train_acc_at_best": as_float(best_row, "train_acc") if best_row else None,
                "note": spec["note"],
            }
        )

    return logs, summaries


def write_summary_table(summaries: list[dict], output_path: Path):
    fieldnames = [
        "model",
        "params",
        "epochs_recorded",
        "best_epoch",
        "best_metric",
        "best_val_acc",
        "best_test_acc",
        "train_acc_at_best",
        "note",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    **item,
                    "best_val_acc": format_pct(item["best_val_acc"]),
                    "best_test_acc": format_pct(item["best_test_acc"]),
                    "train_acc_at_best": format_pct(item["train_acc_at_best"]),
                }
            )


def plot_best_accuracy_bar(summaries: list[dict], output_path: Path):
    labels = [
        f"{item['model']}\n(smoke)" if "smoke test" in item["note"] else item["model"]
        for item in summaries
    ]
    values = [
        item["best_val_acc"] if item["best_val_acc"] is not None else item["best_test_acc"]
        for item in summaries
    ]
    metric_labels = [
        "Val" if item["best_val_acc"] is not None else "Test"
        for item in summaries
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, [v * 100 for v in values], color=["#6c757d", "#2f80ed", "#27ae60"])
    plt.ylabel("Accuracy (%)")
    plt.title("Best Available Accuracy by Model")
    plt.ylim(0, max(v * 100 for v in values) + 8)
    plt.grid(axis="y", alpha=0.25)

    for bar, value, metric in zip(bars, values, metric_labels):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value * 100:.2f}% ({metric})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_accuracy_curves(logs: dict, output_path: Path):
    plt.figure(figsize=(9, 5.2))
    for name in ("ResNet20", "VGG16-BN"):
        epochs, values = metric_series(logs[name], "val_acc")
        plt.plot(epochs, [v * 100 for v in values], label=f"{name} Val Acc", linewidth=2)

    # Plot the legacy SimpleCNN result as a reference marker rather than a curve.
    simple_best, simple_epoch, _ = best_metric(logs["SimpleCNN"], "test_acc")
    if simple_best is not None:
        plt.scatter([simple_epoch], [simple_best * 100], color="#6c757d", zorder=4, label="SimpleCNN Best Test")
    else:
        simple_best, simple_epoch, _ = best_metric(logs["SimpleCNN"], "val_acc")
        if simple_best is not None:
            plt.scatter(
                [simple_epoch],
                [simple_best * 100],
                color="#6c757d",
                zorder=4,
                label="SimpleCNN Smoke Val",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_loss_curves(logs: dict, output_path: Path):
    plt.figure(figsize=(9, 5.2))
    for name in ("ResNet20", "VGG16-BN"):
        epochs, values = metric_series(logs[name], "val_loss")
        plt.plot(epochs, values, label=f"{name} Val Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_train_val_gap(logs: dict, output_path: Path):
    plt.figure(figsize=(9, 5.2))
    for name in ("ResNet20", "VGG16-BN"):
        epochs = []
        gaps = []
        for row in logs[name]:
            train_acc = as_float(row, "train_acc")
            val_acc = as_float(row, "val_acc")
            if train_acc is not None and val_acc is not None:
                epochs.append(as_int(row, "epoch"))
                gaps.append((train_acc - val_acc) * 100)
        plt.plot(epochs, gaps, label=f"{name} Train-Val Gap", linewidth=2)

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Gap (percentage points)")
    plt.title("Train vs Validation Accuracy Gap")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_kfold(output_path: Path):
    path = ROOT / "results" / "resnet20" / "kfold_summary.csv"
    if not path.exists():
        return None

    rows = read_log(path)
    folds = [f"Fold {as_int(row, 'fold')}" for row in rows]
    test_accs = [as_float(row, "test_acc") * 100 for row in rows]
    mean = float(np.mean(test_accs))
    std = float(np.std(test_accs))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, test_accs, color="#2f80ed")
    plt.axhline(mean, color="#eb5757", linestyle="--", linewidth=1.5, label=f"Mean {mean:.2f}%")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"ResNet20 5-Fold Test Accuracy (std {std:.2f} pp)")
    plt.ylim(min(test_accs) - 2, max(test_accs) + 2)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    for bar, value in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return mean, std


def write_markdown_summary(summaries: list[dict], kfold_stats, output_path: Path):
    lines = [
        "# Result Comparison Summary",
        "",
        "| Model | Params | Best Epoch | Best Val Acc | Best Test Acc | Note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {model} | {params} | {best_epoch} | {best_val} | {best_test} | {note} |".format(
                model=item["model"],
                params=item["params"],
                best_epoch=item["best_epoch"],
                best_val=format_pct(item["best_val_acc"]) or "-",
                best_test=format_pct(item["best_test_acc"]) or "-",
                note=item["note"],
            )
        )

    if kfold_stats is not None:
        mean, std = kfold_stats
        lines.extend(
            [
                "",
                f"ResNet20 5-fold mean test accuracy: {mean:.2f}% +/- {std:.2f} percentage points.",
            ]
        )

    lines.extend(
        [
            "",
            "Generated files:",
            "- best_accuracy_bar.png",
            "- validation_accuracy_curves.png",
            "- validation_loss_curves.png",
            "- train_val_accuracy_gap.png",
            "- resnet20_kfold_accuracy.png",
            "- summary_table.csv",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs, summaries = collect_model_summaries()

    write_summary_table(summaries, OUTPUT_DIR / "summary_table.csv")
    plot_best_accuracy_bar(summaries, OUTPUT_DIR / "best_accuracy_bar.png")
    plot_accuracy_curves(logs, OUTPUT_DIR / "validation_accuracy_curves.png")
    plot_loss_curves(logs, OUTPUT_DIR / "validation_loss_curves.png")
    plot_train_val_gap(logs, OUTPUT_DIR / "train_val_accuracy_gap.png")
    kfold_stats = plot_kfold(OUTPUT_DIR / "resnet20_kfold_accuracy.png")
    write_markdown_summary(summaries, kfold_stats, OUTPUT_DIR / "analysis_summary.md")

    print(f"Saved comparison outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
