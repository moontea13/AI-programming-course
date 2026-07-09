#!/usr/bin/env python3
"""Hyperparameter tuning with cross-validation for CIFAR-10 CNN methods.

Usage:
    # Quick tuning with validation set (no k-fold)
    python tune.py --method simple_cnn --tune-epochs 20 --trials 8

    # Tuning with 5-fold cross-validation
    python tune.py --method resnet20 --tune-epochs 30 --k-fold 5 --trials 8

    # Full grid search with k-fold on all methods
    python tune.py --method all --tune-epochs 30 --k-fold 5
"""

import argparse
import copy
import csv
import itertools
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Search grids per method
# ---------------------------------------------------------------------------
SEARCH_GRIDS = {
    "simple_cnn": {
        "optimizer.lr": [0.01, 0.001, 0.0005],
        "optimizer.name": ["adam", "sgd"],
        "model.params.dropout": [0.3, 0.5, 0.7],
        "training.batch_size": [64, 128],
        "optimizer.momentum": ["__dynamic__"],
        "optimizer.weight_decay": ["__dynamic__"],
    },
    "resnet20": {
        "optimizer.lr": [0.2, 0.1, 0.05, 0.01],
        "optimizer.name": ["sgd"],
        "optimizer.weight_decay": [1e-4, 5e-4, 1e-3],
        "scheduler.gamma": [0.1, 0.2],
    },
    "vgg16_bn": {
        "optimizer.lr": [0.05, 0.01, 0.005],
        "optimizer.name": ["sgd"],
        "optimizer.weight_decay": [5e-4, 1e-3],
        "model.params.dropout": [0.3, 0.5],
        "scheduler.gamma": [0.1, 0.2],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for CIFAR-10 CNNs.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["simple_cnn", "resnet20", "vgg16_bn", "all"],
                        help="Which method to tune, or 'all'.")
    parser.add_argument("--tune-epochs", type=int, default=30,
                        help="Epochs per trial (fewer than full training).")
    parser.add_argument("--trials", type=int, default=None,
                        help="Max random trials (omit for full grid search).")
    parser.add_argument("--k-fold", type=int, default=None,
                        help="k-fold cross-validation (default: use single val split).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def load_base_config(method: str) -> dict:
    config_path = Path("configs") / f"{method}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_nested(d: dict, key_path: str, value):
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_grid(grid: dict) -> list[dict]:
    """Expand a param grid, resolving __dynamic__ keys after expansion."""
    static = {k: v for k, v in grid.items() if v != ["__dynamic__"]}
    dynamic_keys = [k for k, v in grid.items() if v == ["__dynamic__"]]

    combos = []
    keys = list(static.keys())
    values = list(static.values())
    for combo_values in itertools.product(*values):
        combo = dict(zip(keys, combo_values))
        opt = combo.get("optimizer.name", "adam")
        if "optimizer.momentum" in dynamic_keys:
            combo["optimizer.momentum"] = 0.9 if opt == "sgd" else None
        if "optimizer.weight_decay" in dynamic_keys and "optimizer.weight_decay" not in combo:
            combo["optimizer.weight_decay"] = 1e-4 if opt == "sgd" else None
        combos.append(combo)
    return combos


def apply_combo(base_config: dict, combo: dict) -> dict:
    config = copy.deepcopy(base_config)
    for key_path, value in combo.items():
        if value is not None:
            set_nested(config, key_path, value)
    return config


def config_to_label(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        short_k = k.split(".")[-1]
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{short_k}={v:.0e}" if v < 0.01 else f"{short_k}={v}")
        else:
            parts.append(f"{short_k}={v}")
    return "_".join(parts)


def parse_best_val_acc(stdout: str):
    """Parse 'Best Val Acc: X.XXXX at epoch Y' from training output."""
    for line in stdout.splitlines():
        if "Best Val Acc:" in line:
            parts = line.strip().split()
            try:
                return float(parts[3]), int(parts[6])
            except (ValueError, IndexError):
                pass
    # Fallback: try test acc
    for line in stdout.splitlines():
        if "Test Acc (best val checkpoint):" in line:
            parts = line.strip().split()
            try:
                return float(parts[-1]), None
            except ValueError:
                pass
    return None, None


def run_single_trial(config: dict, method: str, label: str, tune_epochs: int,
                     k_fold: int or None) -> dict:
    """Run one hyperparameter trial. Returns dict with val_acc, test_acc, etc."""
    t0 = time.time()

    if k_fold:
        return run_kfold_trial(config, method, label, tune_epochs, k_fold, t0)

    # Single validation split
    config["method"] = f"{method}/tune_{label}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        tmp_path = f.name

    cmd = [sys.executable, "train.py", "--config", tmp_path, "--epochs", str(tune_epochs)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - t0
    success = result.returncode == 0
    val_acc, best_epoch = parse_best_val_acc(result.stdout) if success else (None, None)

    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

    if not success:
        print(f"    FAILED (exit {result.returncode})", flush=True)
        stderr_lines = result.stderr.strip().splitlines()
        for l in stderr_lines[-3:]:
            print(f"    stderr: {l}", flush=True)

    return {
        "label": label,
        "method": method,
        "val_acc": val_acc,
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 1),
        "success": success,
    }


def run_kfold_trial(config: dict, method: str, label: str, tune_epochs: int,
                    k: int, t0: float) -> dict:
    """Run one trial with k-fold CV. Averages val_acc across folds."""
    print(f"    Running {k}-fold CV...", flush=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        tmp_path = f.name

    config["method"] = f"{method}/tune_{label}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    cmd = [
        sys.executable, "train.py", "--config", tmp_path,
        "--epochs", str(tune_epochs), "--k-fold", str(k),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - t0
    success = result.returncode == 0

    val_acc = None
    if success:
        # Parse k-fold summary: "Val Acc: X.XXXX +/- Y.YYYY"
        for line in result.stdout.splitlines():
            if "Val Acc:" in line and "+/-" in line:
                parts = line.strip().split()
                try:
                    val_acc = float(parts[2])
                except ValueError:
                    pass
                break

    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

    if not success:
        print(f"    FAILED (exit {result.returncode})", flush=True)
        stderr_lines = result.stderr.strip().splitlines()
        for l in stderr_lines[-3:]:
            print(f"    stderr: {l}", flush=True)

    return {
        "label": label,
        "method": method,
        "val_acc": val_acc,
        "best_epoch": None,
        "elapsed_s": round(elapsed, 1),
        "success": success,
        "k_fold": k,
    }


def main():
    args = parse_args()

    methods = (
        list(SEARCH_GRIDS.keys()) if args.method == "all"
        else [args.method]
    )

    all_results = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Tuning: {method}")
        cv_mode = f"{args.k_fold}-fold CV" if args.k_fold else "val split"
        print(f"  Mode: {cv_mode}, epochs/trial: {args.tune_epochs}")
        print(f"{'='*60}", flush=True)

        base_config = load_base_config(method)
        grid = SEARCH_GRIDS[method]
        combos = expand_grid(grid)

        if args.trials is not None and args.trials < len(combos):
            import random
            random.seed(args.seed)
            combos = random.sample(combos, args.trials)

        print(f"  Total trials: {len(combos)}", flush=True)

        for i, combo in enumerate(combos):
            config = apply_combo(base_config, combo)
            label = config_to_label(combo)

            print(f"\n  [{i+1}/{len(combos)}] {label}", flush=True)

            result = run_single_trial(config, method, label, args.tune_epochs, args.k_fold)
            all_results.append(result)

            if result["success"]:
                print(f"    -> Val Acc: {result['val_acc']:.4f}, "
                      f"Time: {result['elapsed_s']:.0f}s", flush=True)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Tuning Summary")
    print(f"{'='*60}")

    success_results = [r for r in all_results if r["success"] and r["val_acc"] is not None]
    if success_results:
        success_results.sort(key=lambda r: r["val_acc"], reverse=True)

        print(f"{'Method':<15} {'Label':<55} {'Val Acc':>10} {'Epoch':>8} {'Time(s)':>8}")
        print("-" * 100)
        for r in success_results:
            ep = r["best_epoch"] or 0
            print(f"{r['method']:<15} {r['label']:<55} {r['val_acc']:>10.4f} {ep:>8} {r['elapsed_s']:>8.0f}")

        # Best per method
        print(f"\nBest per method:")
        for method in methods:
            method_results = [r for r in success_results if r["method"] == method]
            if method_results:
                best = max(method_results, key=lambda r: r["val_acc"])
                print(f"  {method}: {best['label']}")
                print(f"    Val Acc={best['val_acc']:.4f}, Time={best['elapsed_s']:.0f}s")

        # Save
        out_path = "results/tuning_summary.csv"
        fieldnames = ["method", "label", "val_acc", "best_epoch", "elapsed_s", "success"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nFull results saved to: {out_path}")
    else:
        print("No successful trials.")


if __name__ == "__main__":
    main()
