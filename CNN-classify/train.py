import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from model import build_model
from utils import ensure_dirs, plot_confusion_matrix, plot_curves, set_seed, write_training_log


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_DOWNLOAD_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--k-fold", type=int, default=None, help="k-fold cross-validation (overrides val_split).")
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key in ["epochs", "batch_size", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            config["training"][key] = val

    if args.lr is not None:
        config.setdefault("optimizer", {})["lr"] = args.lr

    return config


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


def build_dataloaders(config, train_transform, eval_transform):
    data_dir = config["data"]["dir"]
    training = config["training"]

    datasets.CIFAR10.url = CIFAR10_DOWNLOAD_URL

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                    transform=eval_transform)

    val_split = training.get("val_split", 0.0)
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        gen = torch.Generator().manual_seed(training["seed"])
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                  generator=gen)
        # Validation uses eval transform; override the transform
        val_dataset = _override_transform(val_dataset, eval_transform, train_dataset)
    else:
        val_dataset = None

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=training["batch_size"], shuffle=True,
                              num_workers=training["num_workers"], pin_memory=pin_memory)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=training["batch_size"], shuffle=False,
                                num_workers=training["num_workers"], pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=training["batch_size"], shuffle=False,
                             num_workers=training["num_workers"], pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _override_transform(subset, new_transform, fallback_dataset):
    """Replace the transform applied to a Subset by wrapping it."""
    indices = subset.indices
    base_dataset = fallback_dataset if hasattr(fallback_dataset, 'dataset') is False else None
    # For random_split output, we can't easily change transform of the underlying dataset.
    # Instead, create a new Subset from a copy of the dataset with eval transform.
    return Subset(
        datasets.CIFAR10(root=subset.dataset.root, train=True, download=True,
                         transform=new_transform),
        indices,
    )


def build_optimizer(model, cfg: dict) -> optim.Optimizer:
    name = cfg.get("name", "adam").lower()
    lr = cfg["lr"]
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr,
                         momentum=cfg.get("momentum", 0.9),
                         weight_decay=cfg.get("weight_decay", 1e-4))
    else:
        return optim.Adam(model.parameters(), lr=lr)


def build_scheduler(optimizer, cfg: dict):
    name = cfg.get("name", "none").lower()
    if name == "step":
        return StepLR(optimizer, step_size=cfg.get("step_size", 40),
                      gamma=cfg.get("gamma", 0.1))
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_confusion=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion_matrix = np.zeros((10, 10), dtype=np.int64) if collect_confusion else None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

        if collect_confusion:
            for true_label, predicted_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_label, predicted_label] += 1

    return running_loss / total, correct / total, confusion_matrix


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Returns True if this is a new best (model should be saved)."""
        if self.best_score is None:
            self.best_score = metric
            return True

        if self.mode == "max":
            improved = metric > self.best_score + self.min_delta
        else:
            improved = metric < self.best_score - self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def single_train(config, ckpt_dir, results_dir, device):
    """Run one complete training session. Returns (best_val_acc, best_epoch)."""
    training = config["training"]
    es_cfg = config.get("early_stopping", {})

    set_seed(training["seed"])

    train_transform, eval_transform = build_transforms()
    train_loader, val_loader, test_loader = build_dataloaders(config, train_transform, eval_transform)

    model_cfg = config["model"]
    model = build_model(model_cfg["name"], **model_cfg.get("params", {})).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cfg['name']}, Params: {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.get("optimizer", {"name": "adam", "lr": 0.001}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}))

    has_val = val_loader is not None
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",
    ) if has_val else None

    log_rows = []
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, training["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss = val_acc = None
        if has_val:
            val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else "",
            "val_acc": round(val_acc, 6) if val_acc is not None else "",
        }
        log_rows.append(row)

        # Log line
        parts = [f"Epoch [{epoch}/{training['epochs']}]", f"LR: {current_lr:.6f}"]
        parts.append(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if has_val:
            parts.append(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(" | ".join(parts), flush=True)

        # Checkpoint based on val accuracy (or train acc if no val)
        monitor = val_acc if has_val else train_acc
        if monitor is not None and monitor > best_val_acc:
            best_val_acc = monitor
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                best_model_path,
            )

        should_stop = False
        if early_stopping is not None:
            improved = early_stopping.step(val_acc)
            if improved:
                print(f"  -> New best val acc: {val_acc:.4f} (epoch {epoch})", flush=True)
            if early_stopping.should_stop:
                print(f"  Early stopping triggered at epoch {epoch} "
                      f"(patience={es_cfg.get('patience', 15)}, best_val={early_stopping.best_score:.4f})", flush=True)
                should_stop = True

        write_training_log(log_rows, str(results_dir / "training_log.csv"))

        if should_stop:
            break

    # Plot curves
    plot_curves(log_rows, str(results_dir), config["method"])

    # Load best and evaluate on test set
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion_matrix = evaluate(model, test_loader, criterion, device, collect_confusion=True)
    plot_confusion_matrix(confusion_matrix, str(results_dir / "confusion_matrix.png"))

    print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}", flush=True)
    print(f"Test Acc (best val checkpoint): {test_acc:.4f}", flush=True)
    print(f"Results saved to: {results_dir.resolve()}", flush=True)

    return best_val_acc, best_epoch, test_acc


def run_kfold(config, args, ckpt_dir, results_dir, device):
    """k-fold cross-validation on the training set.

    Splits training data into k folds, trains k times, each time using 1 fold
    as validation and the rest as training. Averages results across folds.
    """
    k = args.k_fold
    training = config["training"]
    data_dir = config["data"]["dir"]

    datasets.CIFAR10.url = CIFAR10_DOWNLOAD_URL
    _, eval_transform = build_transforms()

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                  transform=eval_transform)
    n = len(full_train)
    indices = np.random.RandomState(training["seed"]).permutation(n)
    fold_size = n // k

    fold_results = []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{k}  (train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*50}", flush=True)

        # Create fold-specific config
        fold_config = config.copy()
        fold_config["method"] = f"{config['method']}/fold_{fold}"
        fold_config["training"] = training.copy()
        fold_config["training"]["seed"] = training["seed"] + fold

        fold_ckpt = ckpt_dir / f"fold_{fold}"
        fold_res = results_dir / f"fold_{fold}"
        ensure_dirs(str(fold_ckpt), str(fold_res))

        # Override dataloader building for k-fold
        val_acc, best_epoch, test_acc = _train_fold(
            fold_config, fold_ckpt, fold_res, device,
            train_idx, val_idx, full_train,
        )
        fold_results.append({"fold": fold, "val_acc": val_acc, "best_epoch": best_epoch,
                             "test_acc": test_acc})

    # Summary
    val_accs = [r["val_acc"] for r in fold_results if r["val_acc"] is not None]
    test_accs = [r["test_acc"] for r in fold_results if r["test_acc"] is not None]
    print(f"\n{'='*50}")
    print(f"k-Fold CV Summary (k={k})")
    print(f"  Val Acc: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")
    print(f"  Test Acc: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val={r['val_acc']:.4f}, Test={r['test_acc']:.4f}")

    return np.mean(val_accs), np.mean(test_accs)


def _train_fold(config, ckpt_dir, results_dir, device,
                train_idx, val_idx, full_dataset):
    """Single fold training with pre-split indices."""
    training = config["training"]
    es_cfg = config.get("early_stopping", {})

    set_seed(training["seed"])
    _, eval_transform = build_transforms()

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Override transforms for training set
    train_transform, _ = build_transforms()
    train_dataset = datasets.CIFAR10(root=full_dataset.root, train=True, download=True,
                                     transform=train_transform)
    train_subset = Subset(train_dataset, train_idx)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=training["batch_size"], shuffle=True,
                              num_workers=training["num_workers"], pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=training["batch_size"], shuffle=False,
                            num_workers=training["num_workers"], pin_memory=pin_memory)
    # Test loader
    test_dataset = datasets.CIFAR10(root=full_dataset.root, train=False, download=True,
                                    transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=training["batch_size"], shuffle=False,
                             num_workers=training["num_workers"], pin_memory=pin_memory)

    model_cfg = config["model"]
    model = build_model(model_cfg["name"], **model_cfg.get("params", {})).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.get("optimizer", {"name": "adam", "lr": 0.001}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}))

    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, training["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch}/{training['epochs']}] LR: {current_lr:.6f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "best_val_acc": best_val_acc, "config": config},
                best_model_path,
            )

        improved = early_stopping.step(val_acc)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch} (best val={early_stopping.best_score:.4f})", flush=True)
            break

    # Load best and evaluate on test
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)

    print(f"  Fold result: Best Val={best_val_acc:.4f} @ epoch {best_epoch}, "
          f"Test={test_acc:.4f}", flush=True)

    return best_val_acc, best_epoch, test_acc


def main():
    args = parse_args()
    config = load_config(args.config, args)

    method = config["method"]
    ckpt_dir = Path("checkpoints") / method
    results_dir = Path("results") / method

    ensure_dirs(str(ckpt_dir), str(results_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Method: {method}", flush=True)
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    if args.k_fold is not None and args.k_fold > 1:
        run_kfold(config, args, ckpt_dir, results_dir, device)
    else:
        single_train(config, ckpt_dir, results_dir, device)


if __name__ == "__main__":
    main()
