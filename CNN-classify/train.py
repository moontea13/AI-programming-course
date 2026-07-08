import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from utils import ensure_dirs, plot_confusion_matrix, plot_curves, set_seed, write_training_log


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_DOWNLOAD_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN baseline on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for CIFAR-10 data.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for logs and figures.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 for Windows stability.")
    return parser.parse_args()


def build_dataloaders(args):
    datasets.CIFAR10.url = CIFAR10_DOWNLOAD_URL

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


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


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.results_dir, args.checkpoints_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, test_loader = build_dataloaders(args)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_rows = []
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = Path(args.checkpoints_dir) / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
        }
        log_rows.append(row)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}",
            flush=True,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_acc": best_test_acc,
                    "args": vars(args),
                },
                best_model_path,
            )

        write_training_log(log_rows, str(Path(args.results_dir) / "training_log.csv"))
        plot_curves(log_rows, args.results_dir)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion_matrix = evaluate(model, test_loader, criterion, device, collect_confusion=True)
    plot_confusion_matrix(confusion_matrix, str(Path(args.results_dir) / "confusion_matrix.png"))

    print(f"Best Test Acc: {best_test_acc:.4f} at epoch {best_epoch}", flush=True)
    print(f"Final evaluated best checkpoint Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}", flush=True)
    print(f"Saved log and figures to: {Path(args.results_dir).resolve()}", flush=True)
    print(f"Saved best model to: {best_model_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
