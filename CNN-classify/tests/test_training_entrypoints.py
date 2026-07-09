import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import train
import tune


class TinyCIFAR:
    calls = []

    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        TinyCIFAR.calls.append(
            {"root": root, "train": train, "download": download, "transform": transform}
        )

    def __len__(self):
        return 10 if self.train else 4

    def __getitem__(self, index):
        return index, index % 10


class TrainingEntrypointTests(unittest.TestCase):
    def test_build_dataloaders_allows_first_run_dataset_download(self):
        TinyCIFAR.calls = []
        config = {
            "data": {"dir": "dataset"},
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "seed": 42,
                "val_split": 0.0,
            },
        }

        with patch.object(train.datasets, "CIFAR10", TinyCIFAR):
            train.build_dataloaders(config, train_transform="train_tf", eval_transform="eval_tf")

        self.assertEqual([call["download"] for call in TinyCIFAR.calls], [True, True])

    def test_tune_trial_writes_isolated_method_to_temp_config_before_training(self):
        captured = {}
        base_config = {
            "method": "resnet20",
            "model": {"name": "ResNet20", "params": {"num_classes": 10}},
            "training": {"epochs": 150, "batch_size": 128},
            "optimizer": {"name": "sgd", "lr": 0.1, "weight_decay": 0.0001},
        }

        def fake_run(cmd, capture_output, text):
            config_path = cmd[3]
            with open(config_path, "r", encoding="utf-8") as f:
                captured["config"] = yaml.safe_load(f)
            return types.SimpleNamespace(
                returncode=0,
                stdout="Best Val Acc: 0.9000 at epoch 3\n",
                stderr="",
            )

        with patch.object(tune.subprocess, "run", fake_run):
            tune.run_single_trial(
                copy.deepcopy(base_config),
                method="resnet20",
                label="lr=0.1_weight_decay=0.0001",
                tune_epochs=30,
                k_fold=None,
            )

        self.assertEqual(
            captured["config"]["method"],
            "resnet20/tune_lr=0.1_weight_decay=0.0001",
        )

    def test_single_train_writes_early_stop_epoch_to_training_log(self):
        config = {
            "method": "tiny",
            "model": {"name": "TinyModel", "params": {}},
            "training": {
                "epochs": 3,
                "batch_size": 2,
                "num_workers": 0,
                "seed": 42,
                "val_split": 0.1,
            },
            "optimizer": {"name": "sgd", "lr": 0.1},
            "scheduler": {"name": "none"},
            "early_stopping": {"patience": 1, "min_delta": 0.001},
            "data": {"dir": "dataset"},
        }
        val_results = [(1.0, 0.5, None), (1.1, 0.4, None)]

        def fake_evaluate(model, loader, criterion, device, collect_confusion=False):
            if collect_confusion:
                return 1.2, 0.3, [[1]]
            return val_results.pop(0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "checkpoints").mkdir()
            (tmp_path / "results").mkdir()
            with (
                patch.object(train, "build_dataloaders", return_value=(object(), object(), object())),
                patch.object(train, "build_model", return_value=train.nn.Linear(1, 1)),
                patch.object(train, "train_one_epoch", return_value=(0.9, 0.6)),
                patch.object(train, "evaluate", fake_evaluate),
                patch.object(train, "plot_curves"),
                patch.object(train, "plot_confusion_matrix"),
            ):
                train.single_train(
                    config,
                    ckpt_dir=tmp_path / "checkpoints",
                    results_dir=tmp_path / "results",
                    device=train.torch.device("cpu"),
                )

            self.assertIn("2,0.9,0.6,1.1,0.4,,", (tmp_path / "results" / "training_log.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
