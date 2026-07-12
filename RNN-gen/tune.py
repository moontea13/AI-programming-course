import argparse
import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    _configure_logging,
    _device,
    _device_name,
    _load_data,
)
from experiment_config import get_experiment
from experiment_core import build_model, train_experiment
from utils import init_pretrained_embeddings, set_seed


TUNING_CANDIDATES = (
    {"hidden_dim": 256, "lr": 5e-4, "effective_batch_size": 32},
    {"hidden_dim": 256, "lr": 1e-3, "effective_batch_size": 64},
    {"hidden_dim": 512, "lr": 5e-4, "effective_batch_size": 64},
    {"hidden_dim": 512, "lr": 1e-3, "effective_batch_size": 32},
    {"hidden_dim": 512, "lr": 3e-3, "effective_batch_size": 32},
    {"hidden_dim": 768, "lr": 5e-4, "effective_batch_size": 32},
)


def select_best_trial(trials):
    return min(
        trials,
        key=lambda trial: (
            trial["best_valid_loss"],
            trial["train_seconds"],
            trial["hidden_dim"],
        ),
    )


def _parser():
    parser = argparse.ArgumentParser(description="Tune the shared HANDOFF training recipe")
    parser.add_argument("--trials", type=int, default=6, choices=(6,))
    parser.add_argument("--epochs", type=int, default=5, choices=(5,))
    return parser


def run_tuning(trial_count=6, epochs=5):
    _configure_logging("handoff_tuning")
    base_spec = replace(get_experiment("poetry2_causal"), num_epoch=epochs)
    config, train_data, valid_data, _, word2idx, idx2word = _load_data(base_spec)
    device = _device()
    embedding_weights = None
    if base_spec.use_pretrained_embeddings:
        embedding_weights, embedding_dim = init_pretrained_embeddings(word2idx, idx2word)
        if embedding_dim != base_spec.embedding_dim:
            raise ValueError("Pretrained embedding dimension mismatch")

    tuning_dir = RESULTS_DIR / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_parent = CHECKPOINT_DIR / "tuning"
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, candidate in enumerate(TUNING_CANDIDATES[:trial_count], start=1):
        print("Starting tuning trial {}/{}: {}".format(index, trial_count, candidate), flush=True)
        set_seed(base_spec.seed)
        micro_batch_size = min(32, candidate["effective_batch_size"])
        spec = replace(
            base_spec,
            experiment_id="tune_trial_{:02d}".format(index),
            hidden_dim=candidate["hidden_dim"],
            lr=candidate["lr"],
            batch_size=micro_batch_size,
            effective_batch_size=candidate["effective_batch_size"],
            gradient_accumulation_steps=(
                candidate["effective_batch_size"] // micro_batch_size
            ),
        )
        train_loader = DataLoader(
            torch.from_numpy(train_data),
            batch_size=spec.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        valid_loader = DataLoader(
            torch.from_numpy(valid_data),
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        model = build_model(spec, len(word2idx))
        if embedding_weights is not None:
            model.embeddings = torch.nn.Embedding.from_pretrained(
                embedding_weights.clone(), freeze=False
            )
        with tempfile.TemporaryDirectory(dir=checkpoint_parent) as checkpoint_dir:
            result = train_experiment(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                spec=spec,
                vocabulary=word2idx,
                device=device,
                checkpoint_dir=checkpoint_dir,
                resume=False,
            )
        rows.append(
            {
                "trial": index,
                **candidate,
                "best_epoch": result["best_epoch"],
                "best_valid_loss": result["best_valid_loss"],
                "train_seconds": result["run_seconds"],
                "device": _device_name(device),
            }
        )
        print(
            "Finished trial {}: best_valid_loss={:.6f}, seconds={:.1f}".format(
                index, result["best_valid_loss"], result["run_seconds"]
            ),
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = select_best_trial(rows)
    trials_path = tuning_dir / "trials.csv"
    with trials_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    best_payload = {
        "selection_metric": "best_valid_loss",
        "source_trial": best["trial"],
        "parameters": {
            key: best[key]
            for key in ("hidden_dim", "lr", "effective_batch_size")
        },
        "best_valid_loss": best["best_valid_loss"],
        "train_seconds": best["train_seconds"],
    }
    (tuning_dir / "best_config.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows, best_payload


def main(argv=None):
    args = _parser().parse_args(argv)
    run_tuning(args.trials, args.epochs)


if __name__ == "__main__":
    main()
