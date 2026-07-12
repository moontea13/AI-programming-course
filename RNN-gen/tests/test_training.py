from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from experiment_config import get_experiment
from experiment_core import (
    build_model,
    build_optimizer_and_scheduler,
    save_training_checkpoint,
    train_experiment,
)


def _tiny_spec(num_epoch):
    return replace(
        get_experiment("poetry2_legacy_recipe"),
        embedding_dim=4,
        hidden_dim=4,
        batch_size=2,
        num_epoch=num_epoch,
        lr=1e-2,
    )


def _tiny_loaders():
    sequences = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [1, 3, 2, 4, 0],
            [1, 2, 2, 4, 0],
            [1, 3, 3, 4, 0],
        ]
    )
    loader = DataLoader(sequences, batch_size=2, shuffle=False)
    return loader, loader


def test_train_experiment_writes_best_and_last_checkpoints(tmp_path):
    spec = _tiny_spec(num_epoch=2)
    model = build_model(spec, vocab_size=5)
    train_loader, valid_loader = _tiny_loaders()

    result = train_experiment(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        spec=spec,
        vocabulary={"PAD": 0, "SOP": 1, "春": 2, "月": 3, "EOP": 4},
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume=False,
    )

    assert len(result["history"]) == 2
    assert result["best_epoch"] in {1, 2}
    assert result["best_valid_loss"] > 0
    assert all(row["ss_prob"] == 1.0 for row in result["history"])
    assert (tmp_path / (spec.experiment_id + ".best.pt")).exists()
    assert (tmp_path / (spec.experiment_id + ".last.pt")).exists()
    assert {"epoch", "train_loss", "valid_loss", "learning_rate", "ss_prob"} <= set(
        result["history"][0]
    )


def test_train_experiment_resumes_after_last_completed_epoch(tmp_path):
    spec = _tiny_spec(num_epoch=2)
    model = build_model(spec, vocab_size=5)
    train_loader, valid_loader = _tiny_loaders()
    vocabulary = {"PAD": 0, "SOP": 1, "春": 2, "月": 3, "EOP": 4}
    optimizer, scheduler = build_optimizer_and_scheduler(model, spec, len(train_loader))
    save_training_checkpoint(
        tmp_path / (spec.experiment_id + ".last.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        spec=spec,
        vocabulary=vocabulary,
        epoch=1,
        global_step=2,
        best_valid_loss=1.5,
        best_epoch=1,
        patience_counter=0,
        ss_prob=1.0,
        history=[{"epoch": 1, "train_loss": 2.0, "valid_loss": 1.5}],
    )

    resumed_model = build_model(spec, vocab_size=5)
    result = train_experiment(
        model=resumed_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        spec=spec,
        vocabulary=vocabulary,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume=True,
    )

    assert [row["epoch"] for row in result["history"]] == [1, 2]
    assert result["global_step"] == 4


def test_gradient_accumulation_counts_optimizer_steps(tmp_path):
    spec = replace(
        _tiny_spec(num_epoch=1),
        batch_size=1,
        effective_batch_size=2,
        gradient_accumulation_steps=2,
    )
    model = build_model(spec, vocab_size=5)
    sequences = torch.tensor(
        [[1, 2, 3, 4, 0], [1, 3, 2, 4, 0], [1, 2, 2, 4, 0], [1, 3, 3, 4, 0]]
    )
    loader = DataLoader(sequences, batch_size=1, shuffle=False)

    result = train_experiment(
        model=model,
        train_loader=loader,
        valid_loader=loader,
        spec=spec,
        vocabulary={"PAD": 0, "SOP": 1, "春": 2, "月": 3, "EOP": 4},
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume=False,
    )

    assert result["global_step"] == 2


def test_scheduled_sampling_reaches_configured_end_probability(tmp_path):
    spec = replace(
        get_experiment("poetry2_controlled"),
        experiment_id="tiny_scheduled_sampling",
        embedding_dim=4,
        hidden_dim=4,
        batch_size=2,
        effective_batch_size=2,
        gradient_accumulation_steps=1,
        num_epoch=2,
        use_pretrained_embeddings=False,
    )
    model = build_model(spec, vocab_size=5)
    train_loader, valid_loader = _tiny_loaders()

    result = train_experiment(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        spec=spec,
        vocabulary={"PAD": 0, "SOP": 1, "春": 2, "月": 3, "EOP": 4},
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume=False,
    )

    assert [row["ss_prob"] for row in result["history"]] == [1.0, 0.2]
