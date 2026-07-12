from dataclasses import replace

import pytest
import torch

from experiment_config import get_experiment
from experiment_core import (
    build_model,
    load_training_checkpoint,
    save_training_checkpoint,
    vocab_fingerprint,
)


def _tiny_training_objects():
    spec = replace(
        get_experiment("poetry2_legacy_recipe"),
        embedding_dim=4,
        hidden_dim=4,
    )
    model = build_model(spec, vocab_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    input_ids = torch.tensor([[1, 2, 3]])
    output, _ = model(input_ids)
    output.sum().backward()
    optimizer.step()
    scheduler.step()
    return spec, model, optimizer, scheduler


def test_full_checkpoint_round_trip_restores_training_state(tmp_path):
    spec, model, optimizer, scheduler = _tiny_training_objects()
    checkpoint_path = tmp_path / "model.last.pt"
    vocabulary = {"PAD": 0, "SOP": 1, "春": 2}
    history = [{"epoch": 1, "train_loss": 2.0, "valid_loss": 1.5}]

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        spec=spec,
        vocabulary=vocabulary,
        epoch=1,
        global_step=10,
        best_valid_loss=1.5,
        best_epoch=1,
        patience_counter=0,
        ss_prob=0.8,
        history=history,
    )

    restored_model = build_model(spec, vocab_size=8)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=4)
    state = load_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        spec=spec,
        vocabulary=vocabulary,
    )

    assert state["epoch"] == 1
    assert state["global_step"] == 10
    assert state["best_valid_loss"] == 1.5
    assert state["best_epoch"] == 1
    assert state["ss_prob"] == 0.8
    assert state["history"] == history
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_vocabulary_mismatch(tmp_path):
    spec, model, optimizer, scheduler = _tiny_training_objects()
    checkpoint_path = tmp_path / "model.last.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        spec=spec,
        vocabulary={"PAD": 0, "春": 1},
        epoch=1,
        global_step=1,
        best_valid_loss=1.0,
        best_epoch=1,
        patience_counter=0,
        ss_prob=1.0,
        history=[],
    )

    with pytest.raises(ValueError, match="Vocabulary fingerprint mismatch"):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            spec=spec,
            vocabulary={"PAD": 0, "月": 1},
        )


def test_weight_only_checkpoint_cannot_resume_training(tmp_path):
    spec, model, optimizer, scheduler = _tiny_training_objects()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="weight-only checkpoint"):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            spec=spec,
            vocabulary={"PAD": 0},
        )


def test_vocabulary_fingerprint_is_order_independent():
    first = {"PAD": 0, "春": 1, "月": 2}
    second = {"月": 2, "PAD": 0, "春": 1}

    assert vocab_fingerprint(first) == vocab_fingerprint(second)
