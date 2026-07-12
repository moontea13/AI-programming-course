import hashlib
import json
import logging
import math
import os
import time

import torch
import torch.nn.functional as F

from model import PoetryModel, PoetryModel2, PoetryModel3


MODEL_TYPES = {
    "poetry1": PoetryModel,
    "poetry2": PoetryModel2,
    "poetry3": PoetryModel3,
}

logger = logging.getLogger(__name__)


def build_model(spec, vocab_size):
    return MODEL_TYPES[spec.model_type](vocab_size, spec.embedding_dim, spec.hidden_dim)


def token_cross_entropy(logits, targets, mask):
    flat_mask = mask.reshape(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))[flat_mask]
    flat_targets = targets.reshape(-1)[flat_mask]
    if flat_targets.numel() == 0:
        return logits.sum() * 0, 0
    return F.cross_entropy(flat_logits, flat_targets, reduction="sum"), flat_targets.numel()


def target_mask(targets):
    return targets.ne(0)


def build_optimizer_and_scheduler(model, spec, steps_per_epoch):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=spec.lr,
        weight_decay=spec.weight_decay,
    )
    if spec.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=math.ceil(steps_per_epoch / spec.gradient_accumulation_steps)
            * spec.num_epoch,
        )
    else:
        scheduler = None
    return optimizer, scheduler


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.long().to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            output, _ = model(inputs)
            logits = output.view(inputs.size(0), inputs.size(1), -1)
            loss_sum, token_count = token_cross_entropy(logits, targets, target_mask(targets))
            total_loss += loss_sum.item()
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def _training_loss(logits, targets, mask, label_smoothing):
    flat_mask = mask.reshape(-1)
    active_logits = logits.reshape(-1, logits.size(-1))[flat_mask]
    active_targets = targets.reshape(-1)[flat_mask]
    if active_targets.numel() == 0:
        return logits.sum() * 0, 0
    loss_sum = F.cross_entropy(
        active_logits,
        active_targets,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    return loss_sum, active_targets.numel()


def train_experiment(
    *,
    model,
    train_loader,
    valid_loader,
    spec,
    vocabulary,
    device,
    checkpoint_dir,
    resume,
):
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, spec.experiment_id + ".best.pt")
    last_path = os.path.join(checkpoint_dir, spec.experiment_id + ".last.pt")
    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, spec, len(train_loader))

    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    ss_prob = spec.ss_start_prob
    history = []
    if resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError("Resume checkpoint not found: {}".format(last_path))
        state = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            spec=spec,
            vocabulary=vocabulary,
            map_location=device,
        )
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_valid_loss = state["best_valid_loss"]
        best_epoch = state["best_epoch"]
        patience_counter = state["patience_counter"]
        ss_prob = state["ss_prob"]
        history = list(state["history"])

    run_started = time.perf_counter()
    ss_decay = (
        (spec.ss_start_prob - spec.ss_end_prob) / max(spec.num_epoch - 1, 1)
        if spec.scheduled_sampling and spec.num_epoch > 1
        else 0.0
    )
    for epoch in range(start_epoch, spec.num_epoch + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_tokens = 0
        optimizer.zero_grad()
        for batch_index, batch in enumerate(train_loader):
            batch = batch.long().to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            if spec.scheduled_sampling and ss_prob < 1.0:
                with torch.no_grad():
                    first_output, _ = model(inputs)
                    predictions = first_output.view(
                        inputs.size(0), inputs.size(1), -1
                    ).argmax(dim=-1)
                    replace_mask = torch.rand(inputs.size(), device=device) > ss_prob
                    replace_mask[:, 0] = False
                    noisy_inputs = inputs.clone()
                    noisy_inputs[replace_mask] = predictions[replace_mask]
                output, _ = model(noisy_inputs)
            else:
                output, _ = model(inputs)

            logits = output.view(inputs.size(0), inputs.size(1), -1)
            loss_sum, token_count = _training_loss(
                logits,
                targets,
                target_mask(targets),
                spec.label_smoothing,
            )
            loss = loss_sum / max(token_count, 1)
            group_start = (
                batch_index // spec.gradient_accumulation_steps
            ) * spec.gradient_accumulation_steps
            group_size = min(
                spec.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )
            (loss / group_size).backward()
            should_step = (
                (batch_index + 1) % spec.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                if spec.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss_sum.item()
            total_tokens += token_count
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                logger.info(
                    "%s epoch=%d batch=%d/%d loss=%.4f lr=%.2e ss_prob=%.2f",
                    spec.experiment_id,
                    epoch,
                    batch_index + 1,
                    len(train_loader),
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                    ss_prob,
                )

        train_loss = total_loss / max(total_tokens, 1)
        valid_loss = evaluate_model(model, valid_loader, device)
        improved = valid_loss < best_valid_loss - spec.min_delta
        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "ss_prob": ss_prob,
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        next_ss_prob = max(spec.ss_end_prob, ss_prob - ss_decay)
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "spec": spec,
            "vocabulary": vocabulary,
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "ss_prob": next_ss_prob,
            "history": history,
        }
        if improved:
            save_training_checkpoint(best_path, **checkpoint_kwargs)
        save_training_checkpoint(last_path, **checkpoint_kwargs)
        logger.info(
            "%s epoch=%d train_loss=%.4f valid_loss=%.4f best_epoch=%s",
            spec.experiment_id,
            epoch,
            train_loss,
            valid_loss,
            best_epoch,
        )
        ss_prob = next_ss_prob
        if spec.early_stopping and patience_counter >= spec.patience:
            break

    return {
        "experiment_id": spec.experiment_id,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "global_step": global_step,
        "run_seconds": time.perf_counter() - run_started,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
    }


def vocab_fingerprint(vocabulary):
    payload = json.dumps(
        sorted(vocabulary.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    spec,
    vocabulary,
    epoch,
    global_step,
    best_valid_loss,
    best_epoch,
    patience_counter,
    ss_prob,
    history,
):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "checkpoint_kind": "rnn_gen_training",
        "experiment_id": spec.experiment_id,
        "config": spec.to_dict(),
        "vocab_fingerprint": vocab_fingerprint(vocabulary),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "ss_prob": ss_prob,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    spec,
    vocabulary,
    map_location="cpu",
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_kind") != "rnn_gen_training":
        raise ValueError("Cannot resume training from a weight-only checkpoint")
    if checkpoint["experiment_id"] != spec.experiment_id:
        raise ValueError("Checkpoint experiment does not match requested experiment")
    if checkpoint["config"] != spec.to_dict():
        raise ValueError("Checkpoint configuration does not match requested experiment")
    if checkpoint["vocab_fingerprint"] != vocab_fingerprint(vocabulary):
        raise ValueError("Vocabulary fingerprint mismatch")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return {
        key: checkpoint[key]
        for key in (
            "epoch",
            "global_step",
            "best_valid_loss",
            "best_epoch",
            "patience_counter",
            "ss_prob",
            "history",
        )
    }
