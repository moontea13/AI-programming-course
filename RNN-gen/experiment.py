import argparse
import csv
import json
import logging
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from ai_evaluation import write_ai_simulated_evaluation
from evaluation import PoetryGenerator, calculate_generation_metrics, generate_fixed_samples
from experiment_config import (
    CORE_QUICK_SUITE,
    EXPERIMENTS,
    HANDOFF_FULL_SUITE,
    get_experiment,
)
from experiment_core import build_model, evaluate_model, train_experiment
from process import get_split_data
from reporting import (
    build_summary_report,
    write_experiment_artifacts,
    write_loss_plot,
    write_metric_plot,
)
from utils import init_pretrained_embeddings, set_seed


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
TUNING_BEST_CONFIG = RESULTS_DIR / "tuning" / "best_config.json"


def _suite_experiments(suite):
    if suite == "handoff_full":
        return HANDOFF_FULL_SUITE
    return CORE_QUICK_SUITE


def _resolved_spec(experiment_id):
    spec = get_experiment(experiment_id)
    if experiment_id in HANDOFF_FULL_SUITE and TUNING_BEST_CONFIG.exists():
        payload = json.loads(TUNING_BEST_CONFIG.read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        spec = replace(
            spec,
            hidden_dim=int(parameters["hidden_dim"]),
            lr=float(parameters["lr"]),
            effective_batch_size=int(parameters["effective_batch_size"]),
        )
    micro_batch_size = min(spec.batch_size, spec.effective_batch_size)
    if spec.model_type == "poetry3" and spec.hidden_dim >= 768:
        micro_batch_size = min(micro_batch_size, 16)
    return replace(
        spec,
        batch_size=micro_batch_size,
        gradient_accumulation_steps=spec.effective_batch_size // micro_batch_size,
    )


def _reduced_micro_batch(spec):
    if spec.batch_size <= 1 or spec.effective_batch_size % (spec.batch_size // 2) != 0:
        raise RuntimeError("Cannot reduce micro batch while preserving effective batch size")
    micro_batch_size = spec.batch_size // 2
    return replace(
        spec,
        batch_size=micro_batch_size,
        gradient_accumulation_steps=spec.effective_batch_size // micro_batch_size,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run reproducible RNN poetry experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    run_parser.add_argument("--resume", action="store_true")

    suite_parser = subparsers.add_parser("run-suite")
    suite_parser.add_argument(
        "--suite", default="core_quick", choices=("core_quick", "handoff_full")
    )
    suite_parser.add_argument("--resume", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--suite", default="core_quick", choices=("core_quick", "handoff_full")
    )

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument(
        "--suite", default="core_quick", choices=("core_quick", "handoff_full")
    )
    ai_parser = subparsers.add_parser("ai-score")
    ai_parser.add_argument("--suite", default="handoff_full", choices=("handoff_full",))
    return parser


def build_runtime_config(spec):
    config = Config()
    for name in (
        "split_strategy",
        "use_author_weighted_sampling",
        "num_epoch",
        "batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "lr",
        "weight_decay",
        "max_len",
        "max_gen_len",
        "embedding_dim",
        "hidden_dim",
        "temperature",
        "top_p",
        "scheduled_sampling",
        "ss_start_prob",
        "ss_end_prob",
        "grad_clip",
        "label_smoothing",
        "lr_scheduler",
        "early_stopping",
        "patience",
        "min_delta",
    ):
        setattr(config, name, getattr(spec, name))
    config.do_train = True
    config.do_test = True
    config.do_predict = False
    config.do_load_model = False
    return config


def build_payload(
    *,
    spec,
    training_result,
    test_loss,
    generated,
    generation_metrics,
    device_name,
):
    return {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "spec": spec.to_dict(),
        "runtime": {
            "device": device_name,
            "train_seconds": training_result["run_seconds"],
        },
        "training": {
            "history": training_result["history"],
            "best_epoch": training_result["best_epoch"],
            "best_valid_loss": training_result["best_valid_loss"],
        },
        "test": {
            "loss": test_loss,
            "perplexity": math.exp(test_loss),
        },
        "generation_metrics": generation_metrics,
        "generated": generated,
        "checkpoints": {
            "best": str(training_result["best_checkpoint"]),
            "last": str(training_result["last_checkpoint"]),
        },
    }


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def _configure_logging(experiment_id):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        RESULTS_DIR / (experiment_id + ".log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(stream_handler)


def _load_data(spec):
    config = build_runtime_config(spec)
    train_data, valid_data, test_data, word2idx, idx2word = get_split_data(config)
    config.word2idx = word2idx
    config.idx2word = idx2word
    return config, train_data, valid_data, test_data, word2idx, idx2word


def _data_loader(data, spec, shuffle):
    return DataLoader(
        torch.from_numpy(data),
        batch_size=spec.batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def _load_best_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _evaluate_and_generate(model, spec, test_loader, word2idx, idx2word, device):
    test_loss = evaluate_model(model, test_loader, device)
    generator = PoetryGenerator(
        model=model,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device,
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_gen_len=spec.max_gen_len,
    )
    generated = generate_fixed_samples(generator)
    return test_loss, generated, calculate_generation_metrics(generated)


def _run_experiment_attempt(spec, resume):
    _configure_logging(spec.experiment_id)
    set_seed(spec.seed)
    device = _device()
    _, train_data, valid_data, test_data, word2idx, idx2word = _load_data(spec)
    train_loader = _data_loader(train_data, spec, shuffle=True)
    valid_loader = _data_loader(valid_data, spec, shuffle=False)
    test_loader = _data_loader(test_data, spec, shuffle=False)
    model = build_model(spec, len(word2idx))
    if spec.use_pretrained_embeddings and not resume:
        weights, embedding_dim = init_pretrained_embeddings(word2idx, idx2word)
        if embedding_dim != spec.embedding_dim:
            raise ValueError(
                "Pretrained embedding dimension {} does not match {}".format(
                    embedding_dim, spec.embedding_dim
                )
            )
        model.embeddings = torch.nn.Embedding.from_pretrained(weights, freeze=False)

    training_result = train_experiment(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        spec=spec,
        vocabulary=word2idx,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=resume,
    )
    _load_best_weights(model, training_result["best_checkpoint"], device)
    test_loss, generated, metrics = _evaluate_and_generate(
        model, spec, test_loader, word2idx, idx2word, device
    )
    payload = build_payload(
        spec=spec,
        training_result=training_result,
        test_loss=test_loss,
        generated=generated,
        generation_metrics=metrics,
        device_name=_device_name(device),
    )
    write_experiment_artifacts(RESULTS_DIR, payload)
    return payload


def run_experiment(experiment_id, resume=False):
    spec = _resolved_spec(experiment_id)
    while True:
        try:
            return _run_experiment_attempt(spec, resume)
        except RuntimeError as exc:
            is_cuda_oom = "out of memory" in str(exc).lower()
            if not is_cuda_oom or resume or spec.batch_size <= 1:
                raise
            logger = logging.getLogger(__name__)
            reduced = _reduced_micro_batch(spec)
            logger.warning(
                "%s CUDA OOM at micro batch %d; retrying with micro batch %d and accumulation %d",
                experiment_id,
                spec.batch_size,
                reduced.batch_size,
                reduced.gradient_accumulation_steps,
            )
            spec = reduced
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def evaluate_experiment(experiment_id):
    spec = _resolved_spec(experiment_id)
    _configure_logging(experiment_id + "_evaluate")
    set_seed(spec.seed)
    device = _device()
    _, _, _, test_data, word2idx, idx2word = _load_data(spec)
    test_loader = _data_loader(test_data, spec, shuffle=False)
    model = build_model(spec, len(word2idx)).to(device)
    best_path = CHECKPOINT_DIR / (experiment_id + ".best.pt")
    checkpoint = _load_best_weights(model, best_path, device)
    test_loss, generated, metrics = _evaluate_and_generate(
        model, spec, test_loader, word2idx, idx2word, device
    )
    training_result = {
        "history": checkpoint["history"],
        "best_epoch": checkpoint["best_epoch"],
        "best_valid_loss": checkpoint["best_valid_loss"],
        "run_seconds": sum(row.get("duration_seconds", 0.0) for row in checkpoint["history"]),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(CHECKPOINT_DIR / (experiment_id + ".last.pt")),
    }
    payload = build_payload(
        spec=spec,
        training_result=training_result,
        test_loss=test_loss,
        generated=generated,
        generation_metrics=metrics,
        device_name=_device_name(device),
    )
    write_experiment_artifacts(RESULTS_DIR, payload)
    return payload


def _load_ai_scores(suite="core_quick"):
    path = RESULTS_DIR / (
        "ai_simulated_scores.csv"
        if suite == "handoff_full"
        else "ai_qualitative_scores.csv"
    )
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = []
    for row in rows:
        values = {key: float(row[key]) for key in ("fluency", "imagery", "relevance", "structure")}
        scores.append(
            {
                "experiment_id": row["experiment_id"],
                **values,
                "average": sum(values.values()) / len(values),
                "evaluator_type": row.get("evaluator_type", "AI assisted evaluation"),
            }
        )
    return scores


def write_report(suite="core_quick"):
    payloads = []
    for experiment_id in _suite_experiments(suite):
        path = RESULTS_DIR / (experiment_id + ".json")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    tuning = None
    tuning_trials = None
    multi_rater_summary = None
    agreement = None
    if suite == "handoff_full" and TUNING_BEST_CONFIG.exists():
        tuning = json.loads(TUNING_BEST_CONFIG.read_text(encoding="utf-8"))
        trials_path = RESULTS_DIR / "tuning" / "trials.csv"
        if trials_path.exists():
            with trials_path.open(encoding="utf-8-sig", newline="") as handle:
                tuning_trials = []
                for row in csv.DictReader(handle):
                    tuning_trials.append(
                        {
                            "trial": int(row["trial"]),
                            "hidden_dim": int(row["hidden_dim"]),
                            "lr": float(row["lr"]),
                            "effective_batch_size": int(row["effective_batch_size"]),
                            "best_epoch": int(row["best_epoch"]),
                            "best_valid_loss": float(row["best_valid_loss"]),
                            "train_seconds": float(row["train_seconds"]),
                        }
                    )
        multi_rater_dir = RESULTS_DIR / "ai_multi_rater"
        summary_path = multi_rater_dir / "model_summary.csv"
        agreement_path = multi_rater_dir / "agreement_report.json"
        if summary_path.exists() and agreement_path.exists():
            with summary_path.open(encoding="utf-8-sig", newline="") as handle:
                multi_rater_summary = []
                for row in csv.DictReader(handle):
                    converted = dict(row)
                    for key in (
                        "fluency_mean",
                        "fluency_std",
                        "imagery_mean",
                        "imagery_std",
                        "relevance_mean",
                        "relevance_std",
                        "structure_mean",
                        "structure_std",
                        "average",
                    ):
                        converted[key] = float(converted[key])
                    converted["high_disagreement_samples"] = int(
                        converted["high_disagreement_samples"]
                    )
                    multi_rater_summary.append(converted)
            agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    report = build_summary_report(
        payloads,
        _load_ai_scores(suite),
        tuning=tuning,
        tuning_trials=tuning_trials,
        multi_rater_summary=multi_rater_summary,
        agreement=agreement,
    )
    report_name = (
        "final_handoff_report.md" if suite == "handoff_full" else "final_experiment_report.md"
    )
    report_path = RESULTS_DIR / report_name
    report_path.write_text(report, encoding="utf-8")
    write_loss_plot(RESULTS_DIR, payloads)
    write_metric_plot(RESULTS_DIR, payloads)
    return report_path


def run_ai_scoring(suite="handoff_full"):
    payloads = []
    for experiment_id in _suite_experiments(suite):
        path = RESULTS_DIR / (experiment_id + ".json")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return write_ai_simulated_evaluation(RESULTS_DIR, payloads)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_experiment(args.experiment, resume=args.resume)
    elif args.command == "run-suite":
        for experiment_id in _suite_experiments(args.suite):
            run_experiment(experiment_id, resume=args.resume)
        write_report(args.suite)
    elif args.command == "evaluate":
        for experiment_id in _suite_experiments(args.suite):
            evaluate_experiment(experiment_id)
    elif args.command == "report":
        write_report(args.suite)
    elif args.command == "ai-score":
        run_ai_scoring(args.suite)
        write_report(args.suite)


if __name__ == "__main__":
    main()
