import csv
import json

from ai_evaluation import build_blind_rows, score_ai_simulated_rows, summarize_scores
from experiment import (
    _reduced_micro_batch,
    build_parser,
    build_payload,
    build_runtime_config,
)
from experiment_config import get_experiment
from reporting import build_summary_report, write_experiment_artifacts, write_metric_plot


def _payload(experiment_id="poetry1_controlled"):
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "spec": {"model_type": "poetry1", "recipe": "controlled"},
        "runtime": {"device": "cpu", "train_seconds": 1.5},
        "training": {
            "best_epoch": 1,
            "best_valid_loss": 1.2,
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 1.4,
                    "valid_loss": 1.2,
                    "learning_rate": 0.001,
                    "ss_prob": 1.0,
                    "duration_seconds": 1.5,
                }
            ],
        },
        "test": {"loss": 1.3, "perplexity": 3.6693},
        "generation_metrics": {
            "distinct_1": 0.6,
            "distinct_2": 0.8,
            "repetition_rate": 0.1,
            "acrostic_accuracy": 0.75,
            "acrostic_completion_rate": 0.75,
        },
        "generated": [
            {"task": "continuation", "prompt": "丽日照残春", "seed": 123, "text": "丽日照残春，花明满故园。"}
        ],
    }


def test_write_experiment_artifacts_uses_stable_schema(tmp_path):
    payload = _payload()

    paths = write_experiment_artifacts(tmp_path, payload)

    stored = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    assert stored["experiment_id"] == "poetry1_controlled"
    with paths["loss_csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1"
    generated = paths["generated"].read_text(encoding="utf-8")
    assert "丽日照残春" in generated
    assert "seed=123" in generated


def test_summary_report_compares_all_experiments():
    payloads = [
        _payload("poetry2_legacy_recipe"),
        _payload("poetry1_controlled"),
        _payload("poetry2_controlled"),
        _payload("poetry3_controlled"),
    ]

    report = build_summary_report(payloads, ai_scores=[])

    assert "# RNN 古诗生成核心实验报告" in report
    assert "poetry2_legacy_recipe" in report
    assert "poetry3_controlled" in report
    assert "5 轮快速实验" in report
    assert "AI 辅助定性评价" in report
    assert "因果泄漏" in report


def test_cli_supports_run_suite_resume_and_reporting():
    parser = build_parser()

    suite_args = parser.parse_args(["run-suite", "--suite", "core_quick", "--resume"])
    report_args = parser.parse_args(["report", "--suite", "core_quick"])

    assert suite_args.command == "run-suite"
    assert suite_args.resume is True
    assert report_args.command == "report"

    full_args = parser.parse_args(["run-suite", "--suite", "handoff_full", "--resume"])
    ai_args = parser.parse_args(["ai-score", "--suite", "handoff_full"])

    assert full_args.suite == "handoff_full"
    assert ai_args.command == "ai-score"


def test_runtime_config_is_derived_from_experiment_spec():
    spec = get_experiment("poetry3_controlled")

    config = build_runtime_config(spec)

    assert config.split_strategy == "text"
    assert config.batch_size == 32
    assert config.num_epoch == 5
    assert config.max_len == 125
    assert config.do_predict is False


def test_build_payload_contains_required_handoff_fields():
    spec = get_experiment("poetry1_controlled")
    training_result = {
        "history": [{"epoch": 1, "train_loss": 1.4, "valid_loss": 1.2}],
        "best_epoch": 1,
        "best_valid_loss": 1.2,
        "run_seconds": 10.0,
        "best_checkpoint": "checkpoints/a.best.pt",
        "last_checkpoint": "checkpoints/a.last.pt",
    }

    payload = build_payload(
        spec=spec,
        training_result=training_result,
        test_loss=1.3,
        generated=[],
        generation_metrics={
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repetition_rate": 0.0,
            "acrostic_accuracy": 0.0,
            "acrostic_completion_rate": 0.0,
        },
        device_name="cpu",
    )

    assert payload["spec"]["seed"] == 123
    assert payload["training"]["best_epoch"] == 1
    assert payload["test"]["perplexity"] > 1
    assert payload["runtime"]["device"] == "cpu"


def test_metric_plot_is_written(tmp_path):
    payloads = [_payload("poetry1_controlled"), _payload("poetry2_controlled")]

    path = write_metric_plot(tmp_path, payloads)

    assert path.exists()
    assert path.stat().st_size > 0


def test_ai_simulated_scoring_is_blinded_and_keeps_provenance():
    payloads = [_payload("poetry1_causal"), _payload("poetry2_causal"), _payload("poetry3_causal")]

    blind_rows = build_blind_rows(payloads, shuffle_seed=123)
    public_rows, detail_rows = score_ai_simulated_rows(blind_rows)
    summary = summarize_scores(detail_rows)

    assert len(public_rows) == 3
    assert all("experiment_id" not in row for row in public_rows)
    assert {row["experiment_id"] for row in detail_rows} == {
        "poetry1_causal",
        "poetry2_causal",
        "poetry3_causal",
    }
    assert all(row["evaluator_type"] == "AI simulated evaluation" for row in detail_rows)
    assert all(1 <= row["fluency"] <= 5 for row in detail_rows)
    assert len(summary) == 3


def test_handoff_report_contains_structures_tuning_and_ai_provenance():
    payloads = [_payload("poetry1_causal"), _payload("poetry2_causal"), _payload("poetry3_causal")]
    for payload in payloads:
        payload["spec"].update(
            {
                "recipe": "handoff_full",
                "architecture": "Embedding -> UniLSTM -> Linear",
                "seed": 123,
                "num_epoch": 20,
                "batch_size": 32,
                "effective_batch_size": 32,
                "hidden_dim": 256,
                "lr": 5e-4,
                "label_smoothing": 0.1,
            }
        )
        payload["checkpoints"] = {"best": "a.best.pt", "last": "a.last.pt"}
    ai_scores = [
        {
            "experiment_id": payload["experiment_id"],
            "fluency": 3.0,
            "imagery": 3.0,
            "relevance": 3.0,
            "structure": 3.0,
            "average": 3.0,
            "evaluator_type": "AI simulated evaluation",
        }
        for payload in payloads
    ]
    tuning = {
        "source_trial": 1,
        "parameters": {"hidden_dim": 256, "lr": 5e-4, "effective_batch_size": 32},
        "best_valid_loss": 1.2,
    }
    tuning_trials = [
        {
            "trial": index,
            "hidden_dim": 256,
            "lr": 5e-4,
            "effective_batch_size": 32,
            "best_epoch": 1,
            "best_valid_loss": 1.2 + index / 100,
            "train_seconds": 10 + index,
        }
        for index in range(1, 7)
    ]

    report = build_summary_report(
        payloads, ai_scores, tuning=tuning, tuning_trials=tuning_trials
    )

    assert "HANDOFF 第 7、8、13 节正式实验报告" in report
    assert "Embedding -> UniLSTM -> Linear" in report
    assert "自动调参" in report
    assert "AI simulated evaluation" in report
    assert "不是真人盲评" in report
    assert "a.best.pt" in report
    assert "trial 6" in report
    assert '"label_smoothing"' in report
    assert "results/archive/quick_leaky_suite" in report
    assert "poetry2_legacy_recipe 使用历史" not in report


def test_cuda_oom_fallback_preserves_effective_batch_size():
    spec = get_experiment("poetry3_causal")

    reduced = _reduced_micro_batch(spec)

    assert reduced.batch_size == 16
    assert reduced.effective_batch_size == 32
    assert reduced.gradient_accumulation_steps == 2


def test_handoff_report_prefers_multi_rater_summary_and_keeps_single_ai_as_appendix():
    payloads = [_payload("poetry1_causal"), _payload("poetry2_causal"), _payload("poetry3_causal")]
    for payload in payloads:
        payload["spec"].update({"recipe": "handoff_full", "architecture": "causal", "seed": 123, "num_epoch": 20, "effective_batch_size": 32, "hidden_dim": 512, "lr": 1e-3})
        payload["checkpoints"] = {"best": "best.pt", "last": "last.pt"}
    single_ai = [
        {"experiment_id": payload["experiment_id"], "fluency": 3, "imagery": 3, "relevance": 3, "structure": 3, "average": 3}
        for payload in payloads
    ]
    multi_rater = [
        {
            "experiment_id": payload["experiment_id"],
            "fluency_mean": 3.2,
            "fluency_std": 0.4,
            "imagery_mean": 3.0,
            "imagery_std": 0.5,
            "relevance_mean": 3.4,
            "relevance_std": 0.3,
            "structure_mean": 3.1,
            "structure_std": 0.6,
            "average": 3.175,
            "high_disagreement_samples": 2,
        }
        for payload in payloads
    ]
    agreement = {
        "dimensions": {
            dimension: {"icc_2_1": 0.4, "icc_2_k": 0.67, "low_agreement": False}
            for dimension in ("fluency", "imagery", "relevance", "structure")
        },
        "pairwise_spearman": {"rater_01__rater_02": 0.55},
    }

    report = build_summary_report(
        payloads,
        single_ai,
        multi_rater_summary=multi_rater,
        agreement=agreement,
    )

    assert "三智能体模拟盲评" in report
    assert "3.20 ± 0.40" in report
    assert "ICC(2,1)" in report
    assert "rater_01__rater_02" in report
    assert "高分歧样本" in report
    assert "附录：单一启发式 AI 评价" in report
    assert "不是真人盲评" in report
