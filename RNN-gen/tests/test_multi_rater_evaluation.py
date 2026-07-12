import math
import json

import pytest

from multi_rater_evaluation import (
    build_clean_blind_package,
    calculate_agreement,
    combine_rater_scores,
    summarize_models,
    validate_rater_scores,
    prepare_blind_packages,
    aggregate_score_files,
)


def _payload(experiment_id, text):
    return {
        "experiment_id": experiment_id,
        "generated": [
            {
                "task": "continuation",
                "prompt": "春风",
                "seed": 123,
                "text": text,
            }
        ],
    }


def _score(sample_id, rater_id, value):
    return {
        "sample_id": sample_id,
        "rater_id": rater_id,
        "fluency": value,
        "imagery": value,
        "relevance": value,
        "structure": value,
        "reason": "评分理由",
    }


def test_clean_blind_package_hides_model_and_previous_scores():
    payloads = [
        _payload("poetry1_causal", "春风入夜来"),
        _payload("poetry2_causal", "春风过柳堤"),
    ]

    public_rows, mapping = build_clean_blind_package(payloads, shuffle_seed=7)

    assert len(public_rows) == 2
    assert set(public_rows[0]) == {
        "sample_id",
        "task",
        "prompt",
        "generation_seed",
        "text",
    }
    assert {row["sample_id"] for row in public_rows} == set(mapping)
    assert {item["experiment_id"] for item in mapping.values()} == {
        "poetry1_causal",
        "poetry2_causal",
    }


def test_validate_rater_scores_requires_complete_unique_integer_scores():
    expected = {"S001", "S002"}
    valid = [_score("S001", "rater_01", 3), _score("S002", "rater_01", 4)]

    validate_rater_scores(valid, expected, "rater_01")

    with pytest.raises(ValueError, match="duplicate"):
        validate_rater_scores(valid + [valid[0]], expected, "rater_01")
    invalid = [dict(valid[0], fluency=2.5), valid[1]]
    with pytest.raises(ValueError, match="integer"):
        validate_rater_scores(invalid, expected, "rater_01")


def test_combined_scores_restore_models_and_flag_large_disagreement():
    mapping = {
        "S001": {"experiment_id": "poetry1_causal", "task": "continuation"}
    }
    raters = [
        [_score("S001", "rater_01", 1)],
        [_score("S001", "rater_02", 4)],
        [_score("S001", "rater_03", 5)],
    ]

    combined = combine_rater_scores(raters, mapping)

    assert len(combined) == 3
    assert all(row["experiment_id"] == "poetry1_causal" for row in combined)
    assert all(row["high_disagreement"] is True for row in combined)


def test_model_summary_reports_mean_std_ci_and_total_average():
    mapping = {
        "S001": {"experiment_id": "poetry1_causal", "task": "continuation"}
    }
    raters = [
        [_score("S001", "rater_01", 2)],
        [_score("S001", "rater_02", 3)],
        [_score("S001", "rater_03", 4)],
    ]
    combined = combine_rater_scores(raters, mapping)

    summary = summarize_models(combined)[0]

    assert summary["experiment_id"] == "poetry1_causal"
    assert summary["fluency_mean"] == pytest.approx(3.0)
    assert summary["fluency_std"] == pytest.approx(1.0)
    assert summary["fluency_ci95_low"] == pytest.approx(3 - 1.96 / math.sqrt(3))
    assert summary["fluency_ci95_high"] == pytest.approx(3 + 1.96 / math.sqrt(3))
    assert summary["average"] == pytest.approx(3.0)


def test_agreement_reports_icc_and_pairwise_spearman():
    mapping = {
        "S001": {"experiment_id": "poetry1_causal", "task": "continuation"},
        "S002": {"experiment_id": "poetry1_causal", "task": "continuation"},
        "S003": {"experiment_id": "poetry1_causal", "task": "continuation"},
    }
    rater_1 = [_score("S001", "rater_01", 1), _score("S002", "rater_01", 2), _score("S003", "rater_01", 3)]
    rater_2 = [_score("S001", "rater_02", 1), _score("S002", "rater_02", 2), _score("S003", "rater_02", 3)]
    rater_3 = [_score("S001", "rater_03", 2), _score("S002", "rater_03", 3), _score("S003", "rater_03", 4)]
    combined = combine_rater_scores([rater_1, rater_2, rater_3], mapping)

    agreement = calculate_agreement(combined)

    assert agreement["dimensions"]["fluency"]["icc_2_1"] <= 1
    assert agreement["dimensions"]["fluency"]["icc_2_k"] <= 1
    assert agreement["dimensions"]["fluency"]["low_agreement"] is False
    assert agreement["pairwise_spearman"]["rater_01__rater_02"] == pytest.approx(1.0)


def test_prepare_blind_packages_writes_three_clean_independent_orders(tmp_path):
    payloads = [
        _payload("poetry1_causal", "春风入夜来"),
        _payload("poetry2_causal", "春风过柳堤"),
        _payload("poetry3_causal", "春风满故园"),
    ]

    prepare_blind_packages(tmp_path, payloads)

    orders = []
    for rater_number in range(1, 4):
        path = tmp_path / "blind_inputs" / "rater_{:02d}_input.json".format(rater_number)
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert all(set(row) == {"sample_id", "task", "prompt", "generation_seed", "text"} for row in rows)
        orders.append([row["sample_id"] for row in rows])
    assert len({tuple(order) for order in orders}) == 3
    mapping = json.loads((tmp_path / "private_mapping.json").read_text(encoding="utf-8"))
    assert len(mapping) == 3


def test_aggregate_score_files_writes_required_artifacts(tmp_path):
    payloads = [
        _payload("poetry1_causal", "春风入夜来"),
        _payload("poetry2_causal", "春风过柳堤"),
        _payload("poetry3_causal", "春风满故园"),
    ]
    prepare_blind_packages(tmp_path, payloads)
    mapping = json.loads((tmp_path / "private_mapping.json").read_text(encoding="utf-8"))
    sample_ids = sorted(mapping)
    for number in range(1, 4):
        rows = [_score(sample_id, "rater_{:02d}".format(number), number + 1) for sample_id in sample_ids]
        path = tmp_path / "rater_{:02d}_scores.csv".format(number)
        import csv
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    aggregate_score_files(tmp_path)

    assert len(list((tmp_path).glob("rater_*_scores.csv"))) == 3
    assert (tmp_path / "all_scores_detail.csv").exists()
    assert (tmp_path / "model_summary.csv").exists()
    assert (tmp_path / "agreement_report.json").exists()
