import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ("fluency", "imagery", "relevance", "structure")
EVALUATOR_TYPE = "AI multi-rater simulated evaluation"


def build_clean_blind_package(payloads, shuffle_seed):
    rows = []
    for payload in payloads:
        for generated in payload["generated"]:
            rows.append(
                {
                    "experiment_id": payload["experiment_id"],
                    "task": generated["task"],
                    "prompt": generated["prompt"],
                    "generation_seed": generated["seed"],
                    "text": generated["text"],
                }
            )
    random.Random(shuffle_seed).shuffle(rows)
    public_rows = []
    mapping = {}
    for index, row in enumerate(rows, start=1):
        sample_id = "S{:03d}".format(index)
        public_rows.append(
            {
                "sample_id": sample_id,
                "task": row["task"],
                "prompt": row["prompt"],
                "generation_seed": row["generation_seed"],
                "text": row["text"],
            }
        )
        mapping[sample_id] = {
            "experiment_id": row["experiment_id"],
            "task": row["task"],
            "prompt": row["prompt"],
            "generation_seed": row["generation_seed"],
            "text": row["text"],
        }
    return public_rows, mapping


def validate_rater_scores(rows, expected_sample_ids, rater_id):
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id")
    if set(sample_ids) != set(expected_sample_ids):
        raise ValueError("rater scores do not cover the expected sample set")
    for row in rows:
        if row.get("rater_id") != rater_id:
            raise ValueError("rater_id mismatch")
        for dimension in DIMENSIONS:
            value = row.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("scores must be integer values")
            if not 1 <= value <= 5:
                raise ValueError("scores must be between 1 and 5")
        reason = str(row.get("reason", "")).strip()
        if not reason:
            raise ValueError("reason must not be empty")
        if len(reason) > 30:
            raise ValueError("reason must not exceed 30 characters")


def combine_rater_scores(rater_score_sets, mapping):
    by_sample = defaultdict(list)
    for rows in rater_score_sets:
        for row in rows:
            by_sample[row["sample_id"]].append(row)
    high_disagreement = {}
    for sample_id, rows in by_sample.items():
        high_disagreement[sample_id] = any(
            max(row[dimension] for row in rows) - min(row[dimension] for row in rows) >= 3
            for dimension in DIMENSIONS
        )
    combined = []
    for sample_id in sorted(by_sample):
        for row in by_sample[sample_id]:
            combined.append(
                {
                    **row,
                    **mapping[sample_id],
                    "high_disagreement": high_disagreement[sample_id],
                    "evaluator_type": EVALUATOR_TYPE,
                }
            )
    return combined


def _sample_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize_models(combined_rows):
    grouped = defaultdict(list)
    for row in combined_rows:
        grouped[row["experiment_id"]].append(row)
    summaries = []
    for experiment_id in sorted(grouped):
        rows = grouped[experiment_id]
        summary = {
            "experiment_id": experiment_id,
            "sample_count": len({row["sample_id"] for row in rows}),
            "rating_count": len(rows),
            "high_disagreement_samples": len(
                {row["sample_id"] for row in rows if row["high_disagreement"]}
            ),
            "evaluator_type": EVALUATOR_TYPE,
        }
        dimension_means = []
        for dimension in DIMENSIONS:
            values = [row[dimension] for row in rows]
            mean = sum(values) / len(values)
            std = _sample_std(values)
            margin = 1.96 * std / math.sqrt(len(values))
            summary[dimension + "_mean"] = mean
            summary[dimension + "_std"] = std
            summary[dimension + "_ci95_low"] = mean - margin
            summary[dimension + "_ci95_high"] = mean + margin
            dimension_means.append(mean)
        summary["average"] = sum(dimension_means) / len(dimension_means)
        summaries.append(summary)
    return summaries


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2 + 1
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(first, second):
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale == 0 or second_scale == 0:
        return 0.0
    return numerator / (first_scale * second_scale)


def _spearman(first, second):
    return _pearson(_rank(first), _rank(second))


def _icc_2(matrix):
    subject_count = len(matrix)
    rater_count = len(matrix[0])
    grand_mean = sum(sum(row) for row in matrix) / (subject_count * rater_count)
    subject_means = [sum(row) / rater_count for row in matrix]
    rater_means = [
        sum(matrix[subject][rater] for subject in range(subject_count)) / subject_count
        for rater in range(rater_count)
    ]
    ms_subject = rater_count * sum(
        (mean - grand_mean) ** 2 for mean in subject_means
    ) / max(subject_count - 1, 1)
    ms_rater = subject_count * sum(
        (mean - grand_mean) ** 2 for mean in rater_means
    ) / max(rater_count - 1, 1)
    residual_sum = 0.0
    for subject in range(subject_count):
        for rater in range(rater_count):
            residual = (
                matrix[subject][rater]
                - subject_means[subject]
                - rater_means[rater]
                + grand_mean
            )
            residual_sum += residual ** 2
    ms_error = residual_sum / max((subject_count - 1) * (rater_count - 1), 1)
    denominator_21 = (
        ms_subject
        + (rater_count - 1) * ms_error
        + rater_count * (ms_rater - ms_error) / max(subject_count, 1)
    )
    icc_2_1 = (ms_subject - ms_error) / denominator_21 if denominator_21 else 0.0
    denominator_2k = ms_subject + (ms_rater - ms_error) / max(subject_count, 1)
    icc_2_k = (ms_subject - ms_error) / denominator_2k if denominator_2k else 0.0
    return max(-1.0, min(1.0, icc_2_1)), max(-1.0, min(1.0, icc_2_k))


def calculate_agreement(combined_rows):
    sample_ids = sorted({row["sample_id"] for row in combined_rows})
    rater_ids = sorted({row["rater_id"] for row in combined_rows})
    lookup = {
        (row["sample_id"], row["rater_id"]): row for row in combined_rows
    }
    dimensions = {}
    for dimension in DIMENSIONS:
        matrix = [
            [lookup[(sample_id, rater_id)][dimension] for rater_id in rater_ids]
            for sample_id in sample_ids
        ]
        icc_2_1, icc_2_k = _icc_2(matrix)
        dimensions[dimension] = {
            "icc_2_1": icc_2_1,
            "icc_2_k": icc_2_k,
            "low_agreement": icc_2_k < 0.5,
        }
    pairwise = {}
    for left_index, left_id in enumerate(rater_ids):
        for right_id in rater_ids[left_index + 1:]:
            left_values = [
                sum(lookup[(sample_id, left_id)][dimension] for dimension in DIMENSIONS)
                / len(DIMENSIONS)
                for sample_id in sample_ids
            ]
            right_values = [
                sum(lookup[(sample_id, right_id)][dimension] for dimension in DIMENSIONS)
                / len(DIMENSIONS)
                for sample_id in sample_ids
            ]
            pairwise[left_id + "__" + right_id] = _spearman(left_values, right_values)
    return {
        "evaluator_type": EVALUATOR_TYPE,
        "rater_count": len(rater_ids),
        "sample_count": len(sample_ids),
        "dimensions": dimensions,
        "pairwise_spearman": pairwise,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_blind_packages(output_dir, payloads):
    output_dir = Path(output_dir)
    canonical_rows, mapping = build_clean_blind_package(payloads, shuffle_seed=0)
    public_by_id = {row["sample_id"]: row for row in canonical_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "private_mapping.json", mapping)
    blind_dir = output_dir / "blind_inputs"
    blind_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted(public_by_id)
    for rater_number, seed in enumerate((101, 202, 303), start=1):
        ordered_ids = list(sample_ids)
        random.Random(seed).shuffle(ordered_ids)
        write_json(
            blind_dir / "rater_{:02d}_input.json".format(rater_number),
            [public_by_id[sample_id] for sample_id in ordered_ids],
        )


def _read_score_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            converted = dict(row)
            for dimension in DIMENSIONS:
                raw_value = converted[dimension]
                converted[dimension] = (
                    int(raw_value) if raw_value.isdigit() else float(raw_value)
                )
            rows.append(converted)
    return rows


def aggregate_score_files(output_dir):
    output_dir = Path(output_dir)
    mapping = json.loads((output_dir / "private_mapping.json").read_text(encoding="utf-8"))
    expected_sample_ids = set(mapping)
    rater_score_sets = []
    for rater_number in range(1, 4):
        rater_id = "rater_{:02d}".format(rater_number)
        rows = _read_score_csv(output_dir / (rater_id + "_scores.csv"))
        validate_rater_scores(rows, expected_sample_ids, rater_id)
        rater_score_sets.append(rows)
    combined = combine_rater_scores(rater_score_sets, mapping)
    summary = summarize_models(combined)
    agreement = calculate_agreement(combined)
    write_csv(output_dir / "all_scores_detail.csv", combined)
    write_csv(output_dir / "model_summary.csv", summary)
    write_json(output_dir / "agreement_report.json", agreement)
    return combined, summary, agreement
