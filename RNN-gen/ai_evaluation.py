import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from evaluation import acrostic_accuracy, acrostic_completion_rate, mean_repetition_rate


EVALUATOR_TYPE = "AI simulated evaluation"
IMAGERY_CHARACTERS = set("山水月风花云日夜江河春秋雪雨星竹松梅柳鸟雁烟霞天海")


def _clamp_score(value):
    return max(1, min(5, int(value)))


def build_blind_rows(payloads, shuffle_seed=123):
    rows = []
    for payload in payloads:
        for generated in payload["generated"]:
            rows.append(
                {
                    "_experiment_id": payload["experiment_id"],
                    "task": generated["task"],
                    "prompt": generated["prompt"],
                    "generation_seed": generated["seed"],
                    "text": generated["text"],
                }
            )
    random.Random(shuffle_seed).shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["blind_id"] = "B{:03d}".format(index)
    return rows


def _score_row(row):
    text = re.sub(r"\s+", "", row["text"])
    content = re.sub(r"[，。！？；：、,.!?;:]", "", text)
    unique_ratio = len(set(content)) / max(len(content), 1)
    repetition = mean_repetition_rate([content], n=2)
    duplicated_characters = sum(
        first == second for first, second in zip(content, content[1:])
    )
    malformed_punctuation = len(re.findall(r"[，。！？；：、,.!?;:]{2,}", text))
    fluency = 2 + (len(content) >= 20) + (unique_ratio >= 0.55) + (repetition <= 0.05)
    fluency -= min(2, duplicated_characters + malformed_punctuation)
    fluency = _clamp_score(fluency)
    imagery_count = len(IMAGERY_CHARACTERS.intersection(content))
    imagery = _clamp_score(1 + min(imagery_count // 2, 4))
    if row["task"] == "acrostic":
        record = {"prompt": row["prompt"], "text": row["text"]}
        relevance = _clamp_score(1 + round(4 * acrostic_accuracy([record])))
        structure = _clamp_score(1 + round(4 * acrostic_completion_rate([record])))
    else:
        starts_with_prompt = text.startswith(row["prompt"])
        relevance = _clamp_score(1 + 2 * starts_with_prompt + (imagery_count > 0) + (len(content) >= 20))
        sentence_count = len([part for part in re.split(r"[。！？!?]", text) if part])
        structure = _clamp_score(1 + min(sentence_count, 4))
    return {
        "fluency": fluency,
        "imagery": imagery,
        "relevance": relevance,
        "structure": structure,
    }


def score_ai_simulated_rows(blind_rows):
    public_rows = []
    detail_rows = []
    for row in blind_rows:
        scores = _score_row(row)
        public = {
            key: row[key]
            for key in ("blind_id", "task", "prompt", "generation_seed", "text")
        }
        public.update(scores)
        public["evaluator_type"] = EVALUATOR_TYPE
        public_rows.append(public)
        detail = dict(public)
        detail["experiment_id"] = row["_experiment_id"]
        detail_rows.append(detail)
    return public_rows, detail_rows


def summarize_scores(detail_rows):
    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[row["experiment_id"]].append(row)
    summaries = []
    dimensions = ("fluency", "imagery", "relevance", "structure")
    for experiment_id in sorted(grouped):
        rows = grouped[experiment_id]
        values = {
            dimension: sum(row[dimension] for row in rows) / len(rows)
            for dimension in dimensions
        }
        summaries.append(
            {
                "experiment_id": experiment_id,
                **values,
                "average": sum(values.values()) / len(values),
                "sample_count": len(rows),
                "evaluator_type": EVALUATOR_TYPE,
            }
        )
    return summaries


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_ai_simulated_evaluation(results_dir, payloads):
    results_dir = Path(results_dir)
    blind_rows = build_blind_rows(payloads)
    public_rows, detail_rows = score_ai_simulated_rows(blind_rows)
    summaries = summarize_scores(detail_rows)
    _write_csv(results_dir / "ai_simulated_blind_scores.csv", public_rows)
    _write_csv(results_dir / "ai_simulated_scores_detail.csv", detail_rows)
    _write_csv(results_dir / "ai_simulated_scores.csv", summaries)
    return summaries
