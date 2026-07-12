import csv
import json
import math
from pathlib import Path


def write_experiment_artifacts(results_dir, payload):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = payload["experiment_id"]
    json_path = results_dir / (experiment_id + ".json")
    loss_csv_path = results_dir / (experiment_id + "_loss.csv")
    generated_path = results_dir / (experiment_id + "_generated.txt")

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history = payload["training"]["history"]
    fieldnames = [
        "epoch",
        "train_loss",
        "valid_loss",
        "learning_rate",
        "ss_prob",
        "duration_seconds",
    ]
    with loss_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    lines = []
    for item in payload["generated"]:
        lines.append(
            "[{}] prompt={} seed={}\n{}\n".format(
                item["task"], item["prompt"], item["seed"], item["text"]
            )
        )
    generated_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": json_path,
        "loss_csv": loss_csv_path,
        "generated": generated_path,
    }


def write_loss_plot(results_dir, payloads):
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    figure_path = results_dir / "loss_comparison.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for payload in payloads:
        history = payload["training"]["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label=payload["experiment_id"])
        axes[1].plot(epochs, [row["valid_loss"] for row in history], marker="o", label=payload["experiment_id"])
    axes[0].set_title("Train objective loss")
    axes[1].set_title("Validation cross-entropy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def write_metric_plot(results_dir, payloads):
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    figure_path = results_dir / "metric_comparison.png"
    names = [payload["experiment_id"] for payload in payloads]
    metrics = (
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
        ("repetition_rate", "Repetition rate"),
        ("acrostic_accuracy", "Acrostic accuracy"),
        ("acrostic_completion_rate", "Acrostic completion rate"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [payload["generation_metrics"][key] for payload in payloads]
        axis.bar(names, values)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.3)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def _fmt(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return ("{:.%df}" % digits).format(value)


def build_summary_report(
    payloads,
    ai_scores,
    tuning=None,
    tuning_trials=None,
    multi_rater_summary=None,
    agreement=None,
):
    official = bool(payloads) and all(
        payload["spec"].get("recipe") == "handoff_full" for payload in payloads
    )
    title = (
        "# HANDOFF 第 7、8、13 节正式实验报告"
        if official
        else "# RNN 古诗生成核心实验报告"
    )
    lines = [
        title,
        "",
        (
            "> 三种正式模型均使用严格因果结构、相同固定切分和共享调参配置。"
            if official
            else "> 本报告为最多 5 轮快速实验，用于完成统一流程和初步模型比较，不代表充分收敛上限。"
        ),
        "",
        "## 自动评价结果",
        "",
        "| 实验 | 最佳 epoch | Valid Loss | Test Loss | PPL | Distinct-1 | Distinct-2 | 重复率 | 藏头准确率 | 藏头完成率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        training = payload["training"]
        test = payload["test"]
        metrics = payload["generation_metrics"]
        lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                payload["experiment_id"],
                training["best_epoch"],
                _fmt(training["best_valid_loss"]),
                _fmt(test["loss"]),
                _fmt(test["perplexity"]),
                _fmt(metrics["distinct_1"]),
                _fmt(metrics["distinct_2"]),
                _fmt(metrics["repetition_rate"]),
                _fmt(metrics["acrostic_accuracy"]),
                _fmt(metrics.get("acrostic_completion_rate", metrics["acrostic_accuracy"])),
            )
        )

    if official:
        lines.extend(["", "## 模型结构与共享超参数", ""])
        lines.extend([
            "| 实验 | 模型结构 | hidden | lr | effective batch | seed | max epochs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for payload in payloads:
            spec = payload["spec"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    payload["experiment_id"],
                    spec["architecture"],
                    spec["hidden_dim"],
                    spec["lr"],
                    spec["effective_batch_size"],
                    spec["seed"],
                    spec["num_epoch"],
                )
            )
        if tuning:
            parameters = tuning["parameters"]
            lines.extend([
                "",
                "## 自动调参",
                "",
                "- 选择 trial：{}".format(tuning["source_trial"]),
                "- 最佳验证损失：{}".format(_fmt(tuning["best_valid_loss"])),
                "- 共享配置：hidden_dim={}, lr={}, effective_batch_size={}".format(
                    parameters["hidden_dim"],
                    parameters["lr"],
                    parameters["effective_batch_size"],
                ),
            ])
        if tuning_trials:
            lines.extend([
                "",
                "| 调参试验 | hidden | lr | effective batch | 最佳 epoch | Best Valid Loss | 耗时（秒） |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for trial in tuning_trials:
                lines.append(
                    "| trial {} | {} | {} | {} | {} | {} | {:.1f} |".format(
                        trial["trial"],
                        trial["hidden_dim"],
                        trial["lr"],
                        trial["effective_batch_size"],
                        trial["best_epoch"],
                        _fmt(trial["best_valid_loss"]),
                        float(trial["train_seconds"]),
                    )
                )
        lines.extend(["", "## 全部共享超参数", "", "```json"])
        shared_spec = dict(payloads[0]["spec"])
        for key in ("experiment_id", "model_type", "architecture"):
            shared_spec.pop(key, None)
        lines.append(json.dumps(shared_spec, ensure_ascii=False, indent=2))
        lines.append("```")

    experiment_scope = (
        [
            "- 三个正式实验共享固定切分、随机种子、训练策略和调参选出的超参数，仅改变模型结构。",
            "- 调参仅使用训练集和验证集；测试集只在载入最佳 checkpoint 后评估一次。",
            "- 旧五轮泄漏实验完整归档于 `results/archive/quick_leaky_suite/` 和 `checkpoints/archive/quick_leaky_suite/`，不参与正式排名。",
        ]
        if official
        else [
            "- 三个 controlled 实验共享数据、随机种子、训练策略和超参数，仅改变模型结构。",
            "- poetry2_legacy_recipe 使用历史基础训练方案，与完整改进方案进行系统级比较。",
            "- 最佳 epoch 由验证集选择，测试集仅在载入最佳 checkpoint 后评估一次。",
        ]
    )
    lines.extend([
        "",
        "## 训练耗时与设备",
        "",
        "| 实验 | 训练耗时（分钟） | 设备 |",
        "|---|---:|---|",
    ])
    for payload in payloads:
        lines.append(
            "| {} | {:.2f} | {} |".format(
                payload["experiment_id"],
                payload["runtime"]["train_seconds"] / 60,
                payload["runtime"]["device"],
            )
        )

    if official and multi_rater_summary:
        lines.extend([
            "",
            "## 三智能体模拟盲评",
            "",
            "| 实验 | 流畅性 | 意境 | 相关性 | 结构完整性 | 平均分 | 高分歧样本 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for score in multi_rater_summary:
            lines.append(
                "| {experiment_id} | {fluency_mean:.2f} ± {fluency_std:.2f} | {imagery_mean:.2f} ± {imagery_std:.2f} | {relevance_mean:.2f} ± {relevance_std:.2f} | {structure_mean:.2f} ± {structure_std:.2f} | {average:.2f} | {high_disagreement_samples} |".format(**score)
            )
        strongest = max(multi_rater_summary, key=lambda score: score["average"])
        lines.extend([
            "",
            "三智能体平均分最高的是 `{}`（{:.2f}/5）。".format(
                strongest["experiment_id"], strongest["average"]
            ),
            "",
            "> evaluator_type=`AI multi-rater simulated evaluation`；三个智能体独立盲评，但不是真人盲评。",
        ])
        if agreement:
            lines.extend([
                "",
                "### 评价者一致性",
                "",
                "| 维度 | ICC(2,1) | ICC(2,k) | 结论 |",
                "|---|---:|---:|---|",
            ])
            for dimension, values in agreement["dimensions"].items():
                conclusion = "一致性较低，不作强结论" if values["low_agreement"] else "可用于辅助比较"
                lines.append(
                    "| {} | {:.3f} | {:.3f} | {} |".format(
                        dimension,
                        values["icc_2_1"],
                        values["icc_2_k"],
                        conclusion,
                    )
                )
            lines.extend(["", "两两 Spearman 等级相关："])
            for pair, value in agreement["pairwise_spearman"].items():
                lines.append("- `{}`: {:.3f}".format(pair, value))

    appendix_title = (
        "## 附录：单一启发式 AI 评价"
        if official and multi_rater_summary
        else ("## AI 模拟定性评价" if official else "## AI 辅助定性评价")
    )
    lines.extend(["", appendix_title, ""])
    if ai_scores:
        lines.extend([
            "| 实验 | 流畅性 | 意境 | 相关性 | 结构完整性 | 平均分 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for score in ai_scores:
            lines.append(
                "| {experiment_id} | {fluency:.2f} | {imagery:.2f} | {relevance:.2f} | {structure:.2f} | {average:.2f} |".format(**score)
            )
    else:
        lines.append("AI 定性评分尚未写入。")
    if ai_scores:
        strongest = max(ai_scores, key=lambda score: score["average"])
        lines.extend([
            "",
            "当前 AI 定性平均分最高的是 `{}`（{:.2f}/5）。".format(
                strongest["experiment_id"], strongest["average"]
            ),
        ])
    lines.extend([
        "",
        (
            "> evaluator_type=`AI simulated evaluation`；评分时隐藏模型名称，但不是真人盲评。"
            if official
            else "> 以上定性评价由 AI 按统一量表完成，不等同于真人盲评。"
        ),
        "",
        "## 实验口径",
        "",
        *experiment_scope,
        "",
        "## 结果解释限制",
        "",
        (
            "- 正式模型已通过未来 token 不改变历史 logits 的因果性测试。"
            if official
            else "- `PoetryModel` 使用双向 LSTM 预测下一 token，训练和评估时能够读取目标右侧信息，存在因果泄漏。"
        ),
        (
            "- AI 模拟评分用于统一辅助比较，不能替代真实人工评价。"
            if official
            else "- `PoetryModel3` 的双向编码器和全序列 attention 同样读取完整输入，因此其极低 Loss/PPL 不能与严格自回归模型直接比较。"
        ),
        "- 生成质量应结合固定 prompt、Distinct、重复率、藏头准确率和定性评价判断，不能仅按 PPL 排名。",
        (
            "- 单训练 seed 的结果不包含跨 seed 方差。"
            if official
            else "- 五轮内 scheduled sampling 从 1.0 快速降至 0.36，属于快速流程验证，不代表充分调参后的最佳质量。"
        ),
    ])
    if official:
        lines.extend(["", "## Checkpoint 路径", ""])
        for payload in payloads:
            lines.append(
                "- `{}`: best=`{}`, last=`{}`".format(
                    payload["experiment_id"],
                    payload["checkpoints"]["best"],
                    payload["checkpoints"]["last"],
                )
            )
    return "\n".join(lines) + "\n"
