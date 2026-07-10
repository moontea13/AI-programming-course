import csv
import hashlib
import json
import re
from math import sqrt
from statistics import mean, median
from collections import Counter
from pathlib import Path


RAW_DIR = Path("data/tang")
OUT_DIR = Path("data/cleaning_analysis")
CLEAN_TEXT_PATH = OUT_DIR / "peot_cleaned.txt"
TRAIN_TEXT_PATH = OUT_DIR / "peot_cleaned_train.txt"
VALID_TEXT_PATH = OUT_DIR / "peot_cleaned_valid.txt"
TEST_TEXT_PATH = OUT_DIR / "peot_cleaned_test.txt"
AUTHOR_HOLDOUT_TRAIN_TEXT_PATH = OUT_DIR / "peot_author_holdout_train.txt"
AUTHOR_HOLDOUT_VALID_TEXT_PATH = OUT_DIR / "peot_author_holdout_valid.txt"
AUTHOR_HOLDOUT_TEST_TEXT_PATH = OUT_DIR / "peot_author_holdout_test.txt"
AUTHOR_WEIGHT_PATH = OUT_DIR / "author_weights.csv"
TRAIN_SAMPLE_WEIGHT_PATH = OUT_DIR / "train_sample_weights.csv"
AUTHOR_HOLDOUT_WEIGHT_PATH = OUT_DIR / "author_holdout_weights.csv"
AUTHOR_HOLDOUT_SAMPLE_WEIGHT_PATH = OUT_DIR / "author_holdout_train_sample_weights.csv"
FIGURE_DIR = OUT_DIR / "figures"
METADATA_PATH = OUT_DIR / "cleaned_metadata.csv"
DROPPED_PATH = OUT_DIR / "dropped_records.csv"
SUMMARY_PATH = OUT_DIR / "summary.json"
REPORT_PATH = OUT_DIR / "bias_report.md"

MIN_TEXT_LEN = 8
MAX_TEXT_LEN = 123
MIN_AUTHOR_WEIGHT = 0.05
MAX_AUTHOR_WEIGHT = 0.50

PUNCT_TRANSLATION = str.maketrans(
    {
        ",": "，",
        ".": "。",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
        "、": "，",
        "\u3000": "",
        " ": "",
        "\t": "",
        "\n": "",
        "\r": "",
    }
)

NOTE_PATTERNS = [
    re.compile(r"（[^）]*）"),
    re.compile(r"\([^)]*\)"),
    re.compile(r"\{[^}]*\}"),
    re.compile(r"\[[^\]]*\]"),
    re.compile(r"【[^】]*】"),
    re.compile(r"〖[^〗]*〗"),
    re.compile(r"〔[^〕]*〕"),
    re.compile(r"〈[^〉]*〉"),
    re.compile(r"《[^》]*》"),
]

SENTENCE_SPLIT = re.compile(r"[，。！？；：]")
CHUNK_BOUNDARY_SPLIT = re.compile(r"(?<=[。！？；])")
ALLOWED_PUNCTUATION = {"，", "。", "！", "？", "；", "："}
CJK_RANGES = (
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F), # CJK Extensions B-G and compatibility supplement
)

TITLE_KEYWORDS = {
    "送别": ["送", "别", "离", "辞", "饯", "留别"],
    "山水": ["山", "水", "溪", "江", "湖", "峰"],
    "寺庙宗教": ["寺", "僧", "禅", "佛", "观"],
    "宫廷": ["宫", "帝", "皇", "御", "殿"],
    "季节": ["春", "夏", "秋", "冬"],
    "夜月": ["夜", "月", "夕"],
    "边塞战争": ["塞", "军", "战", "戍", "兵"],
    "饮酒": ["酒", "醉"],
    "赠寄酬答": ["赠", "寄", "酬", "答", "奉和"],
}

UNKNOWN_AUTHORS = {"不詳", "不详", "無名氏", "无名氏", "佚名", "未知"}
LENGTH_BUCKETS = ["0-20", "21-40", "41-60", "61-80", "81-100", "101-123", "124+"]


def get_converter():
    try:
        import opencc

        return opencc.OpenCC("t2s").convert, "opencc"
    except Exception:
        return lambda text: text, "none"


def is_chinese_char(ch):
    """Return whether a character belongs to a CJK ideograph block used by the corpus."""
    if ch == "〇":
        return True
    code_point = ord(ch)
    return any(start <= code_point <= end for start, end in CJK_RANGES)


def is_allowed_char(ch):
    """Return whether a character is valid poetry content or approved punctuation."""
    return is_chinese_char(ch) or ch in ALLOWED_PUNCTUATION


def load_raw_records(raw_dir):
    records = []
    for path in sorted(raw_dir.glob("poet.tang.*.json")):
        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            paragraphs = item.get("paragraphs") or []
            records.append(
                {
                    "source_file": path.name,
                    "id": item.get("id", ""),
                    "author": item.get("author", "") or "未知",
                    "title": item.get("title", "") or "无题",
                    "raw_text": "".join(paragraphs),
                    "paragraph_count": len(paragraphs),
                }
            )
    return records


def clean_text(text, convert):
    text = text.translate(PUNCT_TRANSLATION)
    for pattern in NOTE_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"[0-9０-９\-－—]+", "", text)
    text = convert(text)
    text = "".join(ch for ch in text if is_allowed_char(ch))
    text = re.sub(r"[，。！？；：]{2,}", "。", text)
    return text.strip("，。！？；：")


def content_length(text):
    return len(chinese_chars(text))


def chinese_chars(text):
    return [ch for ch in text if is_chinese_char(ch)]


def sentence_lengths(text):
    return [content_length(part) for part in SENTENCE_SPLIT.split(text) if part]


def classify_form(text):
    lengths = sentence_lengths(text)
    if not lengths:
        return "空文本"
    if all(length == 5 for length in lengths):
        return "五言"
    if all(length == 7 for length in lengths):
        return "七言"
    if all(length in {5, 7} for length in lengths):
        return "五七混合"
    return "其他"


def classify_fine_form(text):
    lengths = sentence_lengths(text)
    sentence_count = len(lengths)
    if not lengths:
        return "空文本"
    if all(length == 5 for length in lengths):
        if sentence_count == 4:
            return "五言绝句"
        if sentence_count == 8:
            return "五言律诗"
        if sentence_count > 8:
            return "五言长篇/排律"
        return "五言其他"
    if all(length == 7 for length in lengths):
        if sentence_count == 4:
            return "七言绝句"
        if sentence_count == 8:
            return "七言律诗"
        if sentence_count > 8:
            return "七言长篇/排律"
        return "七言其他"
    if all(length in {5, 7} for length in lengths):
        return "五七混合"
    return "其他"


def has_abnormal_symbols(raw_text, cleaned_text):
    cleaned_set = set(cleaned_text)
    suspicious = []
    for ch in raw_text:
        if ch.isspace():
            continue
        if ch in cleaned_set:
            continue
        if is_allowed_char(ch):
            continue
        if ch in "（）(){}[]【】《》「」『』0123456789０１２３４５６７８９-－—":
            continue
        suspicious.append(ch)
    return "".join(sorted(set(suspicious)))[:30]


def keyword_stats(records):
    counts = Counter()
    for record in records:
        title = record["title"]
        for name, keywords in TITLE_KEYWORDS.items():
            if any(keyword in title for keyword in keywords):
                counts[name] += 1
    return counts


def bucket_length(length):
    if length <= 20:
        return "0-20"
    if length <= 40:
        return "21-40"
    if length <= 60:
        return "41-60"
    if length <= 80:
        return "61-80"
    if length <= 100:
        return "81-100"
    if length <= MAX_TEXT_LEN:
        return "101-123"
    return "124+"


def safe_mean(values):
    return mean(values) if values else 0


def safe_median(values):
    return median(values) if values else 0


def removed_symbol_counter(raw_records, cleaned_candidates):
    counter = Counter()
    for record, cleaned in zip(raw_records, cleaned_candidates):
        cleaned_set = set(cleaned)
        for ch in record["raw_text"]:
            if ch.isspace() or ch in cleaned_set or is_allowed_char(ch):
                continue
            counter[ch] += 1
    return counter


def build_before_after_stats(raw_records, cleaned_candidates, kept):
    raw_lengths = [content_length(record["raw_text"]) for record in raw_records]
    candidate_lengths = [content_length(text) for text in cleaned_candidates]
    final_lengths = [row["char_len"] for row in kept]
    raw_chars = Counter(ch for record in raw_records for ch in chinese_chars(record["raw_text"]))
    candidate_chars = Counter(ch for text in cleaned_candidates for ch in chinese_chars(text))
    final_chars = Counter(ch for row in kept for ch in chinese_chars(row["cleaned_text"]))
    removed_symbols = removed_symbol_counter(raw_records, cleaned_candidates)

    return {
        "raw_avg_len": safe_mean(raw_lengths),
        "candidate_avg_len": safe_mean(candidate_lengths),
        "final_avg_len": safe_mean(final_lengths),
        "raw_median_len": safe_median(raw_lengths),
        "candidate_median_len": safe_median(candidate_lengths),
        "final_median_len": safe_median(final_lengths),
        "raw_vocab_size": len(raw_chars),
        "candidate_vocab_size": len(candidate_chars),
        "final_vocab_size": len(final_chars),
        "raw_total_chars": sum(raw_lengths),
        "candidate_total_chars": sum(candidate_lengths),
        "final_total_chars": sum(final_lengths),
        "removed_symbol_types": len(removed_symbols),
        "removed_symbol_count": sum(removed_symbols.values()),
        "top_removed_symbols": removed_symbols.most_common(30),
    }


def build_author_tail_stats(author_counter, total):
    counts = sorted(author_counter.values(), reverse=True)
    unknown_count = sum(author_counter[name] for name in UNKNOWN_AUTHORS)
    return {
        "author_count": len(author_counter),
        "single_poem_author_count": sum(1 for count in counts if count == 1),
        "single_poem_author_ratio": sum(1 for count in counts if count == 1) / max(len(counts), 1),
        "top1_ratio": sum(counts[:1]) / max(total, 1),
        "top5_ratio": sum(counts[:5]) / max(total, 1),
        "top10_ratio": sum(counts[:10]) / max(total, 1),
        "top20_ratio": sum(counts[:20]) / max(total, 1),
        "unknown_author_count": unknown_count,
        "unknown_author_ratio": unknown_count / max(total, 1),
    }


def assign_split(text):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def assign_author_split(author, parent_key):
    """Keep known authors in one split while hashing unknown-author records by parent id."""
    if author not in UNKNOWN_AUTHORS:
        return assign_split("author:" + author)
    return assign_split("record:" + parent_key)


def split_long_poem(text, max_len=MAX_TEXT_LEN):
    """Split an overlength poem at sentence boundaries without truncating any sentence."""
    sentences = [part for part in CHUNK_BOUNDARY_SPLIT.split(text) if content_length(part)]
    chunks = []
    unsplittable_sentences = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = content_length(sentence)
        if sentence_len > max_len:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            unsplittable_sentences.append(sentence)
            continue
        if current and current_len + sentence_len > max_len:
            chunks.append("".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append("".join(current))
    return chunks, unsplittable_sentences


def build_sample_row(record, text, source_kind, chunk_index, chunk_count, original_char_len, abnormal):
    parent_id = record["id"] or hashlib.md5(
        (record["source_file"] + record["raw_text"]).encode("utf-8")
    ).hexdigest()
    return {
        **record,
        "parent_id": parent_id,
        "training_id": "{}#{}".format(parent_id, chunk_index),
        "source_kind": source_kind,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "original_char_len": original_char_len,
        "cleaned_text": text,
        "char_len": content_length(text),
        "sentence_count": len(sentence_lengths(text)),
        "form": classify_form(text),
        "fine_form": classify_fine_form(text),
        "split": assign_split(parent_id),
        "author_split": assign_author_split(record["author"], parent_id),
        "abnormal_symbols_removed": abnormal,
    }


def build_author_weights(rows, split_field):
    """Build capped inverse-square-root sample weights for one training split."""
    train_rows = [row for row in rows if row[split_field] == "train"]
    author_counter = Counter(row["author"] for row in train_rows)
    raw_weights = {
        author: 1.0 / sqrt(count)
        for author, count in author_counter.items()
    }
    clipped_weights = {
        author: min(MAX_AUTHOR_WEIGHT, max(MIN_AUTHOR_WEIGHT, weight))
        for author, weight in raw_weights.items()
    }
    normalizer = safe_mean([clipped_weights[row["author"]] for row in train_rows])
    normalizer = normalizer or 1.0
    normalized_weights = {
        author: weight / normalizer
        for author, weight in clipped_weights.items()
    }
    author_rows = [
        {
            "author": author,
            "train_sample_count": count,
            "raw_weight": "{:.8f}".format(raw_weights[author]),
            "clipped_weight": "{:.8f}".format(clipped_weights[author]),
            "sample_weight": "{:.8f}".format(normalized_weights[author]),
        }
        for author, count in author_counter.most_common()
    ]
    sample_rows = [
        {
            "sample_index": index,
            "training_id": row["training_id"],
            "parent_id": row["parent_id"],
            "author": row["author"],
            "sample_weight": "{:.8f}".format(normalized_weights[row["author"]]),
        }
        for index, row in enumerate(train_rows)
    ]
    stats = {
        "train_author_count": len(author_counter),
        "train_sample_count": len(train_rows),
        "weight_min": min(normalized_weights.values(), default=0.0),
        "weight_max": max(normalized_weights.values(), default=0.0),
        "weight_ratio": (
            max(normalized_weights.values()) / min(normalized_weights.values())
            if normalized_weights else 0.0
        ),
    }
    return author_rows, sample_rows, stats


def write_split_files(rows, split_field, train_path, valid_path, test_path):
    paths = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }
    for split, path in paths.items():
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(row["cleaned_text"] for row in rows if row[split_field] == split))


def author_holdout_diagnostics(rows):
    known_author_sets = {
        split: {
            row["author"]
            for row in rows
            if row["author_split"] == split and row["author"] not in UNKNOWN_AUTHORS
        }
        for split in ("train", "valid", "test")
    }
    return {
        "known_train_authors": len(known_author_sets["train"]),
        "known_valid_authors": len(known_author_sets["valid"]),
        "known_test_authors": len(known_author_sets["test"]),
        "train_valid_overlap": len(known_author_sets["train"] & known_author_sets["valid"]),
        "train_test_overlap": len(known_author_sets["train"] & known_author_sets["test"]),
        "valid_test_overlap": len(known_author_sets["valid"] & known_author_sets["test"]),
    }


def create_visualizations(author_counter, length_counter, fine_form_counter, before_after):
    """Create report-ready PNG charts from the cleaning audit statistics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    top_authors = author_counter.most_common(15)
    author_names = [name for name, _ in reversed(top_authors)]
    author_counts = [count for _, count in reversed(top_authors)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(author_names, author_counts, color="#2a6f97")
    ax.set_title("Top 15 Authors by Retained Poem Count")
    ax.set_xlabel("Poem count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "author_top15.png", dpi=180)
    plt.close(fig)

    buckets = LENGTH_BUCKETS
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(buckets, [length_counter[bucket] for bucket in buckets], color="#588157")
    ax.set_title("Retained Poem Length Distribution")
    ax.set_xlabel("Chinese character count")
    ax.set_ylabel("Poem count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "length_distribution.png", dpi=180)
    plt.close(fig)

    form_items = fine_form_counter.most_common()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [name for name, _ in form_items],
        [count for _, count in form_items],
        color="#bc6c25",
    )
    ax.set_title("Approximate Poetic Form Distribution")
    ax.set_ylabel("Poem count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fine_form_distribution.png", dpi=180)
    plt.close(fig)

    vocab_labels = ["Raw", "Normalized", "Final"]
    vocab_values = [
        before_after["raw_vocab_size"],
        before_after["candidate_vocab_size"],
        before_after["final_vocab_size"],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(vocab_labels, vocab_values, color=["#6c757d", "#457b9d", "#2a9d8f"])
    ax.set_title("Vocabulary Size Before and After Cleaning")
    ax.set_ylabel("Unique CJK characters")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vocabulary_before_after.png", dpi=180)
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary, author_top20, length_counter, form_counter, fine_form_counter, keyword_counter):
    author_lines = "\n".join(
        f"| {idx + 1} | {author} | {count} | {ratio:.2%} |"
        for idx, (author, count, ratio) in enumerate(author_top20)
    )
    length_lines = "\n".join(
        f"| {bucket} | {length_counter[bucket]} |"
        for bucket in ["0-20", "21-40", "41-60", "61-80", "81-100", "101-123", "124+"]
    )
    form_lines = "\n".join(f"| {name} | {count} |" for name, count in form_counter.most_common())
    fine_form_lines = "\n".join(f"| {name} | {count} |" for name, count in fine_form_counter.most_common())
    keyword_lines = "\n".join(f"| {name} | {count} |" for name, count in keyword_counter.most_common())
    before_after = summary["before_after"]
    author_tail = summary["author_tail"]
    split_distribution = summary["split_distribution"]
    author_holdout_split_distribution = summary["author_holdout_split_distribution"]
    recovery = summary["long_poem_recovery"]
    author_holdout = summary["author_holdout_diagnostics"]
    weight_stats = summary["text_split_weight_stats"]
    removed_symbol_lines = "\n".join(
        f"| `{symbol}` | {count} |"
        for symbol, count in before_after["top_removed_symbols"][:15]
    )

    return f"""# 唐诗数据清洗与偏见分析报告

## 1. 数据来源

本项目使用 `data/tang/*.json` 中的唐诗数据。原始样本包含作者、标题、正文段落和 id。模型训练前的旧数据文件 `data/peot.txt` 只保留正文，因此本分析优先基于原始 JSON 文件完成。

## 2. 清洗规则

本次清洗采用以下规则：

1. 合并同一首诗的 `paragraphs` 字段。
2. 删除括号注释、书名号内容、方括号内容、花括号内容等编辑性说明。
3. 删除编号、横线、半角空格、全角空格和换行符。
4. 统一常见标点，例如英文逗号转中文逗号。
5. 仅保留中文字符和常见中文标点。
6. 过滤空文本、过短文本和重复文本；对过长诗优先按句界分片回收。

长度阈值：正文汉字数小于 {MIN_TEXT_LEN} 的样本会被排除。正文超过 {MAX_TEXT_LEN} 字时，脚本按 `。！？；` 的句界拆分为不超过上限的训练片段；单句本身超过上限时不做强制截断，而是保留在删除清单中。

繁简转换方式：`{summary["converter"]}`。

## 3. 清洗结果

| 指标 | 数值 |
|---|---:|
| 原始样本数 | {summary["raw_count"]} |
| 保留原诗记录数 | {summary["retained_parent_count"]} |
| 原诗记录保留比例 | {summary["retained_parent_ratio"]:.2%} |
| 最终训练样本数 | {summary["kept_count"]} |
| 删除样本/片段数 | {summary["dropped_count"]} |
| 重复文本数 | {summary["drop_reasons"].get("duplicate", 0)} |
| 过短文本数 | {summary["drop_reasons"].get("too_short", 0)} |
| 无法按句界回收的超长句数 | {summary["drop_reasons"].get("too_long_sentence", 0)} |
| 空文本数 | {summary["drop_reasons"].get("empty", 0)} |

### 长诗分片回收

| 指标 | 数值 |
|---|---:|
| 原始超长诗记录数 | {recovery["long_records"]} |
| 成功回收的超长诗记录数 | {recovery["recovered_long_records"]} |
| 新增的长诗训练分片数 | {recovery["recovered_chunks"]} |
| 无法按句界回收的超长句数 | {recovery["unrecoverable_long_sentences"]} |
| 过短尾部分片数 | {recovery["discarded_short_chunks"]} |

长诗的所有分片继承同一个 `parent_id`，因此不会被拆到不同的训练、验证或测试集合中。

### 推荐数据划分

清洗后数据按照文本哈希确定性划分为训练集、验证集和测试集。这样可以保证划分结果可复现，并降低重复样本导致的训练/测试泄漏风险。

| 划分 | 数量 |
|---|---:|
| train | {split_distribution.get("train", 0)} |
| valid | {split_distribution.get("valid", 0)} |
| test | {split_distribution.get("test", 0)} |

### 作者级 Holdout 划分

该划分将同一已知作者的全部诗歌固定放入同一个集合，用于评估模型对未见作者风格的泛化；不详/无名作者仍按原诗 id 哈希，避免把不相关作者错误合并。

| 划分 | 样本数 |
|---|---:|
| train | {author_holdout_split_distribution.get("train", 0)} |
| valid | {author_holdout_split_distribution.get("valid", 0)} |
| test | {author_holdout_split_distribution.get("test", 0)} |

已知作者的 train-valid、train-test、valid-test 重叠数分别为 {author_holdout["train_valid_overlap"]}、{author_holdout["train_test_overlap"]}、{author_holdout["valid_test_overlap"]}。

## 4. 清洗前后对比

| 指标 | 原始数据 | 仅文本规范化后 | 最终保留数据 |
|---|---:|---:|---:|
| 平均汉字长度 | {before_after["raw_avg_len"]:.2f} | {before_after["candidate_avg_len"]:.2f} | {before_after["final_avg_len"]:.2f} |
| 中位汉字长度 | {before_after["raw_median_len"]:.2f} | {before_after["candidate_median_len"]:.2f} | {before_after["final_median_len"]:.2f} |
| 总汉字数 | {before_after["raw_total_chars"]} | {before_after["candidate_total_chars"]} | {before_after["final_total_chars"]} |
| 字符表大小 | {before_after["raw_vocab_size"]} | {before_after["candidate_vocab_size"]} | {before_after["final_vocab_size"]} |

被清洗规则移除的非正文符号类型数为 {before_after["removed_symbol_types"]}，总出现次数为 {before_after["removed_symbol_count"]}。高频被移除符号如下：

| 符号 | 次数 |
|---|---:|
{removed_symbol_lines}

结论：清洗不仅删除了异常符号，还降低了非正文内容对词表和训练目标的污染。长诗通过句界分片回收，因此最终数据同时保留了短篇主流样式和部分长篇诗歌信息。

## 5. 作者分布 Top 20

| 排名 | 作者 | 数量 | 占保留样本比例 |
|---:|---|---:|---:|
{author_lines}

结论：作者分布明显不均衡。高产作者在训练集中占据较大比例，模型更容易学习到这些作者的用字和题材风格，对低频作者的风格学习不足。

### 作者长尾指标

| 指标 | 数值 |
|---|---:|
| 作者总数 | {author_tail["author_count"]} |
| 仅出现 1 首诗的作者数 | {author_tail["single_poem_author_count"]} |
| 仅出现 1 首诗的作者占比 | {author_tail["single_poem_author_ratio"]:.2%} |
| Top 1 作者占比 | {author_tail["top1_ratio"]:.2%} |
| Top 5 作者占比 | {author_tail["top5_ratio"]:.2%} |
| Top 10 作者占比 | {author_tail["top10_ratio"]:.2%} |
| Top 20 作者占比 | {author_tail["top20_ratio"]:.2%} |
| 不详/无名作者样本占比 | {author_tail["unknown_author_ratio"]:.2%} |

这些指标显示数据存在典型长尾分布：少数作者贡献大量文本，大量作者只有极少样本。若模型训练不加控制，评估结果会更接近头部作者风格。

### 作者再平衡采样

训练集为每条样本输出了温和的作者权重：`1 / sqrt(作者在训练集中的样本数)`，并裁剪到 [{MIN_AUTHOR_WEIGHT}, {MAX_AUTHOR_WEIGHT}] 后归一化。该方式避免了直接使用 `1 / count` 时对单样本作者的极端重复采样。

| 指标 | 数值 |
|---|---:|
| 训练集作者数 | {weight_stats["train_author_count"]} |
| 加权训练样本数 | {weight_stats["train_sample_count"]} |
| 最小归一化样本权重 | {weight_stats["weight_min"]:.4f} |
| 最大归一化样本权重 | {weight_stats["weight_max"]:.4f} |
| 最大/最小权重比 | {weight_stats["weight_ratio"]:.2f} |

权重文件可由 `WeightedRandomSampler` 读取，并应与普通随机采样作为两组实验进行比较，而不是默认假定加权一定更优。

## 6. 长度分布

| 汉字长度区间 | 数量 |
|---|---:|
{length_lines}

结论：数据以中短篇诗歌为主。原先会被 `max_len` 截断的长诗现在优先被分片回收；仍无法按句界分片的超长句被明确记录，而非静默截断。

## 7. 体裁统计

| 类型 | 数量 |
|---|---:|
{form_lines}

说明：这里用每句汉字数粗略判断五言、七言和其他类型。该方法不是严格格律判断，但足够反映训练数据的长度结构偏向。

### 细体裁近似统计

| 细体裁 | 数量 |
|---|---:|
{fine_form_lines}

细体裁统计进一步显示，五言绝句、七言绝句、五言律诗和七言律诗是模型最容易学到的结构。长篇/排律样本较少，且更容易受到长度过滤影响。

## 8. 标题主题关键词统计

| 主题代理 | 标题命中数量 |
|---|---:|
{keyword_lines}

结论：标题关键词显示数据集中送别、山水、寺庙宗教、宫廷、季节等传统主题较多。模型生成结果也可能更偏向这些古典主题，而不擅长现代语义。

## 9. 数据偏见总结

1. 作者偏见：高产作者样本更多，模型容易偏向他们的表达方式。
2. 体裁偏见：五言、七言和中短篇诗歌占比较高，模型更容易生成固定长度句式。
3. 主题偏见：标题关键词集中在送别、山水、宫廷、宗教、季节等传统题材。
4. 语言偏见：语料来自古典诗歌，词汇和语法不适合现代汉语任务。
5. 长度偏见：中短篇诗歌占主导；长诗采用句界分片，仍需说明该处理会减少跨片上下文。

## 10. 对模型训练的影响

清洗可以减少注释、乱码、异常符号和截断文本对模型的干扰。数据处理还提供了两种可验证干预：长诗分片回收，以及基于温和作者权重的再平衡采样。模型实验应分别报告文本级切分和作者级 holdout 切分结果，避免将“记住已见作者风格”误解为“对未见作者泛化”。

## 11. 清洗策略的副作用

1. 长诗分片避免了截断，但会削弱跨分片的长程上下文。
2. 繁简转换能降低词表稀疏性，但会损失部分异体字和版本学信息。
3. 删除书名号、括号内容通常能去除注释，但也可能误删极少数正文信息。
4. 精确去重能降低训练/测试泄漏风险，但不同版本的近重复文本仍需更细的相似度检测。
5. 作者级 holdout 更严格，但与文本级切分的困惑度不能直接横向比较，因为测试分布不同。
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convert, converter_name = get_converter()
    raw_records = load_raw_records(RAW_DIR)

    seen_text = set()
    kept = []
    dropped = []
    drop_reasons = Counter()
    cleaned_candidates = []
    recovery_stats = {
        "long_records": 0,
        "recovered_long_records": 0,
        "recovered_chunks": 0,
        "unrecoverable_long_sentences": 0,
        "discarded_short_chunks": 0,
    }

    for record in raw_records:
        cleaned = clean_text(record["raw_text"], convert)
        cleaned_candidates.append(cleaned)
        length = content_length(cleaned)
        abnormal = has_abnormal_symbols(record["raw_text"], cleaned)

        if not cleaned:
            row = build_sample_row(record, cleaned, "original", 0, 1, length, abnormal)
            row["drop_reason"] = "empty"
            drop_reasons["empty"] += 1
            dropped.append(row)
        elif length < MIN_TEXT_LEN:
            row = build_sample_row(record, cleaned, "original", 0, 1, length, abnormal)
            row["drop_reason"] = "too_short"
            drop_reasons["too_short"] += 1
            dropped.append(row)
        elif length > MAX_TEXT_LEN:
            recovery_stats["long_records"] += 1
            chunks, unsplittable_sentences = split_long_poem(cleaned)
            recovery_stats["unrecoverable_long_sentences"] += len(unsplittable_sentences)
            accepted_chunks = 0

            for chunk_index, chunk in enumerate(chunks, start=1):
                row = build_sample_row(
                    record,
                    chunk,
                    "long_poem_chunk",
                    chunk_index,
                    len(chunks),
                    length,
                    abnormal,
                )
                if row["char_len"] < MIN_TEXT_LEN:
                    row["drop_reason"] = "too_short_chunk"
                    drop_reasons["too_short_chunk"] += 1
                    recovery_stats["discarded_short_chunks"] += 1
                    dropped.append(row)
                elif chunk in seen_text:
                    row["drop_reason"] = "duplicate_chunk"
                    drop_reasons["duplicate_chunk"] += 1
                    dropped.append(row)
                else:
                    seen_text.add(chunk)
                    kept.append(row)
                    accepted_chunks += 1

            for sentence in unsplittable_sentences:
                row = build_sample_row(
                    record,
                    sentence,
                    "unrecoverable_long_sentence",
                    0,
                    len(chunks),
                    length,
                    abnormal,
                )
                row["drop_reason"] = "too_long_sentence"
                drop_reasons["too_long_sentence"] += 1
                dropped.append(row)

            if accepted_chunks:
                recovery_stats["recovered_long_records"] += 1
                recovery_stats["recovered_chunks"] += accepted_chunks
            elif not unsplittable_sentences:
                row = build_sample_row(
                    record,
                    cleaned,
                    "unrecoverable_long_record",
                    0,
                    0,
                    length,
                    abnormal,
                )
                row["drop_reason"] = "too_long_unrecoverable"
                drop_reasons["too_long_unrecoverable"] += 1
                dropped.append(row)
        else:
            row = build_sample_row(record, cleaned, "original", 0, 1, length, abnormal)
            if cleaned in seen_text:
                row["drop_reason"] = "duplicate"
                drop_reasons["duplicate"] += 1
                dropped.append(row)
            else:
                seen_text.add(cleaned)
                kept.append(row)

    recovery_stats["retained_parent_records"] = len({row["parent_id"] for row in kept})
    recovery_stats["retained_training_samples"] = len(kept)

    author_counter = Counter(row["author"] for row in kept)
    length_counter = Counter(bucket_length(row["char_len"]) for row in kept)
    form_counter = Counter(row["form"] for row in kept)
    fine_form_counter = Counter(row["fine_form"] for row in kept)
    split_counter = Counter(row["split"] for row in kept)
    author_split_counter = Counter(row["author_split"] for row in kept)
    char_counter = Counter(ch for row in kept for ch in chinese_chars(row["cleaned_text"]))
    keyword_counter = keyword_stats(kept)
    before_after = build_before_after_stats(raw_records, cleaned_candidates, kept)
    author_tail = build_author_tail_stats(author_counter, len(kept))
    author_holdout_stats = author_holdout_diagnostics(kept)
    text_author_weights, text_sample_weights, text_weight_stats = build_author_weights(
        kept,
        "split",
    )
    holdout_author_weights, holdout_sample_weights, holdout_weight_stats = build_author_weights(
        kept,
        "author_split",
    )
    create_visualizations(author_counter, length_counter, fine_form_counter, before_after)

    with CLEAN_TEXT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(row["cleaned_text"] for row in kept))
    write_split_files(
        kept,
        "split",
        TRAIN_TEXT_PATH,
        VALID_TEXT_PATH,
        TEST_TEXT_PATH,
    )
    write_split_files(
        kept,
        "author_split",
        AUTHOR_HOLDOUT_TRAIN_TEXT_PATH,
        AUTHOR_HOLDOUT_VALID_TEXT_PATH,
        AUTHOR_HOLDOUT_TEST_TEXT_PATH,
    )

    metadata_fields = [
        "source_file",
        "id",
        "parent_id",
        "training_id",
        "author",
        "title",
        "source_kind",
        "chunk_index",
        "chunk_count",
        "original_char_len",
        "char_len",
        "sentence_count",
        "form",
        "fine_form",
        "split",
        "author_split",
        "abnormal_symbols_removed",
        "cleaned_text",
    ]
    write_csv(METADATA_PATH, metadata_fields, [{key: row[key] for key in metadata_fields} for row in kept])

    dropped_fields = metadata_fields + ["drop_reason"]
    write_csv(DROPPED_PATH, dropped_fields, [{key: row.get(key, "") for key in dropped_fields} for row in dropped])

    write_csv(
        OUT_DIR / "author_top20.csv",
        ["rank", "author", "count", "ratio"],
        [
            {
                "rank": idx + 1,
                "author": author,
                "count": count,
                "ratio": f"{count / max(len(kept), 1):.6f}",
            }
            for idx, (author, count) in enumerate(author_counter.most_common(20))
        ],
    )
    write_csv(
        OUT_DIR / "length_distribution.csv",
        ["bucket", "count"],
        [{"bucket": bucket, "count": length_counter[bucket]} for bucket in LENGTH_BUCKETS],
    )
    write_csv(
        OUT_DIR / "char_frequency_top100.csv",
        ["rank", "char", "count"],
        [{"rank": idx + 1, "char": ch, "count": count} for idx, (ch, count) in enumerate(char_counter.most_common(100))],
    )
    write_csv(
        OUT_DIR / "form_distribution.csv",
        ["form", "count"],
        [{"form": form, "count": count} for form, count in form_counter.most_common()],
    )
    write_csv(
        OUT_DIR / "fine_form_distribution.csv",
        ["fine_form", "count"],
        [{"fine_form": form, "count": count} for form, count in fine_form_counter.most_common()],
    )
    write_csv(
        OUT_DIR / "split_distribution.csv",
        ["split", "count"],
        [{"split": split, "count": split_counter[split]} for split in ["train", "valid", "test"]],
    )
    write_csv(
        OUT_DIR / "author_holdout_split_distribution.csv",
        ["split", "count"],
        [
            {"split": split, "count": author_split_counter[split]}
            for split in ["train", "valid", "test"]
        ],
    )
    write_csv(
        OUT_DIR / "long_poem_recovery.csv",
        ["metric", "value"],
        [{"metric": metric, "value": value} for metric, value in recovery_stats.items()],
    )
    write_csv(
        OUT_DIR / "author_holdout_diagnostics.csv",
        ["metric", "value"],
        [{"metric": metric, "value": value} for metric, value in author_holdout_stats.items()],
    )
    write_csv(
        AUTHOR_WEIGHT_PATH,
        ["author", "train_sample_count", "raw_weight", "clipped_weight", "sample_weight"],
        text_author_weights,
    )
    write_csv(
        TRAIN_SAMPLE_WEIGHT_PATH,
        ["sample_index", "training_id", "parent_id", "author", "sample_weight"],
        text_sample_weights,
    )
    write_csv(
        AUTHOR_HOLDOUT_WEIGHT_PATH,
        ["author", "train_sample_count", "raw_weight", "clipped_weight", "sample_weight"],
        holdout_author_weights,
    )
    write_csv(
        AUTHOR_HOLDOUT_SAMPLE_WEIGHT_PATH,
        ["sample_index", "training_id", "parent_id", "author", "sample_weight"],
        holdout_sample_weights,
    )
    write_csv(
        OUT_DIR / "author_tail_stats.csv",
        ["metric", "value"],
        [{"metric": metric, "value": value} for metric, value in author_tail.items()],
    )
    write_csv(
        OUT_DIR / "cleaning_before_after.csv",
        ["metric", "value"],
        [
            {"metric": metric, "value": value}
            for metric, value in before_after.items()
            if metric != "top_removed_symbols"
        ],
    )
    write_csv(
        OUT_DIR / "removed_symbols_top30.csv",
        ["rank", "symbol", "count"],
        [
            {"rank": idx + 1, "symbol": symbol, "count": count}
            for idx, (symbol, count) in enumerate(before_after["top_removed_symbols"])
        ],
    )
    write_csv(
        OUT_DIR / "title_keyword_distribution.csv",
        ["keyword_group", "count"],
        [{"keyword_group": name, "count": count} for name, count in keyword_counter.most_common()],
    )

    author_top20 = [
        (author, count, count / max(len(kept), 1))
        for author, count in author_counter.most_common(20)
    ]
    summary = {
        "converter": converter_name,
        "raw_count": len(raw_records),
        "retained_parent_count": recovery_stats["retained_parent_records"],
        "retained_parent_ratio": recovery_stats["retained_parent_records"] / max(len(raw_records), 1),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "kept_ratio": recovery_stats["retained_parent_records"] / max(len(raw_records), 1),
        "drop_reasons": dict(drop_reasons),
        "long_poem_recovery": recovery_stats,
        "author_top10_ratio": sum(count for _, count in author_counter.most_common(10)) / max(len(kept), 1),
        "vocab_size": len(char_counter),
        "top_chars": char_counter.most_common(30),
        "form_distribution": dict(form_counter),
        "fine_form_distribution": dict(fine_form_counter),
        "split_distribution": dict(split_counter),
        "author_holdout_split_distribution": dict(author_split_counter),
        "author_holdout_diagnostics": author_holdout_stats,
        "text_split_weight_stats": text_weight_stats,
        "author_holdout_weight_stats": holdout_weight_stats,
        "before_after": before_after,
        "author_tail": author_tail,
        "figure_dir": str(FIGURE_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        build_report(summary, author_top20, length_counter, form_counter, fine_form_counter, keyword_counter),
        encoding="utf-8",
    )

    print(f"raw_count={summary['raw_count']}")
    print(f"kept_count={summary['kept_count']}")
    print(f"dropped_count={summary['dropped_count']}")
    print(f"converter={summary['converter']}")
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    if summary["converter"] == "none":
        print("warning=OpenCC is not installed. Text cleaning finished, but traditional-to-simplified conversion was skipped.")
        print("hint=Install opencc-python-reimplemented and rerun this script before final training/report submission.")


if __name__ == "__main__":
    main()
