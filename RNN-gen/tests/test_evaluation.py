import math

import pytest
import torch
from torch import nn

from evaluation import (
    ACROSTIC_PROMPTS,
    CONTINUATION_PROMPTS,
    GENERATION_SEEDS,
    PoetryGenerator,
    acrostic_accuracy,
    acrostic_completion_rate,
    calculate_generation_metrics,
    distinct_n,
    generate_fixed_samples,
    mean_repetition_rate,
    normalize_generated_suffix,
)
from experiment_core import target_mask, token_cross_entropy


def test_distinct_n_uses_corpus_level_ngrams():
    poems = ["春风春风", "春风明月"]

    assert distinct_n(poems, 1) == pytest.approx(4 / 8)
    assert distinct_n(poems, 2) == pytest.approx(4 / 6)


def test_mean_repetition_rate_averages_per_poem_bigram_repetition():
    poems = ["哈哈哈哈", "春风明月"]

    assert mean_repetition_rate(poems, n=2) == pytest.approx((2 / 3 + 0) / 2)


def test_acrostic_accuracy_checks_sentence_heads():
    outputs = [
        {"prompt": "春风", "text": "春山如画。风月无边。"},
        {"prompt": "明月", "text": "明河在天。风吹古木。"},
    ]

    assert acrostic_accuracy(outputs) == pytest.approx(3 / 4)


def test_empty_generation_metrics_are_zero():
    assert distinct_n([], 1) == 0.0
    assert mean_repetition_rate([], 2) == 0.0
    assert acrostic_accuracy([]) == 0.0
    assert acrostic_completion_rate([]) == 0.0


def test_generation_metrics_use_normalized_suffix_instead_of_prompt():
    records = [
        {"task": "continuation", "prompt": "春风", "text": "春风哈哈哈哈"},
        {"task": "continuation", "prompt": "明月", "text": "明月天地玄黄"},
    ]

    metrics = calculate_generation_metrics(records)

    assert metrics["distinct_1"] == pytest.approx(5 / 8)
    assert metrics["distinct_2"] == pytest.approx(4 / 6)
    assert metrics["repetition_rate"] == pytest.approx((2 / 3 + 0) / 2)


def test_generated_suffix_normalization_removes_whitespace_and_punctuation():
    assert normalize_generated_suffix("春风", "春风，明 月！") == "明月"


def test_acrostic_completion_rate_counts_missing_sentence_heads():
    outputs = [{"prompt": "春风", "text": "春山如画。"}]

    assert acrostic_accuracy(outputs) == pytest.approx(1 / 2)
    assert acrostic_completion_rate(outputs) == pytest.approx(1 / 2)


def test_token_cross_entropy_ignores_padding_and_is_token_weighted():
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [2.0, 2.0]],
            [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]],
        ]
    )
    targets = torch.tensor([[0, 1, 0], [1, 0, 0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])

    loss_sum, token_count = token_cross_entropy(logits, targets, mask)
    expected = 4 * math.log1p(math.exp(-4))

    assert token_count == 4
    assert loss_sum.item() == pytest.approx(expected, rel=1e-5)


def test_target_mask_includes_eop_but_excludes_following_padding():
    targets = torch.tensor([[5, 6, 2, 0, 0]])

    assert target_mask(targets).tolist() == [[True, True, True, False, False]]


class _PunctuationModel(nn.Module):
    def __init__(self, vocab_size, punctuation_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.punctuation_id = punctuation_id

    def forward(self, input_ids, hidden=None):
        logits = torch.full(
            (input_ids.numel(), self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[:, self.punctuation_id] = 100.0
        return logits, None


def test_poetry_generator_never_emits_control_tokens():
    word2idx = {"PAD": 0, "SOP": 1, "EOP": 2, "UNK": 3, "春": 4, "风": 5, "。": 6}
    idx2word = {index: word for word, index in word2idx.items()}
    generator = PoetryGenerator(
        model=_PunctuationModel(len(word2idx), word2idx["。"]),
        word2idx=word2idx,
        idx2word=idx2word,
        device=torch.device("cpu"),
        temperature=0.8,
        top_p=0.9,
        max_gen_len=8,
    )

    continuation = generator.generate_continuation("春")
    acrostic = generator.generate_acrostic("春风")

    assert all(token not in continuation for token in ("PAD", "SOP", "EOP", "UNK"))
    assert acrostic.startswith("春。风")


def test_fixed_sample_generation_has_three_seeds_per_prompt():
    word2idx = {"PAD": 0, "SOP": 1, "EOP": 2, "UNK": 3, "。": 4}
    for character in set("".join(CONTINUATION_PROMPTS + ACROSTIC_PROMPTS)):
        word2idx.setdefault(character, len(word2idx))
    idx2word = {index: word for word, index in word2idx.items()}
    generator = PoetryGenerator(
        model=_PunctuationModel(len(word2idx), word2idx["。"]),
        word2idx=word2idx,
        idx2word=idx2word,
        device=torch.device("cpu"),
        temperature=0.8,
        top_p=0.9,
        max_gen_len=12,
    )

    records = generate_fixed_samples(generator)

    assert len(records) == (len(CONTINUATION_PROMPTS) + len(ACROSTIC_PROMPTS)) * len(GENERATION_SEEDS)
    assert {record["seed"] for record in records} == set(GENERATION_SEEDS)
