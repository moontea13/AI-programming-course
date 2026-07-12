import re

import torch
import torch.nn.functional as F


CONTINUATION_PROMPTS = (
    "丽日照残春",
    "春风得意马蹄疾",
    "大漠孤烟直",
    "两个黄鹂鸣翠柳",
    "床前明月光",
)
ACROSTIC_PROMPTS = ("深度学习", "人工智能", "春暖花开", "千古风流")
GENERATION_SEEDS = (123, 456, 789)
IGNORED_GENERATION_CHARACTERS = re.compile(r"[\s，。！？；：、,.!?;:]+")


def _ngrams(text, n):
    return [text[index:index + n] for index in range(max(len(text) - n + 1, 0))]


def distinct_n(poems, n):
    ngrams = [gram for poem in poems for gram in _ngrams(poem, n)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def mean_repetition_rate(poems, n=2):
    if not poems:
        return 0.0
    rates = []
    for poem in poems:
        ngrams = _ngrams(poem, n)
        repeated = len(ngrams) - len(set(ngrams))
        rates.append(repeated / len(ngrams) if ngrams else 0.0)
    return sum(rates) / len(rates)


def _sentence_heads(text):
    return [part[0] for part in re.split(r"[。！!?？]", text) if part]


def acrostic_accuracy(outputs):
    correct = 0
    total = 0
    for output in outputs:
        prompt = output["prompt"]
        heads = _sentence_heads(output["text"])
        total += len(prompt)
        correct += sum(actual == expected for actual, expected in zip(heads, prompt))
    return correct / total if total else 0.0


def acrostic_completion_rate(outputs):
    completed = 0
    total = 0
    for output in outputs:
        prompt = output["prompt"]
        total += len(prompt)
        completed += min(len(_sentence_heads(output["text"])), len(prompt))
    return completed / total if total else 0.0


def normalize_generated_suffix(prompt, text):
    suffix = text[len(prompt):] if text.startswith(prompt) else text
    return IGNORED_GENERATION_CHARACTERS.sub("", suffix)


class PoetryGenerator:
    def __init__(
        self,
        *,
        model,
        word2idx,
        idx2word,
        device,
        temperature,
        top_p,
        max_gen_len,
    ):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.banned_ids = {
            word2idx[token]
            for token in ("PAD", "SOP", "UNK")
            if token in word2idx
        }

    def _next_token(self, characters):
        input_tokens = ["SOP"] + characters
        input_ids = [
            self.word2idx.get(token, self.word2idx["UNK"])
            for token in input_tokens
        ]
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(input_tensor)
        logits = output[-1].clone() / self.temperature
        for token_id in self.banned_ids:
            logits[token_id] = float("-inf")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > self.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_indices[remove]] = float("-inf")
        probabilities = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probabilities, 1).item()
        return self.idx2word[token_id]

    def generate_continuation(self, prompt):
        characters = list(prompt)
        for _ in range(self.max_gen_len):
            token = self._next_token(characters)
            if token == "EOP":
                break
            characters.append(token)
        return "".join(characters)

    def generate_acrostic(self, prompt):
        characters = []
        prompt_index = 0
        sentence_start = True
        for _ in range(self.max_gen_len):
            if sentence_start:
                if prompt_index >= len(prompt):
                    break
                characters.append(prompt[prompt_index])
                prompt_index += 1
                sentence_start = False
                continue
            token = self._next_token(characters)
            if token == "EOP":
                break
            characters.append(token)
            sentence_start = token in {"。", "！", "!", "？", "?"}
        return "".join(characters)


def generate_fixed_samples(generator):
    records = []
    for seed in GENERATION_SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for prompt in CONTINUATION_PROMPTS:
            records.append(
                {
                    "task": "continuation",
                    "prompt": prompt,
                    "seed": seed,
                    "text": generator.generate_continuation(prompt),
                }
            )
        for prompt in ACROSTIC_PROMPTS:
            records.append(
                {
                    "task": "acrostic",
                    "prompt": prompt,
                    "seed": seed,
                    "text": generator.generate_acrostic(prompt),
                }
            )
    return records


def calculate_generation_metrics(records):
    continuations = [
        normalize_generated_suffix(record["prompt"], record["text"])
        for record in records
        if record["task"] == "continuation"
    ]
    acrostics = [record for record in records if record["task"] == "acrostic"]
    return {
        "distinct_1": distinct_n(continuations, 1),
        "distinct_2": distinct_n(continuations, 2),
        "repetition_rate": mean_repetition_rate(continuations, 2),
        "acrostic_accuracy": acrostic_accuracy(acrostics),
        "acrostic_completion_rate": acrostic_completion_rate(acrostics),
    }
