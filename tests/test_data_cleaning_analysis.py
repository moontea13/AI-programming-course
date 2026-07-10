import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data_cleaning_analysis import (
    assign_author_split,
    assign_split,
    build_author_weights,
    clean_text,
    classify_fine_form,
    content_length,
    is_chinese_char,
    split_long_poem,
)
from process import SPLIT_CACHE_VERSION, _cache_is_current


class DataCleaningTests(unittest.TestCase):
    def test_clean_text_removes_editorial_parentheses(self):
        result = clean_text("春晓（孟浩然）", lambda text: text)
        self.assertEqual(result, "春晓")

    def test_clean_text_normalizes_punctuation(self):
        result = clean_text("你好,世界.", lambda text: text)
        self.assertEqual(result, "你好，世界")

    def test_clean_text_removes_editorial_brackets(self):
        result = clean_text("琵琶行〖一〗〈校注〉〔节选〕", lambda text: text)
        self.assertEqual(result, "琵琶行")

    def test_extension_a_character_is_preserved(self):
        result = clean_text("䍦风䙰", lambda text: text)
        self.assertEqual(result, "䍦风䙰")
        self.assertTrue(is_chinese_char("䍦"))
        self.assertEqual(content_length(result), 3)

    def test_fine_form_classification(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        self.assertEqual(classify_fine_form(poem), "五言绝句")

    def test_split_assignment_is_deterministic(self):
        poem = "春眠不觉晓，处处闻啼鸟。"
        self.assertEqual(assign_split(poem), assign_split(poem))
        self.assertIn(assign_split(poem), {"train", "valid", "test"})

    def test_long_poem_is_split_on_sentence_boundaries(self):
        poem = "一二三四五。一二三四五。一二三四五。"
        chunks, unsplittable = split_long_poem(poem, max_len=10)
        self.assertEqual(chunks, ["一二三四五。一二三四五。", "一二三四五。"])
        self.assertEqual(unsplittable, [])

    def test_known_author_stays_in_one_holdout_split(self):
        first = assign_author_split("白居易", "poem-a")
        second = assign_author_split("白居易", "poem-b")
        self.assertEqual(first, second)

    def test_author_weights_are_tempered(self):
        rows = [
            {"author": "甲", "split": "train", "training_id": "a1", "parent_id": "a"},
            {"author": "甲", "split": "train", "training_id": "a2", "parent_id": "a"},
            {"author": "乙", "split": "train", "training_id": "b1", "parent_id": "b"},
        ]
        _, sample_rows, stats = build_author_weights(rows, "split")
        self.assertEqual(len(sample_rows), 3)
        self.assertLessEqual(stats["weight_ratio"], 10.0)

    def test_cache_invalidates_when_max_len_changes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "train.txt"
            cache_path = root / "cache.npz"
            source_path.write_text("春眠不觉晓\n", encoding="utf-8")
            np.savez_compressed(
                cache_path,
                max_len=np.array(125),
                cache_version=np.array(SPLIT_CACHE_VERSION),
            )
            self.assertTrue(_cache_is_current(str(cache_path), [str(source_path)], 125))
            self.assertFalse(_cache_is_current(str(cache_path), [str(source_path)], 100))


if __name__ == "__main__":
    unittest.main()
