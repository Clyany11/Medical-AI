"""tests/test_cache_strategy.py — 缓存策略分类器纯单元测试 (无 Redis / LLM / Milvus 依赖)

运行方式:
    python -m unittest tests.test_cache_strategy -v
    # 或
    python -m pytest tests/test_cache_strategy.py -v
"""
import unittest

from cache_strategy import (
    CacheStrategy,
    normalize_query,
    classify_cache_strategy,
    decide_cache_query,
)


class TestNormalizeQuery(unittest.TestCase):
    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(normalize_query("  高血压怎么治疗  "), "高血压怎么治疗")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(normalize_query("高血压  怎么   治疗"), "高血压 怎么 治疗")

    def test_collapses_fullwidth_space(self):
        self.assertEqual(normalize_query("高血压\u3000怎么治疗"), "高血压 怎么治疗")

    def test_preserves_punctuation_and_terms(self):
        # 不归并药名/否定词等语义
        self.assertEqual(normalize_query("阿莫西林是什么药？"), "阿莫西林是什么药？")
        self.assertEqual(normalize_query("不是高血压"), "不是高血压")

    def test_empty_and_none(self):
        self.assertEqual(normalize_query(""), "")
        self.assertEqual(normalize_query(None), "")


class TestClassifyIndependent(unittest.TestCase):
    def test_general_knowledge_questions(self):
        for q in [
            "阿莫西林是什么药？",
            "什么是高血压？",
            "二甲双胍缓释片和普通片有什么区别？",
            "高血压怎么治疗",
            "高血压能治愈吗",
        ]:
            self.assertEqual(
                classify_cache_strategy(q, ""), CacheStrategy.INDEPENDENT, q
            )


class TestClassifyContextual(unittest.TestCase):
    def test_anaphora_and_ellipsis(self):
        for q in [
            "它有什么副作用？",
            "那我应该怎么吃？",
            "刚才说的第二种药呢？",
            "这个和上一个有什么区别？",
        ]:
            self.assertEqual(classify_cache_strategy(q, ""), CacheStrategy.CONTEXTUAL, q)

    def test_other_is_not_anaphora(self):
        # "其他" 不构成回指, 不应判为 CONTEXTUAL
        self.assertNotEqual(classify_cache_strategy("其他降压药有哪些", ""), CacheStrategy.CONTEXTUAL)


class TestClassifySensitive(unittest.TestCase):
    def test_individualized_medical(self):
        for q in [
            "阿莫西林适合我吗？",
            "我应该服用多少剂量？",
            "我胃疼吃什么药",
            "我头疼",
        ]:
            self.assertEqual(
                classify_cache_strategy(q, ""), CacheStrategy.SENSITIVE_OR_UNCERTAIN, q
            )

    def test_individualized_with_pronoun_not_independent(self):
        # 含"这个"+"我"+"肾功能"的组合, 无论归 CONTEXTUAL 还是 SENSITIVE, 都不得判为 INDEPENDENT
        self.assertNotEqual(
            classify_cache_strategy("根据我的肾功能，我能服用这个药吗？", ""),
            CacheStrategy.INDEPENDENT,
        )


class TestClassifyEdgeCases(unittest.TestCase):
    def test_empty_query(self):
        self.assertEqual(classify_cache_strategy("", ""), CacheStrategy.SENSITIVE_OR_UNCERTAIN)
        self.assertEqual(classify_cache_strategy("   ", ""), CacheStrategy.SENSITIVE_OR_UNCERTAIN)
        self.assertEqual(classify_cache_strategy(None, ""), CacheStrategy.SENSITIVE_OR_UNCERTAIN)

    def test_bare_noun_phrase_is_conservative(self):
        # 无问句标记的裸名词短语 → 保守跳过
        self.assertEqual(classify_cache_strategy("高血压", ""), CacheStrategy.SENSITIVE_OR_UNCERTAIN)


class TestDecideCacheQuery(unittest.TestCase):
    def test_independent_key_stable_across_history(self):
        q = "什么是高血压？"
        _, key1 = decide_cache_query(q, "")
        _, key2 = decide_cache_query(q, "用户: 我胃疼\n助手: 建议就医")
        self.assertEqual(key1, key2)
        self.assertEqual(key1, normalize_query(q))

    def test_independent_key_not_empty(self):
        _, key = decide_cache_query("高血压怎么治疗", "")
        self.assertEqual(key, "高血压怎么治疗")

    def test_contextual_skip_cache(self):
        _, key = decide_cache_query("它有什么副作用？", "用户: 阿莫西林怎么吃")
        self.assertEqual(key, "")

    def test_sensitive_skip_cache(self):
        _, key = decide_cache_query("阿莫西林适合我吗？", "")
        self.assertEqual(key, "")

    def test_different_independent_questions_different_key_text(self):
        _, k1 = decide_cache_query("什么是高血压", "")
        _, k2 = decide_cache_query("什么是糖尿病", "")
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
