"""tests/test_redis_cache_flow.py — 缓存 Key 生成 + get_or_compute 命中/未命中/占位符流程测试

不依赖真实 Redis / LLM / Milvus:
  - 若运行环境未安装 redis 包, 注入一个最小 fake redis 模块, 使 redis_db 可导入;
  - 所有 Redis 读写/锁操作均通过 unittest.mock 替换, 不触发真实网络.

运行方式:
    python -m unittest tests.test_redis_cache_flow -v
"""
import hashlib
import importlib.util
import sys
import types
import unittest
from unittest import mock

from cache_strategy import (
    INDEPENDENT_STREAM_PREFIX,
    INDEPENDENT_CACHE_PREFIX,
    normalize_query,
)


def _install_fake_redis_if_needed():
    """redis 包未安装时, 注入最小 fake redis 模块 (仅覆盖 redis_db 用到的符号)."""
    if importlib.util.find_spec("redis") is not None:
        return
    fake = types.ModuleType("redis")

    class RedisError(Exception):
        pass

    class ConnectionError(Exception):
        pass

    class ConnectionPool:
        def __init__(self, **kwargs):
            pass

    class StrictRedis:
        def __init__(self, connection_pool=None, **kwargs):
            self.connection_pool = connection_pool

        def register_script(self, script):
            def _run(keys=None, args=None):
                return 1
            return _run

        def ping(self):
            return True

        def get(self, key):
            return None

        def set(self, *args, **kwargs):
            return True

        def setex(self, *args, **kwargs):
            return True

        def delete(self, *args):
            return 1

    fake.RedisError = RedisError
    fake.ConnectionError = ConnectionError
    fake.ConnectionPool = ConnectionPool
    fake.StrictRedis = StrictRedis
    sys.modules["redis"] = fake


_install_fake_redis_if_needed()

from redis_db import redis_manager  # noqa: E402


def _md5_key(text, prefix):
    """与 redis_db._generate_key 一致的 Key 生成公式 (独立复算, 用于交叉校验)."""
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


class TestIndependentKeyGeneration(unittest.TestCase):
    def test_stream_key_matches_expected(self):
        q = "什么是高血压？"
        key = redis_manager._generate_key(normalize_query(q), prefix=INDEPENDENT_STREAM_PREFIX)
        self.assertEqual(key, _md5_key(normalize_query(q), INDEPENDENT_STREAM_PREFIX))

    def test_cache_key_uses_v2_prefix(self):
        key = redis_manager._generate_key(normalize_query("高血压怎么治疗"), prefix=INDEPENDENT_CACHE_PREFIX)
        self.assertTrue(key.startswith("llm:cache:v2:"))
        self.assertNotIn("ctx:", key)
        self.assertNotIn("user:", key)

    def test_different_questions_different_key(self):
        k1 = redis_manager._generate_key(normalize_query("什么是高血压"), prefix=INDEPENDENT_STREAM_PREFIX)
        k2 = redis_manager._generate_key(normalize_query("什么是糖尿病"), prefix=INDEPENDENT_STREAM_PREFIX)
        self.assertNotEqual(k1, k2)

    def test_whitespace_variants_same_key(self):
        # 仅首尾空白不同, 规范化后应得到同一 Key
        k1 = redis_manager._generate_key(normalize_query("什么是高血压"), prefix=INDEPENDENT_STREAM_PREFIX)
        k2 = redis_manager._generate_key(normalize_query("  什么是高血压  "), prefix=INDEPENDENT_STREAM_PREFIX)
        self.assertEqual(k1, k2)


class TestGetOrComputeFlow(unittest.TestCase):
    """验证 redis_manager.get_or_compute 的命中/未命中/占位符行为未被破坏."""

    def test_second_call_hits_cache(self):
        store = {}

        def fake_get_answer(q):
            return store.get(redis_manager._generate_key(q))

        def fake_set_answer(q, a, expire=3600):
            store[redis_manager._generate_key(q)] = a

        compute_calls = {"n": 0}

        def compute():
            compute_calls["n"] += 1
            return "答案A"

        with mock.patch.object(redis_manager, "get_answer", side_effect=fake_get_answer), \
             mock.patch.object(redis_manager, "set_answer", side_effect=fake_set_answer), \
             mock.patch.object(redis_manager, "acquire_lock", return_value="token-1"), \
             mock.patch.object(redis_manager, "release_lock", return_value=True):
            r1 = redis_manager.get_or_compute("问题", compute)
            r2 = redis_manager.get_or_compute("问题", compute)

        self.assertEqual(r1, "答案A")
        self.assertEqual(r2, "答案A")
        self.assertEqual(compute_calls["n"], 1)  # 第二次命中缓存, 不再 compute

    def test_miss_runs_compute(self):
        compute_calls = {"n": 0}

        def compute():
            compute_calls["n"] += 1
            return "答案B"

        with mock.patch.object(redis_manager, "get_answer", return_value=None), \
             mock.patch.object(redis_manager, "acquire_lock", return_value="token-2"), \
             mock.patch.object(redis_manager, "release_lock", return_value=True), \
             mock.patch.object(redis_manager, "set_answer") as mock_set:
            result = redis_manager.get_or_compute("问题", compute)

        self.assertEqual(result, "答案B")
        self.assertEqual(compute_calls["n"], 1)
        mock_set.assert_called_once()  # 有效答案写入缓存

    def test_empty_answer_writes_placeholder_not_answer(self):
        with mock.patch.object(redis_manager, "get_answer", return_value=None), \
             mock.patch.object(redis_manager, "acquire_lock", return_value="token-3"), \
             mock.patch.object(redis_manager, "release_lock", return_value=True), \
             mock.patch.object(redis_manager, "set_answer") as mock_set, \
             mock.patch.object(redis_manager.client, "setex") as mock_setex:
            result = redis_manager.get_or_compute("问题", lambda: "")

        self.assertEqual(result, "")
        mock_set.assert_not_called()       # 空答案不作为成功答案缓存
        mock_setex.assert_called_once()    # 写入 <EMPTY> 占位符
        args, _ = mock_setex.call_args
        self.assertEqual(args[2], "<EMPTY>")

    def test_lock_fail_fallback_not_cached_as_answer(self):
        with mock.patch.object(redis_manager, "get_answer", return_value=None), \
             mock.patch.object(redis_manager, "acquire_lock", return_value=None), \
             mock.patch.object(redis_manager, "set_answer") as mock_set:
            result = redis_manager.get_or_compute("问题", lambda: "不应被调用")
        self.assertIn(result, (None, "System busy, calculating..."))
        mock_set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
