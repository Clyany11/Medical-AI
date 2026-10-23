"""tests/test_reflection_flow.py — ReflectionAgent.correct() 接入问答流程的测试

覆盖:
  1. 初始评分 >= 阈值 → 不调用 correct()
  2. 初始评分 <  阈值 → 调用 correct()
  3. 修正后答案会再次评估
  4. 修正后分数提升 → 返回修正后的答案
  5. 修正后仍低分 → 不无限重试, 保守处理
  6. 评估异常 / JSON 解析失败 → 不把默认高分当真实结果
  7. 非流式流程不对同一答案重复启动后台评估
  8. fast 流式正常输出, 不修改已发送内容
  9. strict 流式只输出完成评估/修正后的最终答案
  10. 原有来源链接 / 免责声明 / 会话记忆不被破坏

全部使用 Mock LLM / Mock Redis (内存 MockMemory), 不依赖真实模型或外部服务.
运行方式:
    python -m unittest tests.test_reflection_flow -v
"""
import unittest
from unittest import mock

from agents.reflection_agent import ReflectionAgent, ReflectionResult
from agents.orchestrator import AgentOrchestrator
from agents.router_agent import RouteDecision


def _make_reflection(score, needs_correction, evaluation_ok=True, feedback="fb"):
    return ReflectionResult(
        score=score,
        accuracy=score,
        completeness=score,
        safety=score,
        relevance=score,
        feedback=feedback,
        needs_correction=needs_correction,
        duration_ms=0.0,
        evaluation_ok=evaluation_ok,
    )


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_llm(content):
    llm = mock.MagicMock()
    llm.chat.completions.create = mock.AsyncMock(return_value=_FakeResponse(content))
    return llm


class MockMemory:
    """内存版长期/短期记忆, 替代真实 Redis."""

    def __init__(self):
        self.saved = []
        self.turns = []

    async def save_reflection(self, user_id, query, score, feedback, extra=None):
        self.saved.append({
            "user_id": user_id, "query": query, "score": score,
            "feedback": feedback, "extra": extra,
        })

    def add_turn(self, session_id, role, content, metadata=None):
        self.turns.append((session_id, role, content))

    def get_context(self, session_id, last_n=5):
        return ""

    async def load_long_term(self, user_id, key):
        return None


def _make_orchestrator(stream_mode="fast", threshold=0.65):
    orch = object.__new__(AgentOrchestrator)
    orch.enable_reflection = True
    orch.stream_mode = stream_mode
    orch.milvus_manager = None
    orch.ancient_books_manager = None
    orch.neo4j_manager = None
    orch.non_stream_model = "test-model"
    orch.stream_model = "test-model"
    orch.reasoner_model = ""
    orch.memory = MockMemory()
    orch.reflection_agent = ReflectionAgent(
        llm_client=_make_llm("{}"), model="test", threshold=threshold
    )
    return orch


def _capture_task(created):
    """捕获 create_task 调度并关闭协程, 避免 'never awaited' 告警."""
    def _inner(coro):
        created.append(coro)
        coro.close()
    return _inner


def _medical_route():
    return RouteDecision(
        route="medical_rag", intent="medical",
        collection_type="modern_medicine", use_web_search=False,
        is_medical=True,
    )


class TestReflectionAgentEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_valid_json_ok(self):
        agent = ReflectionAgent(llm_client=_make_llm(
            '{"accuracy":0.9,"completeness":0.8,"safety":0.9,"relevance":0.8,"feedback":"ok"}'
        ))
        r = await agent.evaluate("q", "a")
        self.assertTrue(r.evaluation_ok)
        self.assertGreater(r.score, 0.0)
        self.assertFalse(r.needs_correction)

    async def test_invalid_json_not_default_high(self):
        agent = ReflectionAgent(llm_client=_make_llm("这不是JSON"))
        r = await agent.evaluate("q", "a")
        self.assertFalse(r.evaluation_ok)
        self.assertEqual(r.score, -1.0)
        self.assertFalse(r.needs_correction)

    async def test_out_of_range_score_invalid(self):
        agent = ReflectionAgent(llm_client=_make_llm(
            '{"accuracy":1.5,"completeness":0.8,"safety":0.9,"relevance":0.8}'
        ))
        r = await agent.evaluate("q", "a")
        self.assertFalse(r.evaluation_ok)
        self.assertEqual(r.score, -1.0)

    async def test_llm_exception_invalid(self):
        llm = mock.MagicMock()
        llm.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("boom"))
        agent = ReflectionAgent(llm_client=llm)
        r = await agent.evaluate("q", "a")
        self.assertFalse(r.evaluation_ok)
        self.assertEqual(r.score, -1.0)


class TestReflectionAgentCorrect(unittest.IsolatedAsyncioTestCase):
    async def test_correct_returns_new_answer(self):
        agent = ReflectionAgent(llm_client=_make_llm("修正后的答案"))
        ans, ok = await agent.correct("q", "原始", _make_reflection(0.5, True), context="ctx")
        self.assertTrue(ok)
        self.assertEqual(ans, "修正后的答案")

    async def test_correct_empty_falls_back(self):
        agent = ReflectionAgent(llm_client=_make_llm("   "))
        ans, ok = await agent.correct("q", "原始", _make_reflection(0.5, True))
        self.assertFalse(ok)
        self.assertEqual(ans, "原始")

    async def test_correct_exception_falls_back(self):
        llm = mock.MagicMock()
        llm.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("x"))
        agent = ReflectionAgent(llm_client=llm)
        ans, ok = await agent.correct("q", "原始", _make_reflection(0.5, True))
        self.assertFalse(ok)
        self.assertEqual(ans, "原始")


class TestEvaluateAndCorrect(unittest.IsolatedAsyncioTestCase):
    # 1
    async def test_high_score_no_correct(self):
        orch = _make_orchestrator()
        orch.reflection_agent.evaluate = mock.AsyncMock(return_value=_make_reflection(0.8, False))
        orch.reflection_agent.correct = mock.AsyncMock()
        ans, score, rec = await orch._evaluate_and_correct("q", "原始", "ctx")
        self.assertEqual(ans, "原始")
        self.assertEqual(score, 0.8)
        orch.reflection_agent.correct.assert_not_called()

    # 2 / 3 / 4
    async def test_low_score_corrects_and_reevaluates(self):
        orch = _make_orchestrator()
        orch.reflection_agent.evaluate = mock.AsyncMock(side_effect=[
            _make_reflection(0.5, True),
            _make_reflection(0.8, False),
        ])
        orch.reflection_agent.correct = mock.AsyncMock(return_value=("修正", True))
        ans, score, rec = await orch._evaluate_and_correct("q", "原始", "ctx")
        self.assertEqual(ans, "修正")                      # 4: 提升 → 返回修正后
        self.assertEqual(score, 0.8)
        orch.reflection_agent.correct.assert_awaited_once()  # 2: 触发修正
        self.assertEqual(orch.reflection_agent.evaluate.await_count, 2)  # 3: 复评

    # 5
    async def test_still_low_no_retry(self):
        orch = _make_orchestrator()
        orch.reflection_agent.evaluate = mock.AsyncMock(side_effect=[
            _make_reflection(0.5, True),
            _make_reflection(0.55, True),  # 复评仍低分
        ])
        orch.reflection_agent.correct = mock.AsyncMock(return_value=("修正", True))
        ans, score, rec = await orch._evaluate_and_correct("q", "原始", "ctx")
        self.assertEqual(orch.reflection_agent.correct.await_count, 1)  # 不无限重试
        self.assertEqual(orch.reflection_agent.evaluate.await_count, 2)
        self.assertEqual(score, 0.55)
        self.assertTrue(rec["failure_reason"])  # 诚实记录失败原因

    # 6
    async def test_eval_failure_no_correct(self):
        orch = _make_orchestrator()
        orch.reflection_agent.evaluate = mock.AsyncMock(
            return_value=_make_reflection(-1.0, False, evaluation_ok=False, feedback="解析失败")
        )
        orch.reflection_agent.correct = mock.AsyncMock()
        ans, score, rec = await orch._evaluate_and_correct("q", "原始", "ctx")
        self.assertEqual(ans, "原始")
        self.assertEqual(score, -1.0)
        self.assertFalse(rec["evaluation_ok"])
        orch.reflection_agent.correct.assert_not_called()


class TestNonStreamNoDoubleEval(unittest.IsolatedAsyncioTestCase):
    # 7
    async def test_no_background_duplicate_eval(self):
        orch = _make_orchestrator(stream_mode="fast")
        orch.router = mock.MagicMock()
        orch.router.route = mock.AsyncMock(return_value=_medical_route())
        orch.medical_agent = mock.MagicMock()
        orch.medical_agent.generate = mock.AsyncMock(return_value=("原始答案", {}))
        orch._build_history_with_memory = mock.AsyncMock(return_value="")
        orch._evaluate_and_correct = mock.AsyncMock(
            return_value=("最终答案", 0.8, {"initial_feedback": "fb"})
        )
        orch._persist_reflection = mock.AsyncMock()

        import agents.orchestrator as orch_mod
        created = []
        with mock.patch.object(
            orch_mod, "knowledge_search",
            new=mock.AsyncMock(return_value={"context": "ctx", "docs": [], "doc_count": 0}),
        ), mock.patch.object(orch_mod, "extract_doc_urls", return_value=("", "", [])), \
           mock.patch.object(orch_mod.asyncio, "create_task", side_effect=_capture_task(created)):
            result = await orch.execute("高血压怎么治疗", user_id="u1", session_id="s1")

        self.assertEqual(result.answer, "最终答案")
        self.assertEqual(result.reflection_score, 0.8)
        orch._evaluate_and_correct.assert_awaited_once()  # 主流程同步反思
        orch._persist_reflection.assert_awaited_once()
        self.assertEqual(created, [])  # 没有通过 create_task 重复后台评估


class TestStreamFastMode(unittest.IsolatedAsyncioTestCase):
    # 8
    async def test_fast_streams_directly_no_correct(self):
        orch = _make_orchestrator(stream_mode="fast")
        orch.router = mock.MagicMock()
        orch.router.route = mock.AsyncMock(return_value=_medical_route())
        orch._build_history_with_memory = mock.AsyncMock(return_value="")
        orch.reflection_agent.evaluate = mock.AsyncMock(return_value=_make_reflection(0.8, False))
        orch.reflection_agent.correct = mock.AsyncMock()

        async def fake_gen():
            for c in ["原始", "答案"]:
                yield c

        orch.medical_agent = mock.MagicMock()
        orch.medical_agent.generate = mock.AsyncMock(return_value=fake_gen())

        import agents.orchestrator as orch_mod
        created = []
        with mock.patch.object(
            orch_mod, "knowledge_search",
            new=mock.AsyncMock(return_value={"context": "ctx", "docs": [], "doc_count": 0}),
        ), mock.patch.object(orch_mod, "extract_doc_urls", return_value=("", "", [])), \
           mock.patch.object(orch_mod.asyncio, "create_task", side_effect=_capture_task(created)):
            out = []
            async for d in orch.execute_stream("q", user_id="u1", session_id="s1"):
                out.append(d)

        self.assertEqual("".join(out), "原始答案")  # 直接流式输出原答案
        orch.reflection_agent.correct.assert_not_called()  # 不修正已发送内容
        self.assertEqual(len(created), 1)  # 后台评估被调度 (不阻塞)


class TestStreamStrictMode(unittest.IsolatedAsyncioTestCase):
    # 9 / 10
    async def test_strict_outputs_corrected_and_preserves_links_memory(self):
        orch = _make_orchestrator(stream_mode="strict")
        orch.router = mock.MagicMock()
        orch.router.route = mock.AsyncMock(return_value=_medical_route())
        orch._build_history_with_memory = mock.AsyncMock(return_value="")
        orch.reflection_agent.threshold = 0.65
        orch.reflection_agent.evaluate = mock.AsyncMock(side_effect=[
            _make_reflection(0.5, True),
            _make_reflection(0.8, False),
        ])
        orch.reflection_agent.correct = mock.AsyncMock(return_value=("修正后的答案", True))

        async def fake_gen():
            for c in ["原始", "答案"]:
                yield c

        orch.medical_agent = mock.MagicMock()
        orch.medical_agent.generate = mock.AsyncMock(return_value=fake_gen())

        import agents.orchestrator as orch_mod
        with mock.patch.object(
            orch_mod, "knowledge_search",
            new=mock.AsyncMock(return_value={"context": "ctx", "docs": [], "doc_count": 0}),
        ), mock.patch.object(
            orch_mod, "extract_doc_urls", return_value=("http://img", "http://pdf", [])
        ):
            out = []
            async for d in orch.execute_stream("q", user_id="u1", session_id="s1"):
                out.append(d)

        full = "".join(out)
        self.assertIn("修正后的答案", full)      # 只输出修正后最终答案
        self.assertNotIn("原始答案", full)
        self.assertIn("http://img", full)        # 保留来源链接
        self.assertIn("http://pdf", full)
        orch.reflection_agent.correct.assert_awaited_once()
        self.assertEqual(orch.reflection_agent.evaluate.await_count, 2)
        # 会话记忆记录的是最终答案 (非原始/非反思过程)
        assistant_turns = [c for _, r, c in orch.memory.turns if r == "assistant"]
        self.assertTrue(assistant_turns)
        self.assertTrue(any("修正后的答案" in c for c in assistant_turns))


class TestStreamDisclaimerMemory(unittest.IsolatedAsyncioTestCase):
    # 10 (免责声明 + 会话记忆)
    async def test_disclaimer_and_session_memory_preserved(self):
        from web_search import NON_MEDICAL_DISCLAIMER

        orch = _make_orchestrator(stream_mode="fast")
        orch.router = mock.MagicMock()
        orch.router.route = mock.AsyncMock(return_value=RouteDecision(
            route="chitchat", intent="general", is_medical=False, use_web_search=False,
        ))
        orch._build_history_with_memory = mock.AsyncMock(return_value="")

        async def fake_gen():
            for c in ["聊天内容"]:
                yield c

        orch.medical_agent = mock.MagicMock()
        orch.medical_agent.generate = mock.AsyncMock(return_value=fake_gen())

        out = []
        async for d in orch.execute_stream("你好吗", session_id="s1"):
            out.append(d)

        full = "".join(out)
        self.assertTrue(full.startswith(NON_MEDICAL_DISCLAIMER))  # 免责声明保留
        self.assertIn("聊天内容", full)
        self.assertIn(("s1", "user", "你好吗"), orch.memory.turns)  # 会话记忆保留


if __name__ == "__main__":
    unittest.main()
