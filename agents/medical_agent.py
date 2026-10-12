"""
agents/medical_agent.py — 医学问答 Agent
职责: 执行 RAG 检索 + 生成医学回答
   - 调用 Milvus 向量检索获取相关文档
   - 构建 Prompt + 调用 LLM 生成回答
   - 流式输出支持
   - 支持工具调用 (knowledge_search, web_search)
设计要点:
   - 继承 BaseAgent 获得 ReAct + Reflection 能力
   - 但日常请求默认关闭反思 (保证 TTFT)
   - 仅对低置信度回答触发反思修正
"""
import logging
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from agents.base import BaseAgent, AgentResult

logger = logging.getLogger("AgentApp")

# 预编译 Prompt 模板 (避免运行时重复构造)
# 模板在模块加载时一次性构建为常量字符串, 运行时仅做 .format() 填充变量,
# 消除每次请求的字符串拼接开销 (对高并发场景有意义)
MEDICAL_SYSTEM_PROMPT = (
    "System: 你是一个非常得力的医学助手, 你可以通过从数据库中检索出的信息找到问题的答案."
)

# 医学问答 User Prompt 模板 — 设计要点:
# 1: XML 标签 (<context>...</context>, <question>...</question>) 包裹上下文和问题:
#      - 让 LLM 明确区分"检索结果"与"用户问题"两个语义区域, 避免信息混淆
#      - XML 标签是 LLM 最稳定识别的结构化分隔符, 比 markdown 或自然语言标记更可靠
# 2: "如果提供的信息为空, 则按照你的经验知识来给出尽可能严谨准确的回答":
#      - 这是 LLM 知识兜底条款, 确保当 Milvus 检索不到相关文档时不会输出空答案
#      - 但优先级低于检索结果 (先看 context, context 为空才使用自身知识)
# 3: "不知道的时候坦诚的承认不了解, 务必保证诚实, 不要编造不真实的信息":
#      - 诚实性条款 (honesty clause): 对医学场景至关重要, 防止 LLM 幻觉造成误诊风险
#      - 明确告诉模型"不知道就说不知道", 比单纯说"不要编造"更有效
MEDICAL_USER_PROMPT_TEMPLATE = (
    "User: 利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题, "
    "具体的问题介于<question>和</question>之间. 如果提供的信息为空, "
    "则按照你的经验知识来给出尽可能严谨准确的回答, "
    "不知道的时候坦诚的承认不了解, 务必保证诚实, 不要编造不真实的信息.\n"
    "<context>\n{context}\n</context>\n\n{history_block}"
    "<question>\n{query}\n</question>"
)

# 多轮对话历史注入块 — 设计要点:
# 1: 仅当本会话存在历史轮次时, 才会被填入 {history_block} 占位符;
#    无历史时 history_block 为空字符串, 此时拼出的 Prompt 与升级前【逐字节一致】,
#    因此首轮 / 无状态(压测、语音)请求的 TTFT、token 成本、缓存键完全不受影响.
# 2: 显式提示模型: 当问题含指代词("它"/"这个"/"上面提到的")时结合历史理解真实意图,
#    这正是解决"第一轮问头疼、第二轮问吃什么药就失忆"问题的关键.
# 3: <history> 标签与 <context>/<question> 同构, 保持 LLM 对结构化分隔符的稳定识别.
HISTORY_BLOCK_TEMPLATE = (
    "下面<history>与</history>之间是本次对话的历史记录(最早的在前, 最近的在后). "
    "请优先关注最近(靠后)的对话轮次来理解用户当前意图; "
    "当用户当前的问题中出现指代或省略(如\"它\", \"这个\", \"那种药\", \"上面提到的\"), "
    "请优先关联最近一轮的话题, 同时保留更早的对话作为背景.\n"
    "<history>\n{history}\n</history>\n\n"
)


def _build_history_block(history: str) -> str:
    """
    将历史对话文本包装成可注入 Prompt 的 <history> 区块.
    关键不变量: history 为空(None 或空串)时返回空字符串,
    使得最终拼出的 Prompt 与升级前【逐字节一致】——
    首轮 / 无状态请求因此完全不受多轮记忆功能影响 (TTFT / token / 缓存键均不变).
    :param history: 已格式化的多轮对话文本 (形如 "用户: xxx\n助手: yyy")
    :return: 含 <history> 标签的注入块, 或空字符串
    """
    if not history:
        return ""
    return HISTORY_BLOCK_TEMPLATE.format(history=history)


# 联网搜索 System Prompt — 与医学版的区别:
#   - 医学版: "从数据库中检索出的信息" → 数据来源是本地 Milvus 向量数据库 (RAG)
#   - 联网版: "通过联网搜索获取的信息" → 数据来源是外部搜索引擎 (实时性更强)
#        - 联网版不包含诚实性条款和 LLM 知识兜底, 因为搜索结果通常不会为空,
#        - 且搜索场景是通用问答, 不需要像医学场景那样严格防幻觉
WEB_SEARCH_SYSTEM_PROMPT = (
    "System: 你是一个智能助手, 你可以通过联网搜索获取的信息来回答问题."
)

# 联网搜索 User Prompt 模板 — 同样使用 XML 标签分隔上下文和问题
# 与医学版的区别: 更简洁, 没有兜底条款和诚实性条款, 直接要求"基于搜索结果给出准确的回答"
WEB_SEARCH_USER_PROMPT_TEMPLATE = (
    "User: 利用介于<context>和</context>之间的联网搜索结果来回答问题, "
    "具体的问题介于<question>和</question>之间. 请基于搜索结果给出准确的回答.\n"
    "<context>\n{context}\n</context>\n\n{history_block}"
    "<question>\n{query}\n</question>"
)


class MedicalAgent(BaseAgent):
    """
    医学问答 Agent

    两种执行模式:
      1. 快速模式 (默认): 单次 RAG + LLM, 无反思, 优先 TTFT
      2. 深度模式: ReAct 循环 + 反思 + 修正, 答案质量优先

    工具集:
      - knowledge_search: 从 Milvus 检索医学知识
      - web_search: 联网搜索补充信息
    """

    def __init__(self, llm_client: Any, model: str = "deepseek-chat", reasoner_model: str = ""):
        # 调用 BaseAgent 基类构造函数, 每个参数的设计考量:
        # - tools=[]: 空列表, 表示 MedicalAgent 本身不直接持有工具
        #   工具 (knowledge_search, web_search_tool) 通过 Skills 层 (skills/tools.py) 注入到
        #   Orchestrator 编排器中, 由编排器决定调用时机, 实现 Agent 与工具的解耦
        # - reflect_enabled=False: 默认关闭自我反思 (ReflectionAgent 的质量评估)
        #   原因: 反思需要额外一轮 LLM 调用 (打分 + 可能的修正), 会增加 ~2-3s 延迟
        #   在 TTFT (首字时间) 敏感的线上场景, 优先保证响应速度
        #   仅在 generate_with_react() 深度模式中临时开启
        # - reflect_threshold=0.6: 比 BaseAgent 默认值 0.7 更低
        #   含义: 只有当反思评分低于 0.6 时才触发自我修正
        #   医学场景设置更宽松的阈值, 是因为 RAG 检索到的专业知识通常已经比较准确,
        #   不需要像通用场景那样频繁触发修正, 避免不必要的延迟
        # - max_iterations=3: 比 BaseAgent 默认值 5 更少
        #   医学问答是典型的"检索 → 生成"单步任务, 很少需要多轮工具调用循环
        #   限制迭代次数可以防止 Agent 陷入无意义的循环, 同时节省 token 开销
        super().__init__(
            name="MedicalAgent",
            system_prompt=MEDICAL_SYSTEM_PROMPT,
            llm_client=llm_client,
            model=model,
            tools=[],               # 工具通过 skills 层注入
            max_iterations=3,
            reflect_enabled=False,  # 默认关闭反思 (保证 TTFT)
            reflect_threshold=0.6,
        )
        # 深度思考模型 ID (为空时在 _stream_generate 中回退到默认模型)
        self.reasoner_model = reasoner_model

    async def generate(
        self,
        query: str,
        context: str,
        retrieval_docs: List = None,
        stream: bool = False,
        think_type: str = "disabled",
        model_override: str = None,
        history: str = "",
    ) -> Any:
        """
        生成医学回答
        :param query: 用户问题
        :param context: 检索到的上下文
        :param retrieval_docs: 原始检索文档 (用于溯源)
        :param stream: 是否流式输出
        :param think_type: 思考类型 ("disabled" / "enabled")
        :param model_override: 覆盖默认模型
        :param history: 多轮对话历史文本 (默认空串). 为空时 Prompt 与升级前逐字节一致,
                        保证首轮/无状态请求 TTFT 不受影响
        :return: 流式时返回 AsyncGenerator, 非流式返回 str
        """
        # model_override 允许调用方在运行时覆盖默认模型
        # 使用场景: Orchestrator 根据意图路由结果选择不同模型 (如 pro 模型处理复杂问题)
        # 如果未传入 model_override, 则使用 __init__ 中设定的默认模型
        model = model_override or self.model
        # 拼接 System Prompt + User Prompt, 将检索到的上下文、历史对话和用户问题填入模板
        # history 为空时 _build_history_block 返回空串, final_prompt 与无多轮版本完全一致
        final_prompt = MEDICAL_SYSTEM_PROMPT + MEDICAL_USER_PROMPT_TEMPLATE.format(
            context=context, query=query, history_block=_build_history_block(history)
        )
        # 分支逻辑: 流式 vs 非流式
        # - 流式 (stream=True): 返回 AsyncGenerator, 用于 SSE 推送, 降低用户感知延迟 (TTFT)
        # - 非流式 (stream=False): 返回完整字符串, 用于需要完整答案的场景 (如缓存写入、反思评估)
        if stream:
            return self._stream_generate(final_prompt, model, think_type)
        else:
            return await self._non_stream_generate(final_prompt, model)

    # 联网搜索回答生成方法 — 与 generate() 的区别:
    # - generate() 使用 MEDICAL_*_PROMPT (RAG 场景, 上下文来自 Milvus 向量库)
    # - generate_with_web() 使用 WEB_SEARCH_*_PROMPT (联网场景, 上下文来自搜索引擎)
    # 两者底层共享相同的 _stream_generate / _non_stream_generate 实现,
    # 仅 Prompt 模板不同, 体现了"Prompt 即策略"的设计思路
    # 
    async def generate_with_web(
        self,
        query: str,
        search_context: str,
        stream: bool = False,
        think_type: str = "disabled",
        model_override: str = None,
        history: str = "",
    ) -> Any:
        """
        基于联网搜索结果生成回答
        :param query: 用户问题
        :param search_context: 搜索结果上下文
        :param stream: 是否流式输出
        :param history: 多轮对话历史文本 (默认空串). 为空时 Prompt 与升级前逐字节一致
        :return: 流式时返回 AsyncGenerator, 非流式返回 str
        """
        model = model_override or self.model
        # 使用联网搜索专用的 Prompt 模板, 将搜索结果填入 <context> 区域
        # history 为空时 history_block 为空串, 最终 Prompt 与无多轮版本完全一致
        final_prompt = WEB_SEARCH_SYSTEM_PROMPT + WEB_SEARCH_USER_PROMPT_TEMPLATE.format(
            context=search_context, query=query, history_block=_build_history_block(history)
        )
        if stream:
            return self._stream_generate(final_prompt, model, think_type)
        else:
            return await self._non_stream_generate(final_prompt, model)

    async def _non_stream_generate(self, prompt: str, model: str) -> Tuple[str, Dict]:
        """非流式生成, 返回 (answer, tokens_used)"""
        # 调用 DeepSeek (OpenAI 兼容) Chat Completions 接口
        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        # 提取 token 用量 — 用于成本监控 (usage 含 prompt_tokens / completion_tokens / total_tokens)
        tokens_used = {}
        usage = getattr(response, "usage", None)
        if usage:
            tokens_used = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        # 提取回答文本 (choices[0].message.content)
        answer = response.choices[0].message.content or ""
        return answer, tokens_used

    async def _stream_generate(
        self, prompt: str, model: str, think_type: str = "disabled"
    ) -> AsyncGenerator[str, None]:
        """流式生成器"""
        # 深度思考: think_type != "disabled" 时切换到推理模型 (deepseek-reasoner)
        # 注意: 推理模型会把思考过程放在 delta.reasoning_content, 正文仍走 delta.content,
        #       因此下面的循环只取 content, 不会把思考过程吐给用户
        if think_type != "disabled" and self.reasoner_model:
            model = self.reasoner_model
        # 调用 DeepSeek Chat Completions 流式接口, 返回 AsyncIterator[ChatCompletionChunk]
        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in response:
            # 部分 chunk (如 usage 统计块) 的 choices 为空, 需跳过
            if not getattr(chunk, "choices", None):
                continue
            delta_text = getattr(chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text  # yield 增量文本, 由上层 SSE 接口推送给客户端

    # =================== 深度模式入口 ===================
    # 日常请求走 generate() (快速模式, 无反思)
    # 当遇到复杂医学问题 (如多步推理、需要工具辅助) 时,
    # 调用此方法开启 ReAct 循环 + 反思, 牺牲延迟换取答案质量
    async def generate_with_react(
        self,
        query: str,
        context: str = "",
        **kwargs,
    ) -> AgentResult:
        """
        深度模式: 启用 ReAct + 反思
        用于复杂医学问题, 需要多步推理或工具辅助
        """
        # 临时开启反思 — 仅对本次请求生效
        self.reflect_enabled = True
        try:
            # 调用 BaseAgent.run() 执行完整的 ReAct 循环 + 反思 + 修正 流程
            result = await self.run(query, context=context, **kwargs)
            return result
        finally:
            # try/finally 保证无论 run() 成功还是抛异常, reflect_enabled 都会被重置
            # 如果不重置, 后续的快速模式请求都会意外触发反思, 导致延迟升高
            self.reflect_enabled = False