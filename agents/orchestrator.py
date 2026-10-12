"""
agents/orchestrator.py — Multi-Agent 编排器
核心模块: 将原 app.py 中的单链路 RAG 流程升级为 Multi-Agent 协作
编排流程:
  1. RouterAgent: 意图识别 + 路由决策 (零 LLM, <1ms)
  2. Skills 层: 工具调用 (Milvus 检索 / 联网搜索)
  3. MedicalAgent: RAG 生成 (流式/非流式)
  4. ReflectionAgent: 质量评估 (后台异步, 不阻塞响应)
  5. Memory: 记录对话 + 反思结果
性能保证:
  - 缓存层 (Redis) 保持不变, 在编排器外层处理
  - 流式输出路径: Router → Skill → Stream LLM (TTFT 不增加)
  - 反思异步化: 流式结束后 fire-and-forget
  - 锁机制保持不变, 编排器内部无锁操作
"""
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from agents.router_agent import RouterAgent, RouteDecision
from agents.medical_agent import MedicalAgent
from agents.reflection_agent import ReflectionAgent
from skills.tools import knowledge_search, web_search_tool, extract_doc_urls, graph_search
from memory.manager import MemoryManager

logger = logging.getLogger("AgentApp")


# ============================================================================
# 编排器结果数据类
# 用于封装非流式执行的完整返回结果, 供 app.py 的 chatbot() 使用
# ============================================================================
@dataclass
class OrchestratorResult:
    """编排器执行结果 (非流式)"""
    # 最终生成的回答文本 (包含免责声明前缀和 URL 后缀)
    answer: str
    # 原始 Milvus 检索结果列表, 保留用于可追溯性 (feedback_logger 记录 milvus_trace)
    retrieval_docs: List = field(default_factory=list)
    # 路由决策对象, 记录意图分类, 使用的集合类型等路由信息
    route_decision: Optional[RouteDecision] = None
    # LLM token 用量统计（prompt_tokens, completion_tokens 等)
    tokens_used: Dict = field(default_factory=dict)
    # 整体执行耗时 (秒), 从 execute() 入口到返回的总时间
    cost_seconds: float = 0.0
    # 所有检索文档中提取的图片 URL 列表, 用于前端渲染多张图
    image_urls: List = field(default_factory=list)
    # 相关性最高的第一张图片 URL, 前端优先展示此图
    top_image_url: str = ""
    # 相关性最高的 PDF 文档 URL, 前端提供下载/预览链接
    top_pdf_url: str = ""
    # 反思评分: -1.0 表示反思未执行 (区别于 0.0 这个真实的极低分数)
    # 正常执行范围为 [0.0, 1.0]，其中 <0.7 会触发修正建议
    reflection_score: float = -1.0  # -1 表示未执行反思
    # 额外元数据字典, 可扩展存放任何附加信息
    metadata: Dict = field(default_factory=dict)


class AgentOrchestrator:
    """
    Multi-Agent 编排器

    对外暴露两个主方法:
      - execute():        非流式执行, 返回完整结果
      - execute_stream(): 流式执行, 逐 token yield

    内部协调:
      RouterAgent → Skills → MedicalAgent → (async) ReflectionAgent
    """

    def __init__(
        self,
        llm_client: Any,
        milvus_manager: Any,
        ancient_books_manager: Any,
        memory_manager: Optional[MemoryManager] = None,
        enable_reflection: bool = True,
        stream_model: str = "deepseek-chat",
        non_stream_model: str = "deepseek-chat",
        reasoner_model: str = "",
        neo4j_manager: Any = None,
    ):
        # LLM 客户端 (DeepSeek, OpenAI 兼容的异步客户端)
        self.llm_client = llm_client
        # 两个 Milvus 管理器对应不同的知识集合:
        #   - milvus_manager: 现代医学知识库 (modern_medicine 集合)
        #   - ancient_books_manager: 中医古籍知识库 (ancient_books 集合, 如《黄帝内经》等)
        # RouterAgent 会根据用户查询的意图决定检索哪个集合
        self.milvus_manager = milvus_manager
        self.ancient_books_manager = ancient_books_manager
        # Neo4j 图数据库管理器 (可选): 医学问题时与 Milvus 并列检索, 结果合并后一起交给大模型
        self.neo4j_manager = neo4j_manager
        # 记忆管理器: 若外部未提供, 则创建默认的 MemoryManager 实例
        # MemoryManager 内部包含三层记忆: ShortTermMemory (LRU 进程内) + LongTermMemory (Redis Hash, 7天 TTL)
        self.memory = memory_manager or MemoryManager()
        # 是否启用反思环节 (ReflectionAgent), 生产环境建议开启以监控回答质量
        self.enable_reflection = enable_reflection
        # ---- 初始化各 Agent (作为编排器的单例成员, 生命周期与编排器一致)----
        # RouterAgent: 零 LLM 调用的规则引擎路由, 纯关键词 + 正则匹配, <1ms 完成
        self.router = RouterAgent()
        # MedicalAgent: 负责 LLM 回答生成 (流式 & 非流式), 内部持有 Prompt 模板
        self.medical_agent = MedicalAgent(
            llm_client=llm_client,
            model=stream_model,
            reasoner_model=reasoner_model,
        )
        # ReflectionAgent: 四维度质量评估 (准确性/完整性/安全性/相关性), 评分 0~1
        self.reflection_agent = ReflectionAgent(
            llm_client=llm_client,
            model=stream_model,
        )
        # 流式模型 vs 非流式模型的选择:
        #   - stream_model: 首 token 延迟更低, 适合流式逐字输出场景
        #   - non_stream_model: 吞吐量更大/成本更低, 适合一次性返回完整回答
        #   - reasoner_model: 深度思考模型 (think_type != "disabled" 时启用)
        self.stream_model = stream_model
        self.non_stream_model = non_stream_model
        self.reasoner_model = reasoner_model

    # 异步非流式 Multi-Agent 主执行函数
    async def execute(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: str = "",
    ) -> OrchestratorResult:
        """
        非流式执行 Multi-Agent 流程
        返回完整的 OrchestratorResult
        """
        start = time.time()
        # 导入静态响应 (垃圾输入回复, 问候语回复, 非医学免责声明, 身份询问回复)
        from web_search import NON_MEDICAL_DISCLAIMER, GARBAGE_RESPONSE, GREETING_RESPONSE, IDENTITY_RESPONSE

        # ---- Phase 1: 路由决策 ----
        # RouterAgent 是纯规则引擎 (关键词 + 正则), 不调用 LLM, 耗时 <1ms
        # 返回 RouteDecision 对象, 包含 route 类型, intent, 是否需要联网搜索等
        route = await self.router.route(query)
        logger.info(
            f"[Orchestrator] 路由决策 | route={route.route} | "
            f"intent={route.intent} | 耗时={route.duration_ms}ms"
        )

        # ---- Phase 2: 快速拦截 ----
        # 对垃圾输入和简单问候直接返回静态回复, 避免不必要的 Milvus 检索和 LLM 调用
        # 这是性能优化的关键: 绝大部分无效请求在此被拦截, 零 LLM 开销
        if route.route == "garbage":
            return OrchestratorResult(
                answer=GARBAGE_RESPONSE,
                route_decision=route,
                cost_seconds=round(time.time() - start, 3),
            )
        if route.route == "greeting":
            return OrchestratorResult(
                answer=GREETING_RESPONSE,
                route_decision=route,
                cost_seconds=round(time.time() - start, 3),
            )
        # 身份/模型询问 → 固定话术, 零 LLM 调用, 杜绝底层大模型信息泄露
        if route.route == "identity":
            return OrchestratorResult(
                answer=IDENTITY_RESPONSE,
                route_decision=route,
                cost_seconds=round(time.time() - start, 3),
            )

        # ---- Phase 3: 执行技能 (检索/搜索) ----
        # 根据路由决策选择三条分支: 联网搜索 vs Milvus RAG vs 直接 LLM (chitchat)
        #   - 联网搜索: 当 route.use_web_search=True 或 route.route=="web_search" 时触发
        #     适用场景: 时效性问题, 需要外部信息的查询 (天气/新闻/股价等)
        #   - Milvus RAG: 医学专业问题, 检索现代医学或中医古籍集合
        #   - 直接 LLM (chitchat): 日常闲聊, 无需搜索也无需 RAG, LLM 直接回答
        #     适用场景: "我放学要去接孩子", "今天白米饭很香" 等非医学非搜索话语
        # 非医学查询需要添加免责声明前缀, 提醒用户本系统主要面向医学领域
        disclaimer_prefix = ""
        if route.intent == "general":
            disclaimer_prefix = NON_MEDICAL_DISCLAIMER

        # ---- 读取多轮对话历史 (仅会话型请求) ----
        # 仅当传入有效 session_id 时才读取历史; 无 session_id 时 history 恒为空串,
        # 此时下游 Prompt 与升级前逐字节一致, 不增加 token/TTFT (无状态/压测路径不受影响).
        # 同时合并长期记忆中的用户关注话题 (仅已登录用户有), 实现跨会话个性化.
        history = await self._build_history_with_memory(session_id, user_id)

        if route.route == "chitchat":
            # ---- 日常闲聊路径 (直接 LLM, 无 RAG/无搜索) ----
            # 用户的输入既非医学问题也不需要搜索引擎 (如"我放学要去接孩子"),
            # 直接让 LLM 用自身知识回答, 不浪费 Milvus 检索或搜索 API 调用.
            # 传入空 context, LLM 会依据自身训练知识自由回答.
            retrieval_docs = []
            answer, tokens_used = await self.medical_agent.generate(
                query=query,
                context="",
                retrieval_docs=[],
                stream=False,
                model_override=self.non_stream_model,
                history=history,
            )
        elif route.use_web_search or route.route == "web_search":
            # ---- 联网搜索路径 ----
            # 调用 web_search_tool 获取外部搜索结果, 返回 context (搜索摘要) 和 urls
            search_result = await web_search_tool(query)
            context = search_result.get("context", "")
            retrieval_docs = []  # 联网搜索无 Milvus 文档
            # Phase 4: Agent 生成
            # generate_with_web() 专门处理联网搜索上下文的 Prompt 模板
            # 与 generate() 不同: 不包含 RAG 文档引用格式, 而是使用搜索摘要格式
            # 非流式模式使用 non_stream_model (lite 版本) 以降低成本
            answer, tokens_used = await self.medical_agent.generate_with_web(
                query=query,
                search_context=context,
                stream=False,
                model_override=self.non_stream_model,
                history=history,
            )
            # 追加搜索来源 URL, 方便用户验证信息出处
            web_urls = search_result.get("urls", [])
            for web_url in web_urls:
                if web_url:
                    answer += f"\n\n{web_url}"
        else:
            # ---- Milvus RAG 路径 (+ 可选 Neo4j 图检索, 两者并列执行) ----
            # 使用 knowledge_search 技能从向量数据库检索相关文档片段
            # collection_type 由 RouterAgent 决定: "modern_medicine" 或 "ancient_books"
            # Neo4j 图检索: 医学问题时与 Milvus 并发执行, 结果合并后一起注入 Prompt
            milvus_task = knowledge_search(
                query=query,
                collection_type=route.collection_type or "modern_medicine",
                milvus_manager=self.milvus_manager,
                ancient_books_manager=self.ancient_books_manager,
            )
            if self._graph_ready(route):
                search_result, graph_result = await asyncio.gather(
                    milvus_task, self._run_graph_search(query)
                )
            else:
                search_result = await milvus_task
                graph_result = {}
            # 合并 Milvus 与 Neo4j 两路上下文
            context = self._merge_graph_context(search_result.get("context", ""), graph_result)
            retrieval_docs = search_result.get("docs", [])  # 原始 Milvus 检索结果, 保留 pk/doc_id/chunk_index 等元数据
            tokens_used = {}
            # Phase 4: Agent 生成
            # generate() 使用 RAG 专用 Prompt 模板, 包含检索文档的结构化引用
            # 与 generate_with_web() 不同: 传入 retrieval_docs 供 Prompt 模板格式化引用来源
            answer, tokens_used = await self.medical_agent.generate(
                query=query,
                context=context,
                retrieval_docs=retrieval_docs,
                stream=False,
                model_override=self.non_stream_model,
                history=history,
            )

        # 添加免责声明前缀
        if disclaimer_prefix:
            answer = disclaimer_prefix + answer
        cost_seconds = round(time.time() - start, 3)
        # 提取文档 URL (带相关性过滤, 传入 query 以判断文档是否与问题相关)
        top_image_url, top_pdf_url, image_urls = extract_doc_urls(
            retrieval_docs, is_medical=route.is_medical, query=query
        )

        # ---- Phase 5: 异步反思 (fire-and-forget, 不阻塞返回) ----
        # asyncio.create_task 创建独立协程, 主流程立刻返回 OrchestratorResult
        # 反思在后台异步完成, 结果存入长期记忆 (Redis)
        # 三个条件同时满足才触发: 有回答 + 反思功能启用 + 是医学问题
        if self.enable_reflection and answer and route.is_medical:
            asyncio.create_task(
                self._async_reflect(
                    query=query,
                    answer=answer,
                    context=context,
                    user_id=user_id,
                    session_id=session_id,
                )
            )

        # 记录到短期记忆 — 用于多轮对话上下文
        # answer[:200] 截断: 短期记忆仅存摘要, 控制内存占用
        if session_id:
            self.memory.add_turn(session_id, "user", query)
            self.memory.add_turn(session_id, "assistant", answer[:200])

        logger.info(
            f"[Orchestrator] 非流式执行完毕 | route={route.route} | "
            f"耗时={cost_seconds}s | tokens={tokens_used}"
        )
        return OrchestratorResult(
            answer=answer,
            retrieval_docs=retrieval_docs,
            route_decision=route,
            tokens_used=tokens_used,
            cost_seconds=cost_seconds,
            image_urls=image_urls,
            top_image_url=top_image_url,
            top_pdf_url=top_pdf_url,
        )

    # 
    # 异步流式 Multi-Agent 主执行函数
    async def execute_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: str = "",
        think_type: str = "disabled",
    ) -> AsyncGenerator[str, None]:
        """
        流式执行 Multi-Agent 流程
        yield 每个 text delta, 保证 TTFT 最优
        流程:
          Router (0ms) → Skill (Milvus/Web) → Stream LLM → yield deltas
          完成后异步: Reflection → Memory
        """
        t0 = time.time()
        # 导入静态响应
        from web_search import NON_MEDICAL_DISCLAIMER, GARBAGE_RESPONSE, GREETING_RESPONSE, IDENTITY_RESPONSE

        # Phase 1: 路由决策 (<1ms)
        route = await self.router.route(query)
        logger.info(
            f"[Orchestrator:Stream] 路由决策 | route={route.route} | "
            f"耗时={route.duration_ms}ms"
        )

        # Phase 2: 快速拦截
        if route.route == "garbage":
            yield GARBAGE_RESPONSE
            return
        if route.route == "greeting":
            yield GREETING_RESPONSE
            return
        # 身份/模型询问 → 固定话术, 零 LLM 调用, 杜绝底层大模型信息泄露
        if route.route == "identity":
            yield IDENTITY_RESPONSE
            return

        # Phase 3: 执行技能 + 流式生成
        # full_response 用于累积完整的回答文本 (反思和记忆需要完整文本)
        full_response = ""
        disclaimer_prefix = NON_MEDICAL_DISCLAIMER if route.intent == "general" else ""

        # ---- 读取多轮对话历史 (仅会话型请求, 不增加 TTFT) ----
        # 无 session_id 时 history 恒为空串, 下游 Prompt 与升级前逐字节一致;
        # 同时合并长期记忆中的用户关注话题 (仅已登录用户有), 实现跨会话个性化.
        history = await self._build_history_with_memory(session_id, user_id)

        if route.route == "chitchat":
            # ---- 日常闲聊 + 流式路径 (直接 LLM, 无 RAG/无搜索) ----
            # 不调用搜索 API, 不检索 Milvus, LLM 用自身知识直接回答
            if disclaimer_prefix:
                yield disclaimer_prefix
                full_response += disclaimer_prefix
            stream_gen = await self.medical_agent.generate(
                query=query,
                context="",
                retrieval_docs=[],
                stream=True,
                think_type=think_type,
                history=history,
            )
            first_token = False
            async for delta in stream_gen:
                if not first_token:
                    ttft = round(time.time() - t0, 3)
                    logger.info(f"[Orchestrator:Stream] TTFT={ttft}s (chitchat)")
                    first_token = True
                full_response += delta
                yield delta

        elif route.use_web_search or route.route == "web_search":
            # ---- 联网搜索 + 流式生成路径 ----
            t1 = time.time()
            search_result = await web_search_tool(query)
            t1_end = time.time()
            logger.info(f"[Orchestrator:Stream] 联网搜索完成 | 耗时={round(t1_end-t1, 3)}s")
            context = search_result.get("context", "")
            web_urls = search_result.get("urls", [])
            # 流式生成 — generate_with_web(stream=True) 返回 AsyncGenerator
            stream_gen = await self.medical_agent.generate_with_web(
                query=query,
                search_context=context,
                stream=True,
                think_type=think_type,
                history=history,
            )
            # 逐 token 输出, 同时记录 TTFT (Time To First Token, 首 token 延迟)
            first_token = False
            async for delta in stream_gen:
                if not first_token:
                    # 记录从请求开始到首个 token 产出的延迟, 这是用户感知响应速度的核心指标
                    ttft = round(time.time() - t0, 3)
                    logger.info(f"[Orchestrator:Stream] TTFT={ttft}s (联网)")
                    first_token = True
                    # 将免责声明前缀与第一个 token 合并输出, 而非单独 yield
                    # 原因: 单独 yield 会在前端造成闪烁/断流感, 合并后用户看到的是一段连贯文本
                    if disclaimer_prefix:
                        delta = disclaimer_prefix + delta
                full_response += delta
                yield delta
            # 流式生成完毕后追加搜索来源 URL
            # URL 不是 LLM 生成的一部分, 而是附加信息, 因此在流结束后单独 yield
            for web_url in web_urls:
                if web_url:
                    url_text = f"\n\n{web_url}"
                    full_response += url_text
                    yield url_text

        else:
            # ---- Milvus RAG (+ 可选 Neo4j 图检索) + 流式生成路径 ----
            t1 = time.time()
            milvus_task = knowledge_search(
                query=query,
                collection_type=route.collection_type or "modern_medicine",
                milvus_manager=self.milvus_manager,
                ancient_books_manager=self.ancient_books_manager,
            )
            if self._graph_ready(route):
                search_result, graph_result = await asyncio.gather(
                    milvus_task, self._run_graph_search(query)
                )
            else:
                search_result = await milvus_task
                graph_result = {}
            t1_end = time.time()
            logger.info(
                f"[Orchestrator:Stream] Milvus 检索完成 | "
                f"结果数={search_result.get('doc_count', 0)} | "
                f"图检索={'命中' if (graph_result or {}).get('context') else '无'} | "
                f"耗时={round(t1_end-t1, 3)}s"
            )
            context = self._merge_graph_context(search_result.get("context", ""), graph_result)
            retrieval_docs = search_result.get("docs", [])
            # 提取文档 URL (图片和 PDF, 带相关性过滤)
            top_image_url, top_pdf_url, _ = extract_doc_urls(
                retrieval_docs, is_medical=route.is_medical, query=query
            )
            # 流式生成 — generate(stream=True) 返回 AsyncGenerator
            stream_gen = await self.medical_agent.generate(
                query=query,
                context=context,
                retrieval_docs=retrieval_docs,
                stream=True,
                think_type=think_type,
                history=history,
            )
            # 逐 token 输出, 记录 TTFT
            first_token = False
            async for delta in stream_gen:
                if not first_token:
                    ttft = round(time.time() - t0, 3)
                    logger.info(f"[Orchestrator:Stream] TTFT={ttft}s (RAG)")
                    first_token = True
                full_response += delta
                yield delta
            # 流式结束后追加图片和/或 PDF URL (最多各一个, 可同时存在)
            # 这些 URL 不参与 LLM 生成, 是基于检索文档元数据提取的附加信息
            if top_image_url:
                url_text = f"\n\n{top_image_url}"
                full_response += url_text
                yield url_text
            if top_pdf_url:
                url_text = f"\n\n{top_pdf_url}"
                full_response += url_text
                yield url_text

        # ---- Phase 4: 后台异步任务 (流式输出完毕后执行, 不阻塞客户端) ----
        # asyncio.create_task 是 fire-and-forget 模式: 创建协程但不 await 它
        # 即使反思失败也不影响已经输出完毕的流式回答
        # 仅对医学问题执行反思 (非医学问题价值不大)
        if full_response and self.enable_reflection and route.is_medical:
            asyncio.create_task(
                self._async_reflect(
                    query=query,
                    answer=full_response,
                    # 安全检查: context 在 if/else 两个分支中都有定义,
                    # 但使用 'context' in dir() 作为双重保险, 防止意外的变量未定义
                    context=context if 'context' in dir() else "",
                    user_id=user_id,
                    session_id=session_id,
                )
            )

        # 记录到短期记忆 — answer[:200] 截断以控制内存占用
        # 短期记忆仅供多轮对话上下文参考, 不需要完整回答
        if session_id:
            self.memory.add_turn(session_id, "user", query)
            self.memory.add_turn(session_id, "assistant", full_response[:200])

        t_end = time.time()
        logger.info(
            f"[Orchestrator:Stream] 完毕 | 总耗时={round(t_end-t0, 3)}s | "
            f"输出={len(full_response)}字符"
        )

    # ---- Neo4j 图检索辅助方法 ----

    def _graph_ready(self, route: RouteDecision) -> bool:
        """是否启用图检索: 已连接 Neo4j 且为医学问题 (非医学问题查图无意义)"""
        return (
            self.neo4j_manager is not None
            and self.neo4j_manager.is_healthy()
            and route.is_medical
        )

    async def _run_graph_search(self, query: str) -> Dict:
        """执行图检索; 任何异常都不影响主链路, 失败时返回空结果"""
        try:
            return await graph_search(
                query=query,
                llm_client=self.llm_client,
                model=self.non_stream_model,
                neo4j_manager=self.neo4j_manager,
            )
        except Exception as e:
            logger.warning(f"[Orchestrator] 图检索失败(忽略): {e}")
            return {}

    @staticmethod
    def _merge_graph_context(milvus_context: str, graph_result: Dict) -> str:
        """合并 Milvus 与 Neo4j 两路上下文, 任一为空时回退到另一路"""
        graph_context = (graph_result or {}).get("context", "")
        if not graph_context:
            return milvus_context
        if milvus_context:
            return milvus_context + "\n\n" + graph_context
        return graph_context

    async def _load_user_memory(self, user_id: str) -> str:
        """
        读取长期记忆 (用户历史关注话题), 格式化为可注入 Prompt 的文本.
        - 匿名用户 / 无记忆时返回空串, 保证不改变无状态请求的 Prompt.
        - 仅取最近若干条反思记录里的 query 作为关注话题, 不注入原始反馈文本,
          避免把内部质量评估内容泄漏给模型或占用过多 token.
        """
        if not user_id or user_id == "anonymous":
            return ""
        try:
            reflections = await self.memory.load_long_term(user_id, "reflection_history") or []
        except Exception as e:
            logger.warning(f"[Orchestrator] 读取长期记忆失败(忽略): {e}")
            return ""
        topics: List[str] = []
        for item in reflections[-5:]:
            if not isinstance(item, dict):
                continue
            q = (item.get("query") or "").strip()
            if q and q not in topics:
                topics.append(q)
        if not topics:
            return ""
        return "该用户历史关注过的健康话题(供个性化参考, 勿直接复述): " + "；".join(topics)

    async def _build_history_with_memory(self, session_id: str, user_id: str) -> str:
        """合并短期会话历史与长期用户记忆, 供 Prompt 注入. 两者皆空时返回空串."""
        history = self.memory.get_context(session_id, last_n=5) if session_id else ""
        user_memory = await self._load_user_memory(user_id)
        if not user_memory:
            return history
        return f"{user_memory}\n{history}" if history else user_memory

    async def _async_reflect(
        self,
        query: str,
        answer: str,
        context: str,
        user_id: str,
        session_id: str,
    ):
        """
        异步反思 — fire-and-forget, 不阻塞主流程
        结果保存到长期记忆, 供后续请求参考
        """
        try:
            # 调用 ReflectionAgent.evaluate() 进行四维度质量评估
            # context[:500] 截断: 反思只需要检索摘要作为参照, 不需要完整上下文, 节省 token
            result = await self.reflection_agent.evaluate(
                query=query, answer=answer, context=context[:500]
            )
            # 仅对已登录用户保存反思结果到长期记忆 (Redis Hash, 7天 TTL)
            # 匿名用户无法跨会话追踪, 保存无意义
            # 保存的反思历史可用于: 1) 了解用户常见问题模式 2) Agent 自我改进参考
            if user_id != "anonymous":
                await self.memory.save_reflection(
                    user_id=user_id,
                    query=query,
                    score=result.score,
                    feedback=result.feedback,
                )
            # 记录需要修正的回答 — 但流式场景下仅记录日志, 不实际修正
            # 原因: 流式回答已经推送给客户端, 无法撤回或替换
            # 修正结果会存入记忆, 当用户下次问同一类问题时可以参考
 
        except Exception as e:
            # 反思失败不影响主流程 — 本方法运行在 fire-and-forget 的独立 task 中
            logger.warning(f"[Orchestrator] 异步反思失败: {e}")