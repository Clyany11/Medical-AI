"""
agents/router_agent.py — 路由 Agent
职责: 意图识别 + 任务分发
  - 复用现有 web_search.py 的 classify_query_intent() 做快速路由
  - 对于复杂/模糊意图, 使用 LLM 进行二次判断
  - 决定请求走哪条处理链路
路由结果:
  - garbage:       无意义输入 → 静态回复
  - greeting:      打招呼 → 静态回复
  - general:       非医疗通用问题 → 联网搜索
  - medical_rag:   医疗问题 → Milvus RAG
  - medical_web:   医疗问题但需要最新信息 → 联网搜索
  - ancient_books: 中医/古籍问题 → 古籍 Collection
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("AgentApp")


# ============================================================
# RouteDecision — 路由决策的返回类型
# 下游组件 (AgentOrchestrator) 根据此对象决定执行路径:
#   - route 字段决定走哪条链路 (garbage/greeting/web_search/medical_rag)
#   - collection_type 字段区分 Milvus 中的古籍/现代医学集合
#   - is_medical 字段告知 Orchestrator 是否需要启动 MedicalAgent
# Orchestrator 读取 RouteDecision 后, 直接分发到对应的 Agent 或静态回复
# ============================================================
@dataclass
class RouteDecision:
    """路由决策结果"""
    route: str                    # 路由目标
    intent: str                   # 原始意图
    collection_type: str = ""     # Milvus collection 类型
    use_web_search: bool = False  # 是否使用联网搜索
    is_medical: bool = False      # 是否是医疗问题
    confidence: float = 1.0       # 置信度
    duration_ms: float = 0.0      # 路由耗时


class RouterAgent:
    """
    路由 Agent — 零 LLM 调用的快速意图识别

    设计要点:
      - 不调用 LLM, 使用规则引擎做意图识别 (保证 TTFT)
      - 复用现有 classify_query_intent / is_medical_query / needs_web_search
      - detect_intent 做古籍/现代医学分流
    """

    def __init__(self):
        # 延迟导入模式 (Lazy Import Pattern):
        # 所有外部函数引用初始化为 None, 在首次调用 route() 时才真正导入
        # 原因: RouterAgent 在 app.py 启动时就会被实例化, 但此时 web_search.py
        # 和 app.py 可能尚未完成模块级别的初始化, 直接 import 会导致循环依赖
        # 通过将 import 推迟到运行时 (首次调用时), 确保所有模块已完成加载
        self._classify_fn = None        # → web_search.classify_query_intent
        self._medical_fn = None         # → web_search.is_medical_query
        self._web_search_fn = None      # → web_search.needs_web_search
        self._ancient_fn = None         # → web_search.is_ancient_query
        self._detect_intent_fn = None   # → self._detect_intent_inline (内联实现)

    def _ensure_imports(self):
        """延迟导入 web_search 模块的函数"""
        # 懒加载守卫: 仅在 _classify_fn 为 None 时才执行 import
        # web_search.py 包含三个核心规则引擎函数:
        #   - classify_query_intent: 粗粒度意图分类 (garbage/greeting/general/medical)
        #   - is_medical_query: 医疗关键词深度匹配, 判断是否属于医疗领域
        #   - needs_web_search: 判断是否需要联网搜索 (时效性/最新信息需求)
        # 这三个函数都是纯规则引擎, 不调用 LLM, 因此路由延迟极低 (<1ms)
        # 之所以不在文件顶部 import, 是因为 web_search.py 可能反向依赖本模块,
        # 或依赖 app.py 中的组件, 延迟导入可以打破循环依赖链
        if self._classify_fn is None:
            from web_search import (
                classify_query_intent,
                is_medical_query,
                needs_web_search,
                is_ancient_query,
            )
            self._classify_fn = classify_query_intent
            self._medical_fn = is_medical_query
            self._web_search_fn = needs_web_search
            self._ancient_fn = is_ancient_query

    def _ensure_intent_import(self):
        """延迟导入 app 模块的 detect_intent (避免循环)"""
        if self._detect_intent_fn is None:
            # detect_intent 原本定义在 app.py 中, 但 app.py 是整个应用的入口,
            # 会导入几乎所有模块。如果从 app.py 导入 detect_intent, 会触发:
            #   router_agent.py → app.py → agents/orchestrator.py → router_agent.py
            # 形成循环导入链, Python 会抛出 ImportError。
            # 解决方案: 将 detect_intent 的逻辑内联到本类的静态方法中,
            # 功能完全等价 (关键词匹配), 但彻底切断了与 app.py 的导入依赖。
            self._detect_intent_fn = self._detect_intent_inline

    def _detect_intent_inline(self, query: str) -> str:
        """意图识别: 现代医学 vs 古籍 (内联版, 避免从 app.py 导入)

        古籍 / 文言关键词统一维护在 web_search.ANCIENT_KEYWORDS, 通过
        self._ancient_fn (即 is_ancient_query) 判断: 命中则走古籍集合,
        否则走现代医学集合. 这样古籍词表只在一处维护, 避免多处漂移.
        """
        if self._ancient_fn and self._ancient_fn(query):
            return "ancient_books"
        return "modern_medicine"

    async def route(self, query: str) -> RouteDecision:
        """
        执行路由决策
        耗时目标: < 1ms (纯规则, 无 LLM 调用)
        """
        # 记录起始时间, 用于计算路由耗时 (目标 <1ms)
        t0 = time.time()
        # 确保延迟导入的函数已就绪 (首次调用时触发实际 import)
        self._ensure_imports()
        self._ensure_intent_import()

        # ================================================================
        # Step 1: 粗粒度意图分类
        # classify_query_intent 是 web_search.py 中的规则引擎函数
        # 返回四种意图之一: garbage / greeting / general / medical
        #   - garbage: 无意义输入 (乱码, 纯符号, 过短文本等)
        #   - greeting: 打招呼 (你好, Hi, 早上好等)
        #   - general: 非医疗的通用问题
        #   - medical: 与医疗健康相关的问题
        # ================================================================
        intent = self._classify_fn(query)

        # ================================================================
        # Step 2: 快速拦截 — garbage 和 greeting 是"快车道"
        # 这两种意图不需要任何后续处理 (无 RAG、无联网搜索),
        # Orchestrator 收到后直接返回预设的静态回复, 延迟极低
        # ================================================================
        if intent == "garbage":
            return RouteDecision(
                route="garbage", intent=intent,
                duration_ms=round((time.time() - t0) * 1000, 2)
            )
        if intent == "greeting":
            return RouteDecision(
                route="greeting", intent=intent,
                duration_ms=round((time.time() - t0) * 1000, 2)
            )
        # 身份/模型询问 — 同样走快车道, 由 Orchestrator 直接返回固定话术,
        # 不进入 RAG/联网/LLM, 从根本上避免暴露底层大模型 (豆包) 信息
        if intent == "identity":
            return RouteDecision(
                route="identity", intent=intent,
                duration_ms=round((time.time() - t0) * 1000, 2)
            )

        # ================================================================
        # Step 3: 医疗领域深度判断
        # is_medical_query 执行比 classify_query_intent 更细粒度的关键词匹配
        # 即使 Step 1 分类为 "general", 这里仍可能检测出医疗意图
        # 返回 True/False, 决定是否需要走 Milvus RAG 链路
        # ================================================================
        is_medical = self._medical_fn(query)

        # ================================================================
        # Step 4: 联网搜索判断
        # - general 意图: 通过 needs_web_search 判断是否真正需要搜索引擎
        #        (天气 / 新闻 / 股价 / 路线 等时效性问题需联网搜索)
        #        日常闲聊("我放学要去接孩子")不需要搜索, 直接让 LLM 回答即可
        # - medical 意图: 同样通过 needs_web_search 判断是否需要最新信息
        #        (如 "最新疫情数据", "2026年新药上市" 等时效性问题需联网搜索)
        # ================================================================
        if intent == "general":
            use_web_search = self._web_search_fn(query)
        else:
            use_web_search = self._web_search_fn(query)

        # ================================================================
        # Step 5: 最终路由决策 — 路由优先级:
        #   1) use_web_search == True → web_search (联网搜索, 最高优先)
        #   2) is_medical == True → medical_rag (走 Milvus 向量检索)
        #          此时进一步调用 detect_intent 区分古籍/现代医学 collection
        #   3) intent == "general" 且不需搜索 → chitchat (LLM 直接回答, 无 RAG/无搜索)
        #          适用于日常闲聊如"我放学要去接孩子", 不浪费搜索 API 调用
        #   4) 兜底 → chitchat (既非医疗也不需搜索, 让 LLM 自由回答)
        # ================================================================
        if use_web_search:
            # 优先级 1: 联网搜索 — 需要时效性/实时信息的问题
            route = "web_search"
        elif is_medical:
            # 优先级 2: 医疗 RAG — 通过 Milvus 向量检索获取知识
            # 进一步区分古籍 (ancient_books) 和现代医学 (modern_medicine)
            # 不同 collection_type 对应 Milvus 中不同的向量集合
            collection_type = self._detect_intent_fn(query)
            route = "medical_rag"
            return RouteDecision(
                route=route, intent=intent,
                collection_type=collection_type,
                use_web_search=False,
                is_medical=True,
                duration_ms=round((time.time() - t0) * 1000, 2)
            )
        else:
            # 优先级 3: 日常闲聊 — 不需搜索也非医疗, LLM 直接回答
            # 如"我放学要去接孩子", "今天白米饭很香"等日常话语,
            # 无需浪费搜索 API 调用, 直接让 LLM 用自身知识自由回答
            route = "chitchat"

        return RouteDecision(
            route=route, intent=intent,
            use_web_search=use_web_search,
            is_medical=is_medical,
            duration_ms=round((time.time() - t0) * 1000, 2)
        )