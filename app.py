import os
import uvicorn
import asyncio
from langchain_core.documents import Document
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import hashlib
from datetime import datetime
import time
import logging
import base64
import hmac
import re
# 导入新的 redis 工具类
from redis_db import redis_manager
from milvus_db import milvus_manager, ancient_books_manager
from neo4j_db import neo4j_manager
from model import create_deepseek_async_client
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import uuid
from feedback_logger import build_interaction_record, log_interaction_async
from feedback_routes import router as feedback_router
# Token 计费模块
from token_billing import record_token_usage, estimate_tokens_from_chars
from billing_scheduler import start_billing_scheduler
# Feature 5: 导入中间件与扩展模块
from middleware import AuditMiddleware, fallback_manager, extension_registry
# 语音全链路模块 (ASR + TTS)
from voice_pipeline import router as voice_router
# Feature 3: 导入图片URL提取工具
from milvus_schema import extract_image_urls_from_search_results
from testAliyunfiles import upload_to_oss
import auc_http_demo
import tts_http_demo
# Feature: 联网搜索 + 医疗领域判断
from web_search import (
    is_medical_query, needs_web_search, web_search_async,
    classify_query_intent, is_ancient_query,
    NON_MEDICAL_DISCLAIMER, GARBAGE_RESPONSE, GREETING_RESPONSE
)
# 会话管理: 从 chat_service 导入数据库操作函数
from chat_service import get_db, create_session, save_message, check_session_exists
# 引入会话历史路由
from chat_service import router as session_router
# 引入登录认证路由
from auth import router as auth_router, resolve_user_from_authorization
# Multi-Agent 架构模块
from agents.orchestrator import AgentOrchestrator
from memory.manager import MemoryManager


# 配置日志：FileHandler 直接写入 app.log, 不依赖 shell 重定向
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_APP_DIR, "app.log")
_log_formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")

# 清除 root logger 上已有的 handlers (避免 milvus_db 等模块的 basicConfig 干扰)
_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.setLevel(logging.INFO)
# FileHandler：所有业务日志写入 app.log
_fh = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_log_formatter)
_root_logger.addHandler(_fh)
# StreamHandler: 同时输出到控制台 (被 nohup 重定向时也会写入 app.log)
_sh = logging.StreamHandler()
_sh.setFormatter(_log_formatter)
_root_logger.addHandler(_sh)

logger = logging.getLogger("AgentApp")

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 全局 Milvus 实例 (单例, 由 milvus_db 模块用 .env 中的 embedding 配置创建, lifespan 中初始化)
# 说明: embedding 必须与现有向量库 (pdf_agent.db, 智谱 embedding-3 / 2048 维) 的向量空间一致,
#       故直接复用 milvus_db.py 中的 milvus_manager, 不再用其它厂商的客户端重建。


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理

    启动时:
      1. 初始化 Milvus 连接并预热集合
      2. 验证 Redis 连接健康状态
      3. 输出服务就绪日志

    关闭时:
      1. 优雅关闭 Milvus 连接, 释放内存
    """
    # 启动时初始化 Milvus
    logger.info("服务启动中, 正在初始化 Milvus...")
    await milvus_manager.initialize()

    # 初始化古籍 Collection（独立 collection，初始化失败不影响主服务）
    try:
        logger.info("正在初始化古籍 Milvus Collection...")
        await ancient_books_manager.initialize()
        logger.info("古籍 Collection 初始化完成 ✅")
    except Exception as e:
        logger.warning(f"古籍 Collection 初始化失败(不影响主服务): {e}")

    # 验证 Redis 健康状态
    if redis_manager.is_healthy():
        logger.info("Redis 连接正常 ✅")
    else:
        logger.warning("Redis 连接异常 ⚠️, 缓存功能将降级")

    # 初始化 Neo4j 图数据库 (未配置 NEO4J_URI 时自动跳过, 连接失败不影响主服务)
    try:
        await neo4j_manager.initialize()
    except Exception as e:
        logger.warning(f"Neo4j 初始化失败(不影响主服务): {e}")

    # 预热图结构: 图结构固定不变, 启动时查一次写入进程内缓存, 之后请求永不再查
    try:
        from skills.tools import warmup_graph_schema
        await warmup_graph_schema(neo4j_manager)
    except Exception as e:
        logger.warning(f"图结构预热失败(不影响主服务): {e}")

    logger.info("Agent 服务启动完成 ✅, 开始接受请求")
    # Token 计费: 启动每 24 小时定时账单任务
    asyncio.create_task(start_billing_scheduler())
    yield

    # 关闭时优雅释放资源
    logger.info("服务关闭中, 正在释放 Milvus 连接...")
    await milvus_manager.close()
    await ancient_books_manager.close()
    await neo4j_manager.close()
    logger.info("服务已安全关闭 ✅")


app = FastAPI(
    title="医疗 Agent API",
    description="基于 RAG + 数据飞轮的医疗智能问答系统",
    version="2.0.0",
    lifespan=lifespan,
)

# ---- 注册路由 ----
# 数据飞轮反馈路由 (点赞/点踩/专家审核)
app.include_router(feedback_router)
# 语音全链路路由（ASR → RAG+LLM → TTS）
app.include_router(voice_router)
# 注册到 app
app.include_router(session_router)
# 登录认证路由 (注册/登录/获取当前用户)
app.include_router(auth_router)

# ---- 注册中间件 (顺序: 后注册先执行) ----
# Feature 5.1: 审计中间件 (记录所有请求的审计日志)
app.add_middleware(AuditMiddleware, skip_paths=["/health", "/metrics", "/docs", "/openapi.json"])

# CORS 中间件 (允许跨域, 生产环境建议限制 allow_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 统一访问源: 将 localhost 收敛到 127.0.0.1 ----
# 浏览器中 localhost 与 127.0.0.1 是两个不同的源(origin), localStorage(登录 token /
# session_id)按源隔离, 会导致"换个地址就掉登录态/界面不一致". 这里对页面请求(GET)统一
# 重定向到 127.0.0.1, 保证两个地址行为完全一致. 仅处理 localhost, 不影响其他域名/端口.
@app.middleware("http")
async def unify_host_middleware(request: Request, call_next):
    host = request.headers.get("host", "")
    if request.method == "GET" and host.split(":", 1)[0].lower() == "localhost":
        target = str(request.url.replace(scheme="http", hostname="127.0.0.1"))
        return RedirectResponse(url=target, status_code=307)
    return await call_next(request)

# ---- 初始化 LLM 客户端 ----
# 注意: milvus_manager 由 milvus_db 模块创建 (单例) 并在此导入, 不重复创建
print("Milvus 连接成功（将在 lifespan 中完成预热）......")


async_deepseek_client = create_deepseek_async_client()
print("创建 DeepSeek 客户端成功......")

# Redis 连接已在 redis_db 模块导入时自动初始化 (单例模式)

# Multi-Agent: 初始化记忆管理器和编排器
memory_manager = MemoryManager(redis_manager=redis_manager)
agent_orchestrator = AgentOrchestrator(
    llm_client=async_deepseek_client,
    milvus_manager=milvus_manager,
    ancient_books_manager=ancient_books_manager,
    memory_manager=memory_manager,
    enable_reflection=True,
    stream_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    non_stream_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    neo4j_manager=neo4j_manager,
)
logger.info("Multi-Agent 编排器初始化完成 ✅")


def detect_intent(query: str) -> str:
    """
    意图识别：判断用户查询是现代医学知识还是古籍知识

    识别策略：
      - 命中古籍 / 文言关键词 (统一维护在 web_search.ANCIENT_KEYWORDS) → "ancient_books"
      - 默认 → "modern_medicine"

    :param query: 用户查询文本
    :return: "ancient_books" 或 "modern_medicine"
    """
    if is_ancient_query(query):
        logger.info(f"[意图识别] 检测到古籍关键词 | 路由到古籍 Collection")
        return "ancient_books"

    # 默认路由到现代医学知识
    logger.debug(f"[意图识别] 未检测到古籍关键词 | 路由到现代医学 Collection")
    return "modern_medicine"


async def retrieve_milvus(query: str, collection_type: str = None):
    """
    Milvus 向量召回，返回 (context_str, raw_docs)
    流式接口和非流式接口统一使用此函数，便于提取 image_url。

    :param query: 查询文本
    :param collection_type: Collection 类型 ("ancient_books" 或 "modern_medicine")，
                           None 时自动进行意图识别
    :return: (context_str, raw_docs)
    """
    # 自动意图识别
    if collection_type is None:
        collection_type = detect_intent(query)

    # 根据意图选择 manager
    if collection_type == "ancient_books":
        manager = ancient_books_manager
        logger.info(f"[检索] 使用古籍 Collection | query: {query[:30]}...")
    else:
        manager = milvus_manager
        logger.debug(f"[检索] 使用现代医学 Collection | query: {query[:30]}...")

    try:
        results = await manager.search(query)
        context = manager.format_docs(results) if results else ""
        return context, results
    except Exception as e:
        logger.warning(f"Milvus 召回失败 (collection: {collection_type}): {e}")
        return "", []


async def retrieve_milvus_with_docs(query: str, collection_type: str = None):
    """
    Milvus 向量召回, 同时返回原始 Document 列表 (用于数据飞轮溯源)

    :param query: 查询文本
    :param collection_type: Collection 类型，None 时自动识别
    :return: (context_str, raw_docs)
    """
    return await retrieve_milvus(query, collection_type=collection_type)


async def build_prompt_async(query: str, context: str) -> str:
    SYSTEM_PROMPT = """
        System: 你是一个非常得力的医学助手, 你可以通过从数据库中检索出的信息找到问题的答案.
    """
    USER_PROMPT = f"""
        User: 利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题, 具体的问题介于<question>和</question>之间. 如果提供的信息为空, 则按照你的经验知识来给出尽可能严谨准确的回答, 不知道的时候坦诚的承认不了解, 不要编造不真实的信息.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

    return SYSTEM_PROMPT + USER_PROMPT


async def perform_rag_and_llm_async(query: str):
    """
    异步版 RAG 流程 — Multi-Agent 版本

    通过 AgentOrchestrator 协调:
      RouterAgent (意图路由) → Skills (检索/搜索) → MedicalAgent (生成)
      → ReflectionAgent (异步质量评估)

    保留原有接口签名, 确保 get_or_compute_async() 无需修改

    说明: 非流式 `/` 接口为无状态单轮场景 (每次请求一个一次性 session_id),
    不参与多轮对话记忆 (多轮记忆仅在会话型流式接口 /chat_stream 生效),
    因此此处保持无状态调用, 不透传 session_id, 确保压测路径 TTFT 零影响.

    返回: (answer: str, retrieval_docs: List[Document], cost_seconds: float, tokens_used: dict)
    """
    start = time.time()
    logger.info(f"[非流式][Multi-Agent] RAG 流程开始 | query: {query[:30]}...")

    # 调用 Multi-Agent 编排器
    result = await agent_orchestrator.execute(query=query)

    cost_seconds = result.cost_seconds
    logger.info(
        f"[非流式][Multi-Agent] 流程完成 | route={result.route_decision.route if result.route_decision else 'N/A'} | "
        f"总耗时: {cost_seconds}s | tokens: {result.tokens_used}"
    )

    return result.answer, result.retrieval_docs, cost_seconds, result.tokens_used


async def _stream_doubao_no_cache(query: str, t0: float, session_id: str = "", user_id: str = "anonymous"):
    """
    流式生成降级函数（锁竞争失败时的兜底路径）— Multi-Agent 版本

    仅在 stream_doubao_generator 获取分布式锁失败、且缓存中也无结果时调用。
    通过 agent_orchestrator.execute_stream() 执行完整 Multi-Agent 流程。
    不写缓存（避免重复加锁）。

    :param session_id/user_id: 透传给编排器以启用多轮对话记忆 (默认空 = 无状态单轮)
    """
    try:
        first_token_received = False
        async for delta in agent_orchestrator.execute_stream(
            query=query, session_id=session_id, user_id=user_id
        ):
            if not first_token_received:
                ttft = round(time.time() - t0, 3)
                logger.info(f"[TTFT][降级][Multi-Agent] 首字符输出 | TTFT: {ttft}s")
                first_token_received = True
            yield delta

    except Exception as e:
        logger.error(f"[TTFT][降级] 流式生成失败: {e}")
        yield f"Error: {str(e)}"


async def stream_doubao_generator(query: str, think_type: str = "disabled", session_id: str = "", user_id: str = "anonymous"):
    """
    流式生成器 (Multi-Agent 版本 + Redis 缓存)

    缓存策略 (三层防护):
      - 缓存命中: 将缓存内容逐字符模拟流式 yield, TTFT 降至 <10ms
      - 缓存未命中: Multi-Agent 编排器流式输出, 完成后异步写入缓存
      - 防击穿: 分布式锁 + 双重检查
      - 防穿透: <EMPTY> 占位符
      - 防雪崩: set_answer() 内置 ±10% TTL 抖动

    多轮记忆 (新增):
      - session_id 非空且该会话已有历史时, 缓存 Key 会附加历史摘要 (cache_query),
        实现"按会话上下文隔离缓存", 避免同一句问题在不同上下文里串话
      - 无 session_id / 无历史 (首轮、压测、语音) 时 cache_query == query,
        缓存 Key 与升级前逐字节一致, TTFT 与命中行为完全不受影响

    Multi-Agent 流程:
      RouterAgent (意图路由, <1ms) → Skills (Milvus/Web) → MedicalAgent (流式 LLM)
      → ReflectionAgent (异步后台, 不阻塞)
    """
    t0 = time.time()
    logger.info(f"[TTFT][Multi-Agent] 流式请求开始 | query: {query[:30]}...")

    # 计算缓存 Key 用的查询文本: 仅当会话已有历史时附加 12 位历史摘要后缀
    # 无历史时 digest 为空 → cache_query == query → 缓存 Key 与升级前完全一致
    # 已登录用户额外附加 user 段: 长期记忆会改变生成结果, 必须按用户隔离缓存,
    # 否则不同用户问同一句(且都无历史)会互相命中对方被个性化过的答案.
    history_digest = memory_manager.get_context_digest(session_id) if session_id else ""
    key_parts = [query]
    if history_digest:
        key_parts.append(f"ctx:{history_digest}")
    if user_id and user_id != "anonymous":
        key_parts.append(f"user:{user_id}")
    cache_query = "\x00".join(key_parts)

    # 缓存命中时不会进入 Orchestrator, 也就不会写短期记忆; 在此显式补记,
    # 否则多轮记忆会在"该轮命中缓存"时断链 (表现为时好时坏、偶尔记不住)
    def _record_turn(answer: str):
        if session_id and answer and answer != "<EMPTY>":
            memory_manager.add_turn(session_id, "user", query)
            memory_manager.add_turn(session_id, "assistant", answer[:200])

    # ── 1. 查流式缓存 (命中则模拟流式输出, TTFT < 10ms) ──
    cached = await redis_manager.get_answer_async_with_prefix(cache_query, prefix="llm:stream:")
    if cached:
        logger.info(f"[TTFT] 流式缓存命中 ✅ | query: {query[:30]}...")
        _record_turn(cached)
        chunk_size = 20
        for i in range(0, len(cached), chunk_size):
            yield cached[i: i + chunk_size]
            await asyncio.sleep(0)
        return

    # ── 2. 加锁防击穿 ──
    hash_key = hashlib.md5(cache_query.encode("utf-8")).hexdigest()
    lock_token = await asyncio.to_thread(redis_manager.acquire_lock, f"stream:{hash_key}")

    if lock_token:
        try:
            # 双重检查
            cached_retry = await redis_manager.get_answer_async_with_prefix(cache_query, prefix="llm:stream:")
            if cached_retry:
                logger.info(f"[TTFT] 流式缓存命中 (Double Check) ✅ | query: {query[:30]}...")
                _record_turn(cached_retry)
                chunk_size = 20
                for i in range(0, len(cached_retry), chunk_size):
                    yield cached_retry[i: i + chunk_size]
                    await asyncio.sleep(0)
                return

            # ── 3. 缓存未命中: Multi-Agent 编排器流式执行 ──
            full_response = ""

            try:
                first_token_received = False
                async for delta in agent_orchestrator.execute_stream(
                    query=query, think_type=think_type,
                    session_id=session_id, user_id=user_id
                ):
                    if not first_token_received:
                        ttft = round(time.time() - t0, 3)
                        logger.info(f"[TTFT][Multi-Agent] 🎯 首字符输出 | TTFT: {ttft}s")
                        first_token_received = True
                    full_response += delta
                    yield delta

                # 3.7 写入流式缓存 (fire-and-forget)
                t5_end = time.time()
                logger.info(
                    f"[TTFT][Multi-Agent] 流式输出完成 | "
                    f"总耗时: {round(t5_end - t0, 3)}s | 输出字符数: {len(full_response)}"
                )

                if full_response:
                    asyncio.create_task(
                        asyncio.to_thread(
                            redis_manager.set_answer_with_prefix,
                            cache_query, full_response, prefix="llm:stream:"
                        )
                    )
                else:
                    # 防穿透: 写入空占位符
                    asyncio.create_task(
                        asyncio.to_thread(
                            redis_manager.client.setex,
                            redis_manager._generate_key(cache_query, prefix="llm:stream:"), 60, "<EMPTY>"
                        )
                    )

            except Exception as e:
                logger.error(f"[TTFT][Multi-Agent] 流式生成失败: {e}")
                yield f"Error: {str(e)}"

        finally:
            # 确保锁一定被释放
            await asyncio.to_thread(redis_manager.release_lock, f"stream:{hash_key}", lock_token)

    else:
        # 获取锁失败, 等待后尝试读缓存
        await asyncio.sleep(0.01)
        cached_fallback = await redis_manager.get_answer_async_with_prefix(cache_query, prefix="llm:stream:")
        if cached_fallback:
            _record_turn(cached_fallback)
            chunk_size = 20
            for i in range(0, len(cached_fallback), chunk_size):
                yield cached_fallback[i: i + chunk_size]
                await asyncio.sleep(0)
        else:
            # 降级: 直接穿透执行流式 (不缓存, 避免重复加锁)
            async for delta in _stream_doubao_no_cache(query, t0, session_id=session_id, user_id=user_id):
                yield delta


@app.post("/chat_stream")
async def chat_stream(request: Request):
    """流式聊天接口(单独使用, 不参与压力测试)"""
    try:
        data = await request.json()
        query = data.get("question")
        think_type = data.get("think_type", "disabled")  # 默认禁用思考过程输出

        if not query:
            t1_start = time.time()
            voiceAdr = data.get("voice_url")
            query = auc_http_demo.main_voice(voiceAdr)
            t1_end = time.time()
            logger.info(f"[ASR] 自动语音识别完成 | query: {query} | 总耗时: {round(t1_end - t1_start, 3)}s")
        
        if not query:
            raise HTTPException(status_code=400, detail="question required")
        
        # 会话管理: 兼容前端传入的 session_id 和 username
        session_id = data.get("session_id")
        # 优先用 JWT 中的登录用户名作为稳定 user_id; 未登录时回退到前端传入的 username.
        # 稳定的 user_id 是长期记忆 (Redis memory:user:{user_id}) 能跨会话命中的前提.
        auth_user = resolve_user_from_authorization(request.headers.get("authorization"))
        if auth_user and auth_user.get("username"):
            user_id = auth_user["username"]
        else:
            user_id = data.get("username", "anonymous")

        async def stream_with_session():
            # 包装 stream_doubao_generator, 增加会话持久化逻辑
            db = None
            try:
                db = await asyncio.to_thread(get_db)

                # 1. session 不存在则创建, 存在则校验
                if not session_id:
                    current_session_id = await asyncio.to_thread(
                        create_session, db, user_id, query, None
                    )
                else:
                    check = await asyncio.to_thread(
                        check_session_exists, db, user_id, session_id
                    )
                    if not check:
                        current_session_id = await asyncio.to_thread(
                            create_session, db, user_id, query, session_id
                        )
                    else:
                        current_session_id = session_id

                # 2. 保存用户消息
                await asyncio.to_thread(
                    save_message, db, current_session_id, "user", query, user_id
                )

                # 3. 流式输出 (原有逻辑不变), 同时收集完整回复
                # 透传 current_session_id (跨轮次稳定) + user_id, 启用多轮对话记忆
                full_answer = ""
                logger.info(f"用户请求送入大模型 | query: {query}")
                async for delta in stream_doubao_generator(
                    query, think_type=think_type,
                    session_id=current_session_id, user_id=user_id
                ):
                    full_answer += delta
                    yield delta

                # 4. 流式结束后保存机器人回复
                if full_answer:
                    await asyncio.to_thread(
                        save_message, db, current_session_id, "bot", full_answer, user_id
                    )

            except Exception as e:
                logger.error(f"[chat_stream] 会话管理异常: {e}")
                # 会话管理(DB 持久化)出错不影响流式输出, 降级继续输出.
                # 但多轮记忆依赖进程内短期记忆 (与 DB 无关), 因此仍透传 session_id/user_id,
                # 保证 DB 异常时多轮记忆依然可用; 否则记忆链路会随降级一起失效.
                async for delta in stream_doubao_generator(
                    query, think_type=think_type,
                    session_id=session_id or "", user_id=user_id
                ):
                    yield delta
            finally:
                if db:
                    await asyncio.to_thread(db.close)

        # 前端传了 session_id 或 username 时走会话管理路径, 否则保持原有行为
        # 注意: 前端用 "username" 标识用户 (非 "user_id"), 此处必须与 line 472-474 保持一致,
        # 否则"首轮新会话(只带 username 不带 session_id)"会漏判, 落入无状态分支而不落库,
        # 导致该会话不出现在左侧历史列表
        if data.get("session_id") is not None or data.get("username") is not None:
            return StreamingResponse(
                stream_with_session(),
                media_type="text/plain"
            )
        else:
            return StreamingResponse(
                stream_doubao_generator(query, think_type=think_type),
                media_type="text/plain"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_or_compute_async(
    question: str, session_id: str, user_id: str
):
    """
    异步版缓存逻辑(优化版本,防击穿/防穿透/防雪崩)

    优化点：
      1. 使用 redis_manager.get_answer_async() 减少线程池开销
      2. 缓存命中时延迟从 ~30ms 降至 ~10ms
      3. 增强日志监控，记录缓存命中率

    返回: (answer: str, retrieval_docs: List[Document])
      - 缓存命中时 retrieval_docs 为空列表 (无 Milvus 召回, image_urls 也为空)
      - 缓存未命中时 retrieval_docs 为完整召回文档列表 (含 image_url 字段)
    """
    a3 = time.time()

    # 1: 查缓存（缓存命中直接返回，不加锁）
    cached = await redis_manager.get_answer_async(question)
    if cached:
        print("REDIS HIT !!!✅😊")
        cache_cost = round(time.time() - a3, 3)
        logger.info(f"Redis命中 ✅ | 耗时:{cache_cost}s | 问题:{question[:20]}...")
        # 缓存命中时仍记录交互日志 (无原始docs, 溯源为空)
        record = build_interaction_record(
            session_id=session_id,
            user_id=user_id,
            query=question,
            retrieval_docs=[],
            generation=cached,
            tokens_used={},
            cost_seconds=cache_cost,
            source="cache_hit",
        )
        asyncio.create_task(log_interaction_async(record))
        # Token 计费: 缓存命中时按答案字符数估算 text tokens
        asyncio.create_task(record_token_usage(
            user_id=user_id, session_id=session_id,
            text_tokens=estimate_tokens_from_chars(len(cached)),
            voice_tokens=0, source="cache_hit",
        ))
        return cached, []   # 缓存命中: 返回答案 + 空文档列表

    # 2: 加锁防击穿
    hash_key = hashlib.md5(question.encode("utf-8")).hexdigest()
    lock_token = await asyncio.to_thread(redis_manager.acquire_lock, hash_key)

    if lock_token:
        try:
            # 双重检查缓存
            cached_ans_retry = await redis_manager.get_answer_async(question)
            if cached_ans_retry:
                print("REDIS HIT (Double Check) !!!✅😊")
                record = build_interaction_record(
                    session_id=session_id,
                    user_id=user_id,
                    query=question,
                    retrieval_docs=[],
                    generation=cached_ans_retry,
                    tokens_used={},
                    cost_seconds=round(time.time() - a3, 3),
                    source="cache_hit_double_check",
                )
                asyncio.create_task(log_interaction_async(record))
                # Token 计费: 缓存命中时按答案字符数估算 text tokens
                asyncio.create_task(record_token_usage(
                    user_id=user_id, session_id=session_id,
                    text_tokens=estimate_tokens_from_chars(len(cached_ans_retry)),
                    voice_tokens=0, source="cache_hit_double_check",
                ))
                return cached_ans_retry, []   # 双重检查命中: 返回答案 + 空文档列表

            print("Cache Miss ❌, Computing async RAG + LLM...")
            b3 = time.time()
            print("Redis read cost time: ", b3 - a3)

            # 调用非流式 RAG 流程 (返回完整文本 + 溯源数据)
            answer, retrieval_docs, cost_seconds, tokens_used = await perform_rag_and_llm_async(question)

            a4 = time.time()
            if answer:
                # 写缓存
                await asyncio.to_thread(redis_manager.set_answer, question, answer)
            else:
                # 防缓存穿透: 写入空占位符
                await asyncio.to_thread(
                    redis_manager.client.setex,
                    redis_manager._generate_key(question), 60, "<EMPTY>",
                )
            b4 = time.time()
            print("Redis write cost time: ", b4 - a4)

            # 记录完整交互日志 (含 Milvus 溯源)
            record = build_interaction_record(
                session_id=session_id,
                user_id=user_id,
                query=question,
                retrieval_docs=retrieval_docs,
                generation=answer,
                tokens_used=tokens_used,
                cost_seconds=cost_seconds,
                source="non_stream",
            )
            asyncio.create_task(log_interaction_async(record))
            # Token 计费: 非流式接口使用 LLM 返回的精确 token 数
            asyncio.create_task(record_token_usage(
                user_id=user_id, session_id=session_id,
                text_tokens=tokens_used.get("total_tokens", 0),
                voice_tokens=0, source="non_stream",
            ))

            return answer, retrieval_docs   # 缓存未命中: 返回答案 + 完整文档列表
        finally:
            # 确保锁一定被释放
            await asyncio.to_thread(redis_manager.release_lock, hash_key, lock_token)
    else:
        # 获取锁失败, 等待重试
        await asyncio.sleep(0.01)
        cached_fallback = await redis_manager.get_answer_async(question)
        return (cached_fallback or "System busy, calculating..."), []


@app.get("/", include_in_schema=False)
async def index_page():
    """登录入口: 统一引导到登录页。

    服务端无法读取 localStorage 中的 token, 因此这里统一重定向到 /login.html,
    由页面自行判断: 已有 token 则直接跳转 /chat.html, 否则展示登录/注册表单。
    POST / 继续保留为原有问答 API。
    """
    return RedirectResponse(url="/login.html")


@app.get("/index.html", include_in_schema=False)
async def landing_page():
    """返回 Web 首页 (落地页)，作为独立地址保留。"""
    return FileResponse(os.path.join(_APP_DIR, "index.html"))


@app.get("/chat.html", include_in_schema=False)
async def chat_page():
    """返回聊天页面，不暴露项目目录中的其他文件。"""
    return FileResponse(os.path.join(_APP_DIR, "chat.html"))


@app.get("/login.html", include_in_schema=False)
async def login_page():
    """返回登录/注册页面。"""
    return FileResponse(os.path.join(_APP_DIR, "login.html"))


@app.post(
    "/",
    openapi_extra={
        # 该接口用 request: Request 手动解析 body, FastAPI 无法自动推断请求体结构,
        # 这里手动补一份 requestBody schema, 使 /docs 页面能渲染出输入框 (便于浏览器调试)
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["question"],
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户问题 (必填)",
                                "example": "高血压怎么治疗？",
                            },
                            "username": {
                                "type": "string",
                                "description": "用户名 (可选, 默认 anonymous)",
                                "default": "anonymous",
                                "example": "user_001",
                            },
                        },
                    }
                }
            },
        }
    },
)
async def chatbot(request: Request):
    """
    非流式问答接口 (主接口, 压测调用)

    请求体:
      {"question": "高血压怎么治疗？", "username": "user_001"}

    响应体:
      {
        "response":   "LLM 生成的答案",
        "status":     200,
        "time":       "2026-06-26 10:00:00",
        "session_id": "uuid",          # 用于提交👍/👎反馈
        "image_urls": [                # Feature 3: 召回文档关联的图片URL列表
          {"rank": 0, "image_url": "https://...", "pdf_url": "https://...", "text_preview": "..."}
        ],
        "chunks": [                    # 召回文档原文片段 (父块正文/来源/序号)
          {"rank": 0, "text": "...", "source": "...", "chunk_index": 0, "doc_id": "..."}
        ]
      }
    """
    try:
        json_post_raw = await request.json()
        # 处理可能的双重编码
        if isinstance(json_post_raw, str):
            json_post_list = json.loads(json_post_raw)
        else:
            json_post_list = json_post_raw

        query = json_post_list.get('question')
        if not query:
            return {"status": 400, "error": "Question is required"}

        # 每次请求生成唯一 session_id, 用于数据飞轮反馈关联
        session_id = str(uuid.uuid4())
        # user_id = json_post_list.get('user_id', 'anonymous')
        user_id = json_post_list.get("username", "anonymous")
        # 调用缓存逻辑, 同时获取答案和召回文档列表
        response, retrieval_docs = await get_or_compute_async(
            query, session_id=session_id, user_id=user_id
        )

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Feature 3: 提取召回文档中的图片URL, 随答案一并返回给前端
        # 非医疗领域不返回文档 URL (避免返回无关的医学文档链接)
        if is_medical_query(query):
            image_urls = extract_image_urls_from_search_results(retrieval_docs)
        else:
            image_urls = []

        # 返回召回文档的原文片段 (chunk), 供前端展示答案依据来源
        # 注意: 方案 B 父子回表后, 这里返回的是父块正文 (最长 2000 字)
        chunks = [
            {
                "rank":        rank,
                "text":        doc.page_content or "",
                "source":      doc.metadata.get("source", ""),
                "chunk_index": doc.metadata.get("chunk_index"),
                "doc_id":      doc.metadata.get("doc_id", ""),
            }
            for rank, doc in enumerate(retrieval_docs or [])
        ]

        return {
            "response":   response,
            "status":     200,
            "time":       time_str,
            "session_id": session_id,   # 返回给前端, 用于提交👍/👎反馈
            "image_urls": image_urls,   # Feature 3: 图片URL列表 (非医疗领域为空列表)
            "chunks":     chunks,       # 召回文档原文片段 (含全文/来源/序号)
        }

    except Exception as e:
        print(f"Server Error: {e}")
        # Feature 5.3: 服务异常时记录降级事件
        fallback_manager.record_fallback("full_fallback")
        # 即使异常也返回 session_id, 确保前端可以提交反馈
        return {
            "status": 500,
            "error": str(e),
            "session_id": session_id if 'session_id' in locals() else str(uuid.uuid4()),
            "response": "抱歉, 服务暂时不可用, 请稍后重试.",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_urls": [],
        }


@app.get("/health", summary="服务健康检查")
async def health_check():
    """
    服务健康检查接口 (供 K8s/Docker 探针调用)

    检查项:
      - Redis 连接状态
      - Milvus 初始化状态
      - 扩展模块健康状态

    返回:
      {"status": "healthy", "redis": true, "milvus": true, "extensions": {...}}
    """
    redis_ok = redis_manager.is_healthy()
    milvus_ok = milvus_manager._initialized
    neo4j_ok = neo4j_manager.is_healthy()

    # Feature 5.4: 检查所有扩展模块健康状态
    extension_health = await extension_registry.health_check_all()

    overall = "healthy" if (redis_ok and milvus_ok) else "degraded"

    return {
        "status":     overall,
        "redis":      redis_ok,
        "milvus":     milvus_ok,
        "neo4j":      neo4j_ok,
        "extensions": extension_health,
        "timestamp":  datetime.now().isoformat() + "Z",
    }


@app.get("/metrics", summary="运行时监控指标")
async def get_metrics():
    """
    运行时监控指标接口 (供 Prometheus / Grafana 接入)

    返回:
      - Redis 缓存命中率, 写入次数, 错误次数
      - 服务降级统计
      - 已注册的扩展模块列表
    """
    return {
        "redis_metrics":    redis_manager.get_metrics(),
        "redis_info":       redis_manager.get_redis_info(),
        "fallback_stats":   fallback_manager.get_fallback_stats(),
        "extensions":       extension_registry.list_modules(),
        "timestamp":        datetime.now().isoformat() + "Z",
    }


@app.post("/graph_query", summary="通用 Cypher 查询 (只读)")
async def graph_query_endpoint(request: Request):
    """
    在图数据库 (Neo4j) 上执行只读 Cypher 查询

    请求体 (JSON):
      {
        "cypher": "MATCH (d:Disease) RETURN d.name LIMIT 10",  # 必填
        "params": {"name": "糖尿病"},                          # 可选, 参数化查询
        "limit":  100                                          # 可选, 返回行数上限
      }

    返回: {"rows": [...], "count": n, "truncated": bool}
    仅允许读操作; 写语句会被 Neo4j 服务端拒绝 (access mode = READ).
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是合法 JSON")

    cypher = (data.get("cypher") or "").strip()
    if not cypher:
        raise HTTPException(status_code=400, detail="缺少 cypher 字段")

    if not neo4j_manager.is_healthy():
        raise HTTPException(
            status_code=503,
            detail="Neo4j 未连接, 请检查 .env 中 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD",
        )

    result = await neo4j_manager.run_read(
        cypher,
        params=data.get("params") or {},
        limit=data.get("limit"),
    )
    return JSONResponse(result)


@app.get("/get_oss_sign")
async def get_oss_sign(request: Request):
    # ===================== OSS基础配置 =====================
    # 从 .env 读取 (与 upload_to_oss 共用同一对凭证)
    ACCESS_KEY_ID = os.getenv("Alibaba_ACCESS_KEY_ID")
    ACCESS_KEY_SECRET = os.getenv("Alibaba_ACCESS_KEY_SECRET")
    BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "xiaozhuteacher")
    # 注意: Host 必须与 Bucket 地域对应, 且通常不带 https:// 在签名计算中, 但返回给前端时需要带
    ENDPOINT = os.getenv("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
    HOST = f"https://{BUCKET_NAME}.{ENDPOINT}"

    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        logger.error("[get_oss_sign] OSS 凭证未配置 (Alibaba_ACCESS_KEY_ID / Alibaba_ACCESS_KEY_SECRET)")
        return JSONResponse(
            content={"code": 500, "msg": "OSS 凭证未配置, 请在 .env 中设置 Alibaba_ACCESS_KEY_ID / Alibaba_ACCESS_KEY_SECRET"},
            status_code=500,
        )

    # 获取文件名
    file_name = request.query_params.get("fileName")
    if not file_name:
        return JSONResponse(content={"code": 400, "msg": "fileName不能为空"}, status_code=400)

    # 设置过期时间 (建议稍微长一点, 比如 300秒/5分钟, 防止网络慢导致签名过期)
    expire_time = int(time.time()) + 300 
    expiration = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expire_time))

    # 构造 Policy
    # 注意: conditions 中的 key 必须与前端 formData 中的 key 字段值一致
    policy_dict = {
        "expiration": expiration,
        "conditions": [
            ["content-length-range", 0, 50 * 1024 * 1024],  # 最大 50MB
            {"bucket": BUCKET_NAME},
            ["eq", "$key", file_name]  # 强制指定上传的文件名
        ]
    }

    # 1. 将 Policy 字典转换为 JSON 字符串
    # ensure_ascii=False 防止中文文件名乱码 (虽然文件名通常是 英文/数字)
    policy_json_str = json.dumps(policy_dict, ensure_ascii=False)
    
    # 2. Base64 编码 Policy
    policy_encoded = base64.b64encode(policy_json_str.encode('utf-8')).decode('utf-8')

    # 3. 使用 AccessKeySecret 对 Base64 后的 Policy 进行 HMAC-SHA1 签名
    signature = base64.b64encode(
        hmac.new(
            ACCESS_KEY_SECRET.encode('utf-8'),
            policy_encoded.encode('utf-8'),
            hashlib.sha1
        ).digest()
    ).decode('utf-8')

    # 返回给前端的数据
    return JSONResponse(content={
        "host": HOST,
        "fileName": file_name,  # 前端会把这个值赋给 formData 的 'key'
        "policy": policy_encoded,
        "accessKeyId": ACCESS_KEY_ID,  # 前端会把这个值赋给 formData 的 'OSSAccessKeyId'
        "signature": signature
    })


@app.post("/tts")
async def tts(request: Request):
    """
    文字转语音接口：接收文本 → 调用字节 TTS API → 上传 OSS → 返回音频 URL

    请求方式:
      POST /tts
      Body JSON: {"text": "要转换的文本"}

    返回:
      {"url": "https://...oss...mp3"}  成功
      {"url": "", "error": "..."}      失败
    """
    # 1. 获取文本（支持 JSON body 和 query params 两种方式）
    text = None
    try:
        body = await request.json()
        text = body.get("text", "")
    except Exception:
        pass
    if not text:
        text = request.query_params.get("text", "")

    if not text or not text.strip():
        return JSONResponse(content={"url": "", "error": "text 参数不能为空"}, status_code=400)

    text = text.strip()

    # 2. 去除文本末尾的 OSS URL (image_url / pdf_url / txt_url 等)
    text = re.sub(r'\s*https?://\S+', '', text).strip()

    if not text:
        return JSONResponse(content={"url": "", "error": "去除 URL 后文本为空"}, status_code=400)

    logger.info(f"[TTS] 收到请求 | 文本长度: {len(text)} | 文本text: {text}")

    local_path = None
    try:
        # 3. 调用字节 TTS API 生成音频文件
        local_path = await tts_http_demo.tts_http_stream_async(text)

        # 4. 上传到阿里云 OSS
        oss_url = await asyncio.to_thread(upload_to_oss, local_path)

        if not oss_url:
            logger.error("[TTS] OSS 上传失败")
            return JSONResponse(content={"url": "", "error": "音频文件上传 OSS 失败"}, status_code=500)

        logger.info(f"[TTS] 完成 | OSS URL: {oss_url}")
        return JSONResponse(content={"url": oss_url})

    except Exception as e:
        logger.error(f"[TTS] 处理失败: {str(e)}")
        return JSONResponse(content={"url": "", "error": f"TTS 处理失败: {str(e)}"}, status_code=500)

    finally:
        # 4. 清理本地临时文件
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.debug(f"[TTS] 临时文件已清理: {local_path}")
            except OSError:
                pass

if __name__ == '__main__':
    # 压测时建议 workers=1 先调试, 稳定后再调整 workers
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1, log_config=None)
    # uvicorn.run("app:app", host='0.0.0.0', port=8103, workers=4)
