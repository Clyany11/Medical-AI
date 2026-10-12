"""
feedback_logger.py — 数据飞轮核心模块
功能概述:
  1. 记录每次用户交互的完整上下文 (query, 召回结果, 生成内容, token消耗, 耗时)
  2. 精确溯源 Milvus 召回文档 (chunk uuid, 原文内容, 图片URL)
  3. 处理点赞/点踩反馈, 驱动数据飞轮持续迭代
  4. 点踩数据自动转入专家审核队列, 产出高价值负样本标注数据
存储架构 (MySQL 主读写 + JSONL 异步备份):
  MySQL 表 (aidoctor 数据库):
    - feedback_interaction      交互记录 (O(1) 索引查询)
    - feedback_thumbs           点赞/点踩记录
    - feedback_expert_queue     专家审核队列
  JSONL 备份 (feedback_logs/):
    - interactions.jsonl        交互记录备份
    - thumbs_up.jsonl           正样本备份
    - thumbs_down.jsonl         负样本备份
    - expert_review_queue.jsonl 专家队列备份
与 app.py 的接口约定:
  from feedback_logger import build_interaction_record, log_interaction_async
  record = build_interaction_record(session_id, user_id, query, retrieval_docs,
                                    generation, tokens_used, cost_seconds, source)
  asyncio.create_task(log_interaction_async(record))
"""
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import aiofiles
from langchain_core.documents import Document
from chat_service import SessionLocal
from sqlalchemy import text
# ===================== 日志配置 =====================
logger = logging.getLogger("FeedbackLogger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# =================== 日志目录与文件路径 (JSONL 备份用) ===================
FEEDBACK_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_logs")
INTERACTIONS_FILE   = os.path.join(FEEDBACK_LOG_DIR, "interactions.jsonl")
THUMBS_UP_FILE      = os.path.join(FEEDBACK_LOG_DIR, "thumbs_up.jsonl")
THUMBS_DOWN_FILE    = os.path.join(FEEDBACK_LOG_DIR, "thumbs_down.jsonl")
EXPERT_QUEUE_FILE   = os.path.join(FEEDBACK_LOG_DIR, "expert_review_queue.jsonl")
# 启动时自动创建日志目录
os.makedirs(FEEDBACK_LOG_DIR, exist_ok=True)
# ======= MySQL 私有辅助函数 (同步, 通过 asyncio.to_thread 调用)=======
def _json_serial(obj):
    """JSON 序列化辅助: 处理 datetime 对象"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)
def _to_json_str(obj) -> str:
    """将 Python 对象序列化为 JSON 字符串, 用于存入 MySQL JSON 列"""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=_json_serial)
def _db_insert_interaction(record: Dict[str, Any]) -> None:
    """同步: 向 feedback_interaction 表插入一条交互记录"""
    ts = record.get("timestamp")
    if isinstance(ts, datetime):
        created_at = ts.replace(tzinfo=None)
    else:
        created_at = datetime.now(timezone.utc)
    sql = text("""
        INSERT INTO feedback_interaction
            (session_id, user_id, query, generation, source,
             cost_seconds, tokens_used, milvus_trace, created_at)
        VALUES
            (:session_id, :user_id, :query, :generation, :source,
             :cost_seconds, :tokens_used, :milvus_trace, :created_at)
    """)
    with SessionLocal() as db:
        db.execute(sql, {
            "session_id":   record.get("session_id", ""),
            "user_id":      record.get("user_id", "anonymous"),
            "query":        record.get("query", ""),
            "generation":   record.get("generation", ""),
            "source":       record.get("source", ""),
            "cost_seconds": record.get("cost_seconds", 0),
            "tokens_used":  _to_json_str(record.get("tokens_used")),
            "milvus_trace": _to_json_str(record.get("milvus_trace")),
            "created_at":   created_at,
        })
        db.commit()
def _db_find_interaction(session_id: str) -> Optional[Dict[str, Any]]:
    """同步: 按 session_id 从 feedback_interaction 表查询 (索引查询 O(1))"""
    sql = text("""
        SELECT session_id, user_id, query, generation, source,
               cost_seconds, tokens_used, milvus_trace, created_at
        FROM feedback_interaction
        WHERE session_id = :session_id
        LIMIT 1
    """)
    with SessionLocal() as db:
        row = db.execute(sql, {"session_id": session_id}).fetchone()
    if not row:
        return None
    return {
        "session_id":   row[0],
        "user_id":      row[1],
        "query":        row[2],
        "generation":   row[3],
        "source":       row[4],
        "cost_seconds": row[5],
        "tokens_used":  json.loads(row[6]) if row[6] else {},
        "milvus_trace": json.loads(row[7]) if row[7] else [],
        "created_at":   row[8],
    }
def _db_insert_thumbs(record: Dict[str, Any]) -> None:
    """同步: 向 feedback_thumbs 表插入点赞/点踩记录"""
    sql = text("""
        INSERT INTO feedback_thumbs
            (session_id, user_id, feedback_type, feedback_reason, content,
             answer, query, bad_generation, milvus_trace, tokens_used,
             cost_seconds, expert_task_id, data_value, created_at)
        VALUES
            (:session_id, :user_id, :feedback_type, :feedback_reason, :content,
             :answer, :query, :bad_generation, :milvus_trace, :tokens_used,
             :cost_seconds, :expert_task_id, :data_value, :created_at)
    """)
    with SessionLocal() as db:
        db.execute(sql, {
            "session_id":     record.get("session_id", ""),
            "user_id":        record.get("user_id", "anonymous"),
            "feedback_type":  record.get("feedback_type", ""),
            "feedback_reason":record.get("feedbackType"),
            "content":        record.get("content"),
            "answer":         record.get("answer"),
            "query":          record.get("query", ""),
            "bad_generation": record.get("bad_generation"),
            "milvus_trace":   _to_json_str(record.get("milvus_trace")),
            "tokens_used":    _to_json_str(record.get("tokens_used")),
            "cost_seconds":   record.get("cost_seconds", 0),
            "expert_task_id": record.get("expert_task_id"),
            "data_value":     record.get("data_value"),
            "created_at":     datetime.now(timezone.utc),
        })
        db.commit()
def _db_insert_thumbs_down_and_expert(
    thumbs_record: Dict[str, Any],
    expert_task: Dict[str, Any],
) -> None:
    """同步: 事务写入 feedback_thumbs + feedback_expert_queue"""
    sql_thumbs = text("""
        INSERT INTO feedback_thumbs
            (session_id, user_id, feedback_type, feedback_reason, content,
             answer, query, bad_generation, milvus_trace, tokens_used,
             cost_seconds, expert_task_id, data_value, created_at)
        VALUES
            (:session_id, :user_id, :feedback_type, :feedback_reason, :content,
             :answer, :query, :bad_generation, :milvus_trace, :tokens_used,
             :cost_seconds, :expert_task_id, :data_value, :created_at)
    """)
    sql_expert = text("""
        INSERT INTO feedback_expert_queue
            (task_id, session_id, user_id, status, feedback_reason, content,
             answer, query, bad_generation, milvus_trace, created_at)
        VALUES
            (:task_id, :session_id, :user_id, :status, :feedback_reason, :content,
             :answer, :query, :bad_generation, :milvus_trace, :created_at)
    """)
    with SessionLocal() as db:
        db.execute(sql_thumbs, {
            "session_id":     thumbs_record.get("session_id", ""),
            "user_id":        thumbs_record.get("user_id", "anonymous"),
            "feedback_type":  thumbs_record.get("feedback_type", "thumbs_down"),
            "feedback_reason":thumbs_record.get("feedbackType"),
            "content":        thumbs_record.get("content"),
            "answer":         thumbs_record.get("answer"),
            "query":          thumbs_record.get("query", ""),
            "bad_generation": thumbs_record.get("bad_generation"),
            "milvus_trace":   _to_json_str(thumbs_record.get("milvus_trace")),
            "tokens_used":    _to_json_str(thumbs_record.get("tokens_used")),
            "cost_seconds":   thumbs_record.get("cost_seconds", 0),
            "expert_task_id": thumbs_record.get("expert_task_id"),
            "data_value":     thumbs_record.get("data_value"),
            "created_at":     datetime.now(timezone.utc),
        })
        db.execute(sql_expert, {
            "task_id":        expert_task.get("task_id", ""),
            "session_id":     expert_task.get("session_id", ""),
            "user_id":        expert_task.get("user_id", "anonymous"),
            "status":         expert_task.get("status", "pending"),
            "feedback_reason":expert_task.get("feedbackType"),
            "content":        expert_task.get("content"),
            "answer":         expert_task.get("answer"),
            "query":          expert_task.get("query", ""),
            "bad_generation": expert_task.get("bad_generation"),
            "milvus_trace":   _to_json_str(expert_task.get("milvus_trace")),
            "created_at":     datetime.now(timezone.utc),
        })
        db.commit()
def _db_update_expert_task(
    task_id: str,
    expert_id: str,
    expert_answer: str,
    notes: Optional[str],
) -> bool:
    """同步: 更新 feedback_expert_queue 中指定任务的状态"""
    sql = text("""
        UPDATE feedback_expert_queue
        SET status = 'completed',
            expert_id = :expert_id,
            expert_answer = :expert_answer,
            notes = :notes,
            reviewed_at = :reviewed_at
        WHERE task_id = :task_id
    """)
    with SessionLocal() as db:
        result = db.execute(sql, {
            "task_id":       task_id,
            "expert_id":     expert_id,
            "expert_answer": expert_answer,
            "notes":         notes,
            "reviewed_at":   datetime.now(timezone.utc),
        })
        db.commit()
        return result.rowcount > 0
def _db_find_expert_task(task_id: str) -> Optional[Dict[str, Any]]:
    """同步: 按 task_id 从 feedback_expert_queue 查询 (唯一索引 O(1))"""
    sql = text("""
        SELECT task_id, session_id, user_id, status, feedback_reason,
               content, answer, query, bad_generation, milvus_trace,
               expert_id, expert_answer, notes, created_at, reviewed_at
        FROM feedback_expert_queue
        WHERE task_id = :task_id
        LIMIT 1
    """)
    with SessionLocal() as db:
        row = db.execute(sql, {"task_id": task_id}).fetchone()
    if not row:
        return None
    return {
        "task_id":        row[0],
        "session_id":     row[1],
        "user_id":        row[2],
        "status":         row[3],
        "feedbackType":   row[4],
        "content":        row[5],
        "answer":         row[6],
        "query":          row[7],
        "bad_generation": row[8],
        "milvus_trace":   json.loads(row[9]) if row[9] else [],
        "expert_id":      row[10],
        "expert_answer":  row[11],
        "notes":          row[12],
        "created_at":     row[13],
        "reviewed_at":    row[14],
    }
def _db_get_stats() -> Dict[str, int]:
    """同步: 从 MySQL 统计各表行数"""
    with SessionLocal() as db:
        total      = db.execute(text("SELECT COUNT(*) FROM feedback_interaction")).scalar() or 0
        thumbs_up  = db.execute(text("SELECT COUNT(*) FROM feedback_thumbs WHERE feedback_type='thumbs_up'")).scalar() or 0
        thumbs_down= db.execute(text("SELECT COUNT(*) FROM feedback_thumbs WHERE feedback_type='thumbs_down'")).scalar() or 0
        pending    = db.execute(text("SELECT COUNT(*) FROM feedback_expert_queue WHERE status='pending'")).scalar() or 0
    return {
        "total_interactions":    int(total),
        "thumbs_up":             int(thumbs_up),
        "thumbs_down":           int(thumbs_down),
        "expert_queue_pending":  int(pending),
    }
def _db_get_expert_queue(
    status: Optional[str],
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    """同步: 分页查询 feedback_expert_queue"""
    if status:
        count_sql = text("SELECT COUNT(*) FROM feedback_expert_queue WHERE status = :status")
        rows_sql  = text("""
            SELECT task_id, session_id, user_id, status, feedback_reason,
                   content, answer, query, bad_generation, milvus_trace,
                   expert_id, expert_answer, notes, created_at, reviewed_at
            FROM feedback_expert_queue
            WHERE status = :status
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        params = {"status": status, "limit": limit, "offset": offset}
        count_params = {"status": status}
    else:
        count_sql = text("SELECT COUNT(*) FROM feedback_expert_queue")
        rows_sql  = text("""
            SELECT task_id, session_id, user_id, status, feedback_reason,
                   content, answer, query, bad_generation, milvus_trace,
                   expert_id, expert_answer, notes, created_at, reviewed_at
            FROM feedback_expert_queue
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        params = {"limit": limit, "offset": offset}
        count_params = {}
    with SessionLocal() as db:
        total = db.execute(count_sql, count_params).scalar() or 0
        rows  = db.execute(rows_sql, params).fetchall()
    tasks = []
    for row in rows:
        tasks.append({
            "task_id":        row[0],
            "session_id":     row[1],
            "user_id":        row[2],
            "status":         row[3],
            "feedbackType":   row[4],
            "content":        row[5],
            "answer":         row[6],
            "query":          row[7],
            "bad_generation": row[8],
            "milvus_trace":   json.loads(row[9]) if row[9] else [],
            "expert_id":      row[10],
            "expert_answer":  row[11],
            "notes":          row[12],
            "created_at":     str(row[13]) if row[13] else None,
            "reviewed_at":    str(row[14]) if row[14] else None,
        })
    return {"tasks": tasks, "total": int(total), "offset": offset, "limit": limit}
# ===================== Milvus 文档精确溯源 =====================
def extract_milvus_trace(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    从 Milvus 召回的 Document 列表中提取精确溯源信息
    Milvus Standalone 模式下, langchain_milvus 将 Milvus 字段映射到
    Document.metadata 中, 常见字段包括:
      - pk / _id / id : Milvus 自动生成的主键 (auto_id=True)
      - doc_id        : vectors.py 写入时设置的文档 UUID
      - source        : 原始文件路径
      - chunk_index   : 分块序号
      - page          : PDF 页码
      - image_url     : 图片 URL
    :param docs: milvus_manager.search() 返回的 List[Document]
    :return: 溯源记录列表，每条记录包含 rank/pk/doc_id/source/chunk_index/
             page/image_url/page_content_preview/full_metadata
    """
    trace = []
    for rank, doc in enumerate(docs):
        meta = doc.metadata or {}
        trace.append({
            # 召回排名 (0-based)
            "rank": rank,
            # Milvus 主键（langchain_milvus 存于 metadata["pk"] 或 "_id"）
            "pk": str(meta.get("pk") or meta.get("_id") or meta.get("id", "")),
            # vectors.py 写入时的文档 UUID (metadata={"doc_id": str(uuid.uuid4())})
            "doc_id": str(meta.get("doc_id", "")),
            # 原始文件来源路径
            "source": str(meta.get("source", "")),
            # 分块序号 (用于定位原文位置)
            "chunk_index": str(meta.get("chunk_index") or meta.get("chunk_id", "")),
            # PDF 页码 (非 PDF 来源时为空)
            "page": str(meta.get("page", "")),
            # 图片 URL (存于 metadata, 不参与向量化)
            "image_url": str(meta.get("image_url", "")),
            # 原文内容预览 (前 200 字, 避免日志文件过大)
            "page_content_preview": (doc.page_content or "")[:200],
            # 完整 metadata 备份 (用于深度溯源和调试)
            "full_metadata": meta,
        })
    return trace
# ===================== 构建交互记录 =====================
def build_interaction_record(
    session_id: str,
    user_id: str,
    query: str,
    retrieval_docs: List[Document],
    generation: str,
    tokens_used: Dict[str, Any],
    cost_seconds: float,
    source: str,
) -> Dict[str, Any]:
    """
    构建完整的交互记录字典, 供 log_interaction_async 写入 JSONL
    此函数签名与 app.py 中的调用完全一致, 不可修改参数名称:
        record = build_interaction_record(
            session_id=session_id,
            user_id=user_id,
            query=question,
            retrieval_docs=retrieval_docs,   # List[Document]，缓存命中时为 []
            generation=answer,
            tokens_used=tokens_used,         # {prompt_tokens, completion_tokens, total_tokens}
            cost_seconds=cost_seconds,
            source="non_stream" | "cache_hit" | "cache_hit_double_check",
        )
    :param session_id:      本次请求的唯一会话 ID (UUID, 由 app.py 生成)
    :param user_id:         用户标识 (前端传入, 默认 "anonymous")
    :param query:           用户原始问题
    :param retrieval_docs:  Milvus 召回的 Document 列表 (缓存命中时为空列表)
    :param generation:      LLM 生成的答案文本
    :param tokens_used:     token 用量字典
    :param cost_seconds:    本次请求总耗时(秒)
    :param source:          请求来源标记
    :return: 结构化交互记录字典
    """
    return {
        # ---- 会话标识 ----
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc),
        # ---- 核心数据 ----
        "query": query,
        "generation": generation,
        # ---- 性能指标 ----
        "source": source,           # 请求来源 (non_stream / cache_hit 等)
        "cost_seconds": cost_seconds,
        "tokens_used": tokens_used, # {prompt_tokens, completion_tokens, total_tokens}
        # ---- Milvus 精确溯源 ----
        # 包含每个召回 chunk 的 uuid, 原文内容, 图片 URL 等
        "milvus_trace": extract_milvus_trace(retrieval_docs),
        # ---- 反馈状态 (初始为 null, 用户操作后更新) ----
        "feedback": None,           # "thumbs_up" | "thumbs_down" | null
        # ---- 专家审核 (点踩后填充) ----
        "expert_task_id": None,     # 专家审核任务 ID
        "expert_answer": None,      # 专家提供的标准答案 (回写后填充)
    }
# ===================== 异步 JSONL 写入 =====================
async def _append_jsonl(filepath: str, record: Dict[str, Any]) -> None:
    """
    异步追加写入 JSONL 文件
    使用 aiofiles 进行非阻塞 I/O, 避免文件写入阻塞 FastAPI 事件循环
    每条记录独占一行, 便于流式读取, grep 和增量处理
    :param filepath: 目标 JSONL 文件路径
    :param record:   待写入的字典记录
    """
    try:
        async with aiofiles.open(filepath, mode="a", encoding="utf-8") as f:
            await f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=lambda obj: obj.isoformat() if isinstance(obj, datetime) else str(obj)
                ) + "\n")
    except Exception as e:
        logger.error(f"写入 JSONL 失败 | 文件: {filepath} | 错误: {e}")
async def log_interaction_async(record: Dict[str, Any]) -> None:
    """
    异步记录交互日志: MySQL 主写 + JSONL 异步备份
    由 app.py 通过 asyncio.create_task() 调用, 不阻塞主请求链路
    :param record: build_interaction_record() 返回的交互记录字典
    """
    try:
        await asyncio.to_thread(_db_insert_interaction, record)
    except Exception as e:
        logger.error(f"MySQL 写入交互记录失败 | session_id: {record.get('session_id', '')} | 错误: {e}")
    # JSONL 异步备份 (fire-and-forget, 不影响主流程)
    asyncio.create_task(_append_jsonl(INTERACTIONS_FILE, record))
    logger.debug(f"交互日志已记录 | session_id: {record.get('session_id', '')}")
# ===================== Feature 1: 反馈处理函数 =====================
async def log_thumbs_up(session_id: str, user_id: str) -> bool:
    """
    处理点赞反馈, MySQL 主写 + JSONL 异步备份
    :param session_id: 会话 ID
    :param user_id:    提交反馈的用户 ID
    :return: 始终返回 True (兼容旧行为)
    """
    record = await _find_interaction_by_session(session_id)
    thumbs_record = {
        "session_id":   session_id,
        "user_id":      user_id,
        "feedback_type":"thumbs_up",
        "timestamp":    datetime.now(timezone.utc),
        "query":        record.get("query", ""),
        "generation":   record.get("generation", ""),
        "milvus_trace": record.get("milvus_trace", []),
        "tokens_used":  record.get("tokens_used", {}),
        "cost_seconds": record.get("cost_seconds", 0),
    }
    try:
        await asyncio.to_thread(_db_insert_thumbs, thumbs_record)
    except Exception as e:
        logger.error(f"MySQL 写入点赞失败 | session_id: {session_id} | 错误: {e}")
    asyncio.create_task(_append_jsonl(THUMBS_UP_FILE, thumbs_record))
    logger.info(f"点赞记录已写入 | session_id: {session_id}")
    return True
async def log_thumbs_down(session_id: str, user_id: str, feedbackType: Optional[str] = None, content: Optional[str] = None, answer: Optional[str] = None) -> Dict[str, Any]:
    """
    处理点踩反馈, MySQL 事务写入 + JSONL 异步备份
    :param session_id:    会话 ID
    :param user_id:       提交反馈的用户 ID
    :param feedbackType:  点踩原因枚举值 (harmful/false_info/not_helpful/other)
    :param content:       用户输入的反馈意见文本
    :param answer:        被点踩的 AI 回复内容
    :return: {"success": bool, "expert_task_id": str | None}
    """
    record = await _find_interaction_by_session(session_id)
    expert_task_id = str(uuid.uuid4())
    thumbs_record = {
        "session_id":     session_id,
        "user_id":        user_id,
        "feedback_type":  "thumbs_down",
        "feedbackType":   feedbackType,
        "content":        content,
        "answer":         answer,
        "timestamp":      datetime.now(timezone.utc),
        "query":          record.get("query", ""),
        "bad_generation": answer if answer else record.get("generation", ""),
        "milvus_trace":   record.get("milvus_trace", []),
        "tokens_used":    record.get("tokens_used", {}),
        "cost_seconds":   record.get("cost_seconds", 0),
        "expert_task_id": expert_task_id,
        "data_value":     "high",
        "note":           "用户标记为不满意, 已转交专家审核, 是改进 Agent 的关键负样本.",
    }
    expert_task = {
        "task_id":        expert_task_id,
        "session_id":     session_id,
        "user_id":        user_id,
        "created_at":     datetime.now(timezone.utc),
        "status":         "pending",
        "feedbackType":   feedbackType,
        "content":        content,
        "answer":         answer,
        "query":          record.get("query", ""),
        "bad_generation": answer if answer else record.get("generation", ""),
        "milvus_trace":   record.get("milvus_trace", []),
        "expert_id":      None,
        "expert_answer":  None,
        "reviewed_at":    None,
        "notes":          None,
    }
    try:
        await asyncio.to_thread(_db_insert_thumbs_down_and_expert, thumbs_record, expert_task)
    except Exception as e:
        logger.error(f"MySQL 写入点踩失败 | session_id: {session_id} | 错误: {e}")
    asyncio.create_task(_append_jsonl(THUMBS_DOWN_FILE, thumbs_record))
    asyncio.create_task(_append_jsonl(EXPERT_QUEUE_FILE, expert_task))
    logger.info(f"专家审核任务已创建 | task_id: {expert_task_id} | session_id: {session_id}")
    return {"success": True, "expert_task_id": expert_task_id}
async def submit_expert_answer(
    task_id: str,
    expert_id: str,
    expert_answer: str,
    notes: Optional[str] = None,
) -> bool:
    """
    专家提交标准答案, MySQL UPDATE + JSONL 备份
    :param task_id:       专家审核任务 ID
    :param expert_id:     专家标识
    :param expert_answer: 专家提供的标准医学答案
    :param notes:         专家备注
    :return: 是否成功找到并更新任务
    """
    task = await _find_expert_task(task_id)
    if not task:
        logger.warning(f"专家提交失败: 未找到 task_id={task_id}")
        return False
    try:
        updated = await asyncio.to_thread(
            _db_update_expert_task, task_id, expert_id, expert_answer, notes
        )
        if not updated:
            logger.warning(f"MySQL UPDATE 未命中 task_id={task_id}")
    except Exception as e:
        logger.error(f"MySQL 更新专家任务失败 | task_id: {task_id} | 错误: {e}")
    # JSONL 备份: 追加一条 completed 记录
    completed_record = dict(task)
    completed_record.update({
        "status":        "completed",
        "expert_id":     expert_id,
        "expert_answer": expert_answer,
        "notes":         notes,
        "reviewed_at":   datetime.now(timezone.utc),
    })
    asyncio.create_task(_append_jsonl(EXPERT_QUEUE_FILE, completed_record))
    logger.info(f"专家答案已提交 | task_id: {task_id} | expert_id: {expert_id}")
    return True
# ===================== 统计与查询函数 =====================
async def get_feedback_stats() -> Dict[str, Any]:
    """
    统计反馈数据, 从 MySQL 读取 (O(1) COUNT 查询)
    :return: 统计字典
    """
    try:
        stats = await asyncio.to_thread(_db_get_stats)
    except Exception as e:
        logger.error(f"MySQL 统计查询失败: {e}")
        stats = {
            "total_interactions":   0,
            "thumbs_up":            0,
            "thumbs_down":          0,
            "expert_queue_pending": 0,
        }
    stats["feedback_log_dir"] = FEEDBACK_LOG_DIR
    return stats
async def get_expert_queue(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    分页查询专家审核队列, 从 MySQL 读取 (索引查询)
    :param status: "pending" | "completed" | None (返回全部)
    :param limit:  每页数量 (默认50)
    :param offset: 分页偏移 (默认0)
    :return: {"tasks": [...], "total": int}
    """
    try:
        return await asyncio.to_thread(_db_get_expert_queue, status, limit, offset)
    except Exception as e:
        logger.error(f"MySQL 查询专家队列失败: {e}")
        return {"tasks": [], "total": 0, "offset": offset, "limit": limit}
# ===================== 内部辅助函数 =====================
async def _find_interaction_by_session(session_id: str) -> Dict[str, Any]:
    """
    按 session_id 查找交互记录, 优先查 MySQL (索引 O(1)), 降级返回空字典
    :param session_id: 目标会话 ID
    :return: 找到的记录字典, 未找到时返回 {}
    """
    try:
        result = await asyncio.to_thread(_db_find_interaction, session_id)
        if result:
            return result
    except Exception as e:
        logger.warning(f"MySQL 查询 session_id={session_id} 失败: {e}")
    return {}
async def _find_expert_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    按 task_id 查找专家审核任务, 从 MySQL 查询 (唯一索引 O(1))
    :param task_id: 目标任务 ID
    :return: 找到的任务字典, 未找到时返回 None
    """
    try:
        return await asyncio.to_thread(_db_find_expert_task, task_id)
    except Exception as e:
        logger.warning(f"MySQL 查询 task_id={task_id} 失败: {e}")
        return None
async def _count_jsonl_lines(filepath: str) -> int:
    """
    统计 JSONL 文件中的有效行数 (非空行)
    高并发安全: 跳过 JSON 解析失败的行 (可能是写入中的不完整行)
    :param filepath: 目标文件路径
    :return: 有效行数, 文件不存在时返回 0
    """
    count = 0
    try:
        async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
            async for line in f:
                line_stripped = line.strip()
                if line_stripped:
                    # 验证是否为有效 JSON (高并发下可能读到写入中的不完整行)
                    try:
                        json.loads(line_stripped)
                        count += 1
                    except json.JSONDecodeError:
                        # 跳过无效行, 不影响统计
                        pass
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"读取文件失败 {filepath}: {e}")
    return count