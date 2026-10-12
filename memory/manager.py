"""
memory/manager.py — 三层记忆管理器
层级:
  1. 短期记忆 (ShortTermMemory): 会话内上下文, 滑动窗口
     - 存储: Redis List (chat:history:{session_id}), 无 Redis 时降级进程内 dict
     - TTL: 7 天 (滑动过期, 持续对话则续期)
     - 用途: 多轮对话上下文 (进程重启不丢失)
  2. 长期记忆 (LongTermMemory): 用户历史, 高频问题
     - 存储: Redis Hash
     - TTL: 7 天
     - 用途: 跨会话记忆, Agent 反思结果持久化
  3. 语义记忆 (SemanticMemory): 问题向量
     - 存储: Milvus (复用现有 collection)
     - TTL: 永久
     - 用途: 相似问题检测, 避免重复计算
设计要点:
  - 复用现有 redis_manager 和 milvus_manager, 不创建新连接
  - 异步友好, 所有 I/O 操作通过 asyncio.to_thread
  - 短期记忆使用 LRU 淘汰策略, 限制内存占用
"""
import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
logger = logging.getLogger("AgentApp")
# 短期记忆最大会话数 — 限制进程内存占用, 超出时通过 LRU 策略淘汰最久未访问的会话
MAX_SESSIONS = 1000
# 每个会话最大消息数 (滑动窗口) — 只保留最近 20 条对话轮次,
# 防止单个会话无限增长导致内存爆炸, 同时 20 条足以覆盖多轮对话上下文
MAX_MESSAGES_PER_SESSION = 20
# 短期记忆 TTL (秒) — 与长期记忆一致, 7 天无活动自动清除
SHORT_TERM_TTL = 7 * 24 * 3600   # 7 天
# 长期记忆 TTL (秒)
LONG_TERM_TTL = 7 * 24 * 3600    # 7 天
@dataclass
class ConversationTurn:
    """一轮对话"""
    # 对话历史的原子单元: 记录一次用户提问或助手回答,
    # 包含角色, 内容, 时间戳, 可选的元数据 (如意图分类, 检索来源等)
    role: str           # "user" / "assistant"
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
class ShortTermMemory:
    """
    短期记忆 — Redis 会话上下文

    存储结构: Redis List, key = chat:history:{session_id}
      - 每个元素是一条 JSON 序列化的 ConversationTurn
      - LPUSH 新消息到头部 (最新在前), LTRIM 保留最近 N 条 (滑动窗口)
      - EXPIRE 7 天, 每次写入续期 (滑动过期)
    这样服务重启后会话上下文依然存在, 解决"重启失忆"问题.

    降级策略: Redis 不可用时退回进程内 OrderedDict, 功能可用(重启丢失),
    保证记忆子系统故障不会拖垮主问答链路.
    """
    def __init__(
        self,
        redis_manager=None,
        max_messages: int = MAX_MESSAGES_PER_SESSION,
        ttl: int = SHORT_TERM_TTL,
    ):
        self._redis = redis_manager
        self._max_messages = max_messages
        self._ttl = ttl
        # 降级存储 (Redis 不可用时使用)
        self._fallback: OrderedDict[str, List[ConversationTurn]] = OrderedDict()

    def _client(self):
        """获取 Redis 客户端, 不可用时返回 None (触发降级)."""
        if self._redis is None:
            try:
                from redis_db import redis_manager
                self._redis = redis_manager
            except Exception as e:
                logger.warning(f"[ShortTermMemory] 无法导入 redis_manager: {e}")
                return None
        try:
            return self._redis.client
        except Exception:
            return None

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:history:{session_id}"

    @staticmethod
    def _serialize(turn: ConversationTurn) -> str:
        return json.dumps({
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "metadata": turn.metadata,
        }, ensure_ascii=False)

    @staticmethod
    def _deserialize(raw) -> Optional[ConversationTurn]:
        try:
            obj = json.loads(raw)
            return ConversationTurn(
                role=obj.get("role", ""),
                content=obj.get("content", ""),
                timestamp=obj.get("timestamp", 0.0),
                metadata=obj.get("metadata", {}) or {},
            )
        except Exception:
            return None

    # ---------- 降级实现 (进程内) ----------
    def _fallback_add(self, session_id: str, turn: ConversationTurn):
        if session_id not in self._fallback:
            if len(self._fallback) >= MAX_SESSIONS:
                self._fallback.popitem(last=False)
            self._fallback[session_id] = []
        else:
            self._fallback.move_to_end(session_id)
        turns = self._fallback[session_id]
        turns.append(turn)
        if len(turns) > self._max_messages:
            self._fallback[session_id] = turns[-self._max_messages:]

    def _fallback_get(self, session_id: str, last_n: int) -> List[ConversationTurn]:
        turns = self._fallback.get(session_id, [])
        return turns[-last_n:]

    # ---------- 对外接口 ----------
    def add(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """添加一轮对话"""
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        client = self._client()
        if client is None:
            self._fallback_add(session_id, turn)
            return
        try:
            key = self._key(session_id)
            pipe = client.pipeline()
            pipe.lpush(key, self._serialize(turn))
            pipe.ltrim(key, 0, self._max_messages - 1)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except Exception as e:
            logger.warning(f"[ShortTermMemory] Redis 写入失败, 降级内存: {e}")
            self._fallback_add(session_id, turn)

    def get(self, session_id: str, last_n: int = 10) -> List[ConversationTurn]:
        """获取会话历史 (最近 N 条, 按时间正序返回)"""
        client = self._client()
        if client is None:
            return self._fallback_get(session_id, last_n)
        try:
            # 头部是最新消息, 取前 last_n 条后反转成正序
            raw = client.lrange(self._key(session_id), 0, last_n - 1)
            turns = [t for t in (self._deserialize(x) for x in raw) if t]
            turns.reverse()
            return turns
        except Exception as e:
            logger.warning(f"[ShortTermMemory] Redis 读取失败, 降级内存: {e}")
            return self._fallback_get(session_id, last_n)

    def get_context_str(self, session_id: str, last_n: int = 5) -> str:
        """获取格式化的上下文字符串 (最近轮次在最后, 靠近当前问题)"""
        # 将最近 N 轮对话格式化为 "用户: xxx\n助手: xxx" 文本,
        # 直接注入 LLM Prompt 中作为多轮对话上下文.
        # 每条内容截断到 200 字符，避免历史对话占用过多 token.
        #
        # 排列顺序: 最早的轮次在最前, 最近的轮次在最后 (紧邻 <question> 标签)
        # 这样 LLM 在生成回答时, 最近轮次的注意力权重最高 (proximity bias),
        # 自然会优先关联最近的对话 (如最近问的"头疼" > 更早问的"胃口不舒服"),
        # 同时更早的历史也不会被丢弃, 仍然作为背景上下文保留.
        turns = self.get(session_id, last_n)
        if not turns:
            return ""
        lines = []
        for turn in turns:
            prefix = "用户" if turn.role == "user" else "助手"
            lines.append(f"{prefix}: {turn.content[:200]}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        """清除会话记忆"""
        self._fallback.pop(session_id, None)
        client = self._client()
        if client is None:
            return
        try:
            client.delete(self._key(session_id))
        except Exception as e:
            logger.warning(f"[ShortTermMemory] Redis 删除失败: {e}")

    @property
    def session_count(self) -> int:
        return len(self._fallback)
class LongTermMemory:
    """
    长期记忆 — Redis Hash 存储
    复用现有 redis_manager, 不创建新连接
    Key 格式:
      memory:user:{user_id} → Hash {
        "last_topics": json,        # 最近讨论的话题
        "preferences": json,        # 用户偏好
        "reflection_history": json, # 反思历史 (Agent 学习)
      }
    """
    def __init__(self, redis_manager=None):
        self._redis = redis_manager
    def _ensure_redis(self):
        """懒加载 Redis 连接 — 首次调用时才导入 redis_manager 单例"""
        # 延迟导入模式: redis_manager 在 redis_db.py 中以模块级单例创建,
        # 若在 __init__ 中直接导入可能引发循环依赖 (memory → redis_db → app → memory)
        # 首次调用 save/load 时才触发导入, 此时所有模块已完成初始化
        if self._redis is None:
            from redis_db import redis_manager
            self._redis = redis_manager
    async def save(self, user_id: str, key: str, value: Any):
        """保存长期记忆"""
        self._ensure_redis()
        # Redis Key 格式: memory:user:{user_id}，使用 Hash 结构存储,
        # 每个 user 一个 Hash, 内含多个 field (如 reflection_history, last_topics 等)
        redis_key = f"memory:user:{user_id}"
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            # asyncio.to_thread: 将同步的 Redis 操作放入线程池执行,
            # 不阻塞事件循环 (Redis Python 客户端本身是同步的)
            await asyncio.to_thread(
                self._redis.client.hset, redis_key, key, serialized
            )
            # 每次写入后刷新 TTL — 滑动过期策略:
            #     1: 只要用户持续使用, 记忆就不会过期
            #     2: 7天无活动后自动清除, 释放 Redis 内存
            await asyncio.to_thread(
                self._redis.client.expire, redis_key, LONG_TERM_TTL
            )
        except Exception as e:
            logger.warning(f"[LongTermMemory] 保存失败: {e}")
    async def load(self, user_id: str, key: str) -> Optional[Any]:
        """加载长期记忆"""
        self._ensure_redis()
        redis_key = f"memory:user:{user_id}"
        try:
            # 从 Redis Hash 中读取指定 field 的值
            # hget 返回 bytes 或 None, json.loads 反序列化为 Python 对象
            raw = await asyncio.to_thread(
                self._redis.client.hget, redis_key, key
            )
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"[LongTermMemory] 加载失败: {e}")
        return None
    async def save_reflection(self, user_id: str, query: str, score: float, feedback: str):
        """保存反思结果 (用于 Agent 学习)"""
        # 从 Redis 加载现有的反思历史列表, 若不存在则初始化空列表
        history = await self.load(user_id, "reflection_history") or []
        # 追加本次反思记录:
        # - query[:100]: 截断到 100 字符, 节省 Redis 内存 (只需记住问题大意)
        # - feedback[:200]: 截断到 200 字符, 保留关键的改进建议
        # - timestamp: 记录时间, 用于后续按时间排序或过期淘汰
        history.append({
            "query": query[:100],
            "score": score,
            "feedback": feedback[:200],
            "timestamp": time.time(),
        })
        # 只保留最近 50 条 — 固定大小队列, 防止列表无限增长
        # 50 条足以覆盖一周左右的使用频率, 同时控制 Redis 单 key 大小
        history = history[-50:]
        await self.save(user_id, "reflection_history", history)
# ===========================================================================
# MemoryManager — 统一记忆管理器 (门面模式 / Facade)
# 外部组件 (Orchestrator, Agent) 统一通过此类访问记忆,
# 无需关心底层是进程内 dict (短期) 还是 Redis Hash (长期),
# 实现了存储细节的封装和解耦
# ===========================================================================
class MemoryManager:
    """
    统一记忆管理器 — 封装三层记忆的读写
    用法:
      memory = MemoryManager()
      memory.add_turn(session_id, "user", "高血压怎么治疗?")
      context = memory.get_context(session_id)
      await memory.save_long_term(user_id, "topics", ["高血压"])
    """
    def __init__(self, redis_manager=None):
        self.short_term = ShortTermMemory(redis_manager=redis_manager)
        self.long_term = LongTermMemory(redis_manager=redis_manager)
    def add_turn(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """添加对话轮次到短期记忆"""
        self.short_term.add(session_id, role, content, metadata)
    def get_context(self, session_id: str, last_n: int = 5) -> str:
        """获取短期记忆上下文"""
        return self.short_term.get_context_str(session_id, last_n)
    def get_context_digest(self, session_id: str, last_n: int = 5) -> str:
        """
        获取历史上下文的短哈希摘要 — 专供缓存 Key 隔离使用.
        多轮场景下, 同一句问题(如"它的副作用是什么")在不同会话上下文里
        应得到不同答案. 若缓存 Key 只按 query 文本哈希, 会发生跨会话"串话"
        (A 在阿莫西林上下文的答案被 B 在布洛芬上下文命中).
        本方法把会话历史压成一个 12 位十六进制摘要, 由调用方拼到缓存 Key 文本里,
        实现"按上下文隔离缓存"的同时仍保留缓存能力(相同历史+相同问题仍可命中).
        关键不变量: 无历史(首轮 / 无 session_id)时返回空串, 缓存 Key 与升级前完全一致,
        因此无状态 / 压测路径的缓存命中行为不受任何影响.
        :return: 12 位 hex 摘要, 或空字符串(无历史时)
        """
        if not session_id:
            return ""
        context = self.short_term.get_context_str(session_id, last_n)
        if not context:
            return ""
        return hashlib.md5(context.encode("utf-8")).hexdigest()[:12]
    def get_turns(self, session_id: str, last_n: int = 10) -> List[ConversationTurn]:
        """获取对话历史"""
        return self.short_term.get(session_id, last_n)
    async def save_long_term(self, user_id: str, key: str, value: Any):
        """保存到长期记忆"""
        await self.long_term.save(user_id, key, value)
    async def load_long_term(self, user_id: str, key: str) -> Optional[Any]:
        """从长期记忆加载"""
        return await self.long_term.load(user_id, key)
    async def save_reflection(self, user_id: str, query: str, score: float, feedback: str):
        """保存反思结果"""
        await self.long_term.save_reflection(user_id, query, score, feedback)
    def clear_session(self, session_id: str):
        """清除会话记忆"""
        self.short_term.clear(session_id)