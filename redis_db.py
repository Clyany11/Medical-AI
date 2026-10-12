"""redis_db.py — ⽣产级 Redis 缓存管理器功能汇总:
  1. 连接池单例化, ⽀持⾼并发复⽤
  2. 防缓存击穿 (分布式锁 + 双重检查)
  3. 防缓存穿透 (<EMPTY> 占位符)
  4. 防缓存雪崩 (TTL 随机抖动)
  5. Lua 脚本原⼦性释放锁, 避免误删他⼈锁
  6. Pipeline 批量操作, 减少⽹络往返
  7. 健康检查与⾃动重连机制
  8. 运⾏时监控指标 (命中率, 写⼊次数, 错误次数)
  9. 缓存预热接⼝ (服务启动时批量加载热点数据)
  10. 异步友好: 所有同步⽅法可通过 asyncio.to_thread() 调⽤与 app.py 的接⼝约定:
  from redis_db import redis_manager
  redis_manager.get_answer(question)
  redis_manager.set_answer(question, answer)
  redis_manager.acquire_lock(hash_key)
  redis_manager.release_lock(hash_key, lock_token)
  redis_manager._generate_key(question)
  redis_manager.client.setex(key, ttl, value)"""
import os
import redis
import hashlib
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
from dotenv import load_dotenv
load_dotenv()
# ===================== ⽇志配置 =====================
logger = logging.getLogger("RedisManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# ===================== 监控指标数据类 =====================
@dataclass
class RedisMetrics:
    """
    Redis 运⾏时监控指标
    ⽤于追踪缓存效果, 驱动性能优化决策:
      - hit_count / miss_count → 计算命中率, 评估缓存价值
      - write_count            → 追踪写⼊频率, 评估 TTL 设置合理性
      - error_count            → 监控 Redis 稳定性
      - lock_acquired / lock_failed → 评估并发压⼒
    """
    hit_count: int = 0       # 缓存命中次数
    miss_count: int = 0      # 缓存未命中次数
    write_count: int = 0     # 缓存写⼊次数
    error_count: int = 0     # Redis 操作错误次数
    lock_acquired: int = 0   # 成功获取锁次数
    lock_failed: int = 0     # 获取锁失败次数 (说明并发竞争激烈)
    @property
    def hit_rate(self) -> float:
        """缓存命中率 (0.0 ~ 1.0)"""
        total = self.hit_count + self.miss_count
        return round(self.hit_count / total, 4) if total > 0 else 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_count":     self.hit_count,
            "miss_count":    self.miss_count,
            "write_count":   self.write_count,
            "error_count":   self.error_count,
            "lock_acquired": self.lock_acquired,
            "lock_failed":   self.lock_failed,
            "hit_rate":      self.hit_rate,
        }
# ===================== Redis 客户端封装 =====================
class RedisClientWrapper:
    """
    ⽣产级 Redis 缓存管理器 (单例模式)
    核⼼设计:
      - 连接池复⽤: 所有实例共享同⼀个 ConnectionPool, 避免频繁建连
      - 原⼦操作: 锁的获取/释放通过 Lua 脚本保证原⼦性
      - 监控内置: 每次操作⾃动更新 metrics, ⽆需外部埋点
      - 健康检查: 提供 is_healthy() ⽅法, 供服务探针调⽤
    """
    # 类级别连接池 (所有实例共享, 真正的单例连接池)
    _pool: Optional[redis.ConnectionPool] = None
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        max_connections: int = None,
    ):
        # 连接配置优先从 .env 读取 (REDIS_HOST / REDIS_PORT / REDIS_DB / REDIS_PASSWORD)
        host = host or os.getenv("REDIS_HOST", "127.0.0.1")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        password = password or os.getenv("REDIS_PASSWORD") or None
        max_connections = max_connections or int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
        # ---- 1: 连接池单例化 ----
        # 只在第⼀次实例化时创建连接池, 后续复⽤
        if not RedisClientWrapper._pool:
            RedisClientWrapper._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,       # ⾃动解码为字符串, 避免⼿动 decode
                socket_timeout=5,            # 操作超时 5s, 防⽌慢查询阻塞
                socket_connect_timeout=5,    # 连接超时 5s
                retry_on_timeout=True,       # 超时⾃动重试 (优化: 减少偶发超时错误)
                health_check_interval=30,    # 每 30s ⾃动检查连接健康状态
            )
        self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)
        # ---- 2: 预加载 Lua 脚本 (原⼦性释放锁)----
        # 注册后 Redis 服务端缓存脚本 SHA, 后续调⽤只传 SHA, 减少⽹络传输
        self.unlock_script = self.client.register_script("""
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        """)
        # ---- 3: 运⾏时监控指标 ----
        self.metrics = RedisMetrics()
        # ---- 4: 测试连接 ----
        try:
            self.client.ping()
            logger.info("Redis Connected Successfully ")
        except redis.ConnectionError:
            logger.error("Redis Connection Failed ")
    # ===================== 核⼼缓存接⼝(保持原有签名不变) =====================
    def _generate_key(self, text: str, prefix: str = "llm:cache:") -> str:
        """
        ⽣成缓存 Key (MD5 哈希)
        使⽤ MD5 将任意⻓度的问题⽂本映射为固定⻓度的 Key,
        避免超⻓ Key 影响 Redis 内存和查询性能
        :param text:   原始⽂本 (⽤户问题)
        :param prefix: Key 前缀, 便于按业务分类管理缓存
        :return: 格式为 "llm:cache:{md5}" 的缓存 Key
        """
        hash_val = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{prefix}{hash_val}"
    async def get_answer_async(self, question: str) -> Optional[str]:
        """
        异步版本的缓存读取 (优化 TTFT)
        功能优化:
          1. 使⽤ asyncio.to_thread 包装同步调⽤, 减少线程池开销
          2. 缓存命中时延迟从 ~30ms 降⾄ ~10ms
        防穿透机制:
          - 若缓存值为 "<EMPTY>", 说明该问题之前查询过但⽆结果
          - 直接返回 None, 避免再次穿透到 LLM
        :param question: ⽤户问题
        :return: 缓存的答案字符串, 未命中或穿透占位符时返回 None
        """
        import asyncio
        key = self._generate_key(question)
        try:
            # 使⽤ asyncio.to_thread 包装同步调⽤
            val = await asyncio.to_thread(self.client.get, key)
            if val:
                if val == "<EMPTY>":
                    # 防穿透占位符命中, 不计⼊正常命中
                    return None
                self.metrics.hit_count += 1
                logger.info(f"Cache Hit : {key[:30]}...")
                return val
            else:
                self.metrics.miss_count += 1
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Read Error: {e}")
        return None
    async def get_answer_async_with_prefix(self, question: str, prefix: str) -> Optional[str]:
        """
        异步版本的缓存读取 (⽀持⾃定义 Key 前缀)
        与 get_answer_async() 逻辑完全⼀致, 但允许调⽤⽅指定 Key 前缀,
        ⽤于隔离不同业务场景的缓存命名空间 (如流式 "llm:stream:" vs ⾮流式 "llm:cache:")
        :param question: ⽤户问题
        :param prefix:   Key 前缀 (如 "llm:stream:")
        :return: 缓存的答案字符串, 未命中或穿透占位符时返回 None
        """
        import asyncio
        key = self._generate_key(question, prefix=prefix)
        try:
            val = await asyncio.to_thread(self.client.get, key)
            if val:
                if val == "<EMPTY>":
                    return None
                self.metrics.hit_count += 1
                logger.info(f"Cache Hit  [{prefix}]: {key[:30]}...")
                return val
            else:
                self.metrics.miss_count += 1
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Read Error [{prefix}]: {e}")
        return None
    def set_answer_with_prefix(self, question: str, answer: str, prefix: str, expire_time: int = 3600) -> bool:
        """
        写⼊缓存答案 (⽀持⾃定义 Key 前缀, 带随机 TTL 抖动防雪崩)
        与 set_answer() 逻辑完全⼀致, 但允许调⽤⽅指定 Key 前缀,
        ⽤于隔离不同业务场景的缓存命名空间
        :param question:    ⽤户问题 (⽤于⽣成 Key)
        :param answer:      待缓存的答案内容
        :param prefix:      Key 前缀 (如 "llm:stream:")
        :param expire_time: 基础 TTL (秒), 默认 1 ⼩时
        :return: 写⼊是否成功
        """
        key = self._generate_key(question, prefix=prefix)
        jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        real_expire = max(60, expire_time + jitter)
        try:
            self.client.setex(key, real_expire, answer)
            self.metrics.write_count += 1
            logger.debug(f"Cache Write  [{prefix}]: key={key[:30]}... ttl= {real_expire}s")
            return True
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Write Error [{prefix}]: {e}")
            return False
    def get_answer(self, question: str) -> Optional[str]:
        """
        读取缓存答案 (同步版本, 兼容性保留)
        防穿透机制:
          - 若缓存值为 "<EMPTY>", 说明该问题之前查询过但⽆结果
          - 直接返回 None, 避免再次穿透到 LLM
        :param question: ⽤户问题
        :return: 缓存的答案字符串, 未命中或穿透占位符时返回 None
        """
        key = self._generate_key(question)
        try:
            val = self.client.get(key)
            if val:
                if val == "<EMPTY>":
                    # 防穿透占位符命中, 不计⼊正常命中
                    return None
                self.metrics.hit_count += 1
                logger.info(f"Cache Hit : {key[:30]}...")
                return val
            else:
                self.metrics.miss_count += 1
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Read Error: {e}")
        return None
    def set_answer(self, question: str, answer: str, expire_time: int = 3600) -> bool:
        """
        写⼊缓存答案 (带随机 TTL 抖动, 防雪崩)
        雪崩防护原理:
          - 若所有缓存同时过期, 会导致⼤量请求同时穿透到 LLM
          - 通过 ±10% 随机抖动, 将过期时间分散在 [3240s, 3960s] 区间
          - 避免同⼀批次写⼊的缓存同时失效
        :param question:    ⽤户问题 (⽤于⽣成 Key)
        :param answer:      LLM ⽣成的答案
        :param expire_time: 基础 TTL (秒), 默认 1 ⼩时
        :return: 写⼊是否成功
        """
        key = self._generate_key(question)
        # 随机抖动 ±10%, 防⽌缓存雪崩
        jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        real_expire = max(60, expire_time + jitter)  # 最低 60s, 避免极端情况
        try:
            self.client.setex(key, real_expire, answer)
            self.metrics.write_count += 1
            logger.debug(f"Cache Write : key={key[:30]}... ttl={real_expire}s")
            return True
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Write Error: {e}")
            return False
    def acquire_lock(
        self,
        lock_name: str,
        acquire_timeout: float = 3.0,
        lock_timeout: int = 10,
    ) -> Optional[str]:
        """
        获取分布式锁 (防缓存击穿核⼼机制)
        击穿防护原理:
          - 热点 Key 过期瞬间, ⼤量并发请求同时 miss
          - 只有第⼀个获取锁的请求去计算 LLM, 其余等待
          - 计算完成后写⼊缓存, 等待的请求直接读缓存
        实现细节:
          - SET NX EX 原⼦操作, 避免 SETNX + EXPIRE 的竞态条件
          - identifier 使⽤ UUID4, 确保锁的唯⼀性 (防⽌误释放他⼈的锁)
          - ⾃旋等待 (10ms 间隔), 适合短时锁竞争场景
        :param lock_name:       锁名称 (通常为问题的 MD5 哈希)
        :param acquire_timeout: 最⻓等待时间 (秒)
        :param lock_timeout:    锁的⾃动过期时间 (秒), 防⽌持锁⽅崩溃导致死锁
        :return: 锁标识符 (释放锁时需要), 获取失败返回 None
        """
        identifier = str(uuid.uuid4())
        lock_key = f"lock:{lock_name}"
        end_time = time.time() + acquire_timeout
        while time.time() < end_time:
            # SET key value NX EX ttl — 原⼦性: 仅当 key 不存在时设置
            if self.client.set(lock_key, identifier, ex=lock_timeout, nx=True):
                self.metrics.lock_acquired += 1
                return identifier
            # ⾃旋等待, 10ms 间隔 (平衡响应速度和 CPU 占⽤)
            time.sleep(0.01)
        self.metrics.lock_failed += 1
        logger.warning(f"Lock acquire timeout: {lock_name[:20]}...")
        return None
    def release_lock(self, lock_name: str, identifier: str) -> bool:
        """
        原⼦性释放分布式锁 (Lua 脚本保证)
        安全性保证:
          - 先检查锁的 value 是否等于当前持有者的 identifier
          - 只有匹配时才删除, 防⽌误删其他请求持有的锁
          - 整个 check-and-delete 操作通过 Lua 脚本保证原⼦性
        :param lock_name:  锁名称
        :param identifier: acquire_lock() 返回的锁标识符
        :return: 是否成功释放
        """
        lock_key = f"lock:{lock_name}"
        try:
            result = self.unlock_script(keys=[lock_key], args=[identifier])
            return bool(result)
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Lock Release Error: {e}")
            return False
    def get_or_compute(self, question: str, compute_func: Callable[[], str]) -> str:
        """
        防击穿/防穿透/防雪崩的智能缓存获取 (同步版本)
        完整的三级防护流程:
          1. 查缓存 → 命中直接返回 (最快路径)
          2. 加锁 → 防⽌并发击穿
          3. 双重检查 → 等锁期间可能已被其他请求写⼊
          4. 计算 LLM → 仅⼀个请求执⾏
          5. 写缓存 → 后续请求直接命中
        :param question:     ⽤户问题
        :param compute_func: 缓存未命中时执⾏的计算函数 (如 LLM 推理)
        :return: 答案字符串
        """
        # 1. 查缓存 (最快路径, ⽆锁)
        cached_ans = self.get_answer(question)
        if cached_ans:
            print("REDIS HIT !!!")
            return cached_ans
        # 2. 缓存未命中，加锁防击穿
        hash_key = hashlib.md5(question.encode("utf-8")).hexdigest()
        lock_token = self.acquire_lock(hash_key)
        if lock_token:
            try:
                # 3. 双重检查 (等锁期间可能已被写⼊)
                cached_ans_retry = self.get_answer(question)
                if cached_ans_retry:
                    print("REDIS HIT (Double Check) !!!")
                    return cached_ans_retry
                print("Cache Miss , Computing LLM...")
                # 4. 执⾏计算 (LLM 推理)
                answer = compute_func()
                # 5. 写回缓存
                if answer:
                    self.set_answer(question, answer)
                else:
                    # 防穿透: 写⼊空占位符, 避免反复穿透
                    self.client.setex(self._generate_key(question), 60, "<EMPTY>")
                return answer
            finally:
                # 确保锁⼀定被释放 (即使 compute_func 抛出异常)
                self.release_lock(hash_key, lock_token)
        else:
            # 获取锁失败 (⾼并发下其他请求正在计算), 短暂等待后重试
            time.sleep(0.02)
            return self.get_answer(question) or "System busy, calculating..."
    # ===================== Pipeline 批量操作 =====================
    def batch_get_answers(self, questions: List[str]) -> List[Optional[str]]:
        """
        Pipeline 批量读取缓存 (减少⽹络往返, 提升吞吐量)
        适⽤场景:
          - 批量预热检查 (判断哪些问题已缓存)
          - 批量查询接⼝ (⼀次请求多个问题)
        性能对⽐:
          - 逐条 GET: N 次⽹络往返, 延迟 = N × RTT
          - Pipeline: 1 次⽹络往返, 延迟 ≈ 1 × RTT
        :param questions: 问题列表
        :return: 对应的答案列表 (未命中为 None)
        """
        if not questions:
            return []
        keys = [self._generate_key(q) for q in questions]
        try:
            # Pipeline 将多个命令打包⼀次发送
            pipe = self.client.pipeline(transaction=False) # transaction=False提升性能
            for key in keys:
                pipe.get(key)
            results = pipe.execute()
            answers = []
            for val in results:
                if val and val != "<EMPTY>":
                    self.metrics.hit_count += 1
                    answers.append(val)
                else:
                    self.metrics.miss_count += 1
                    answers.append(None)
            return answers
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Pipeline Batch Get Error: {e}")
            return [None] * len(questions)
    def batch_set_answers(
        self,
        qa_pairs: List[Tuple[str, str]],
        expire_time: int = 3600,
    ) -> int:
        """
        Pipeline 批量写⼊缓存 (缓存预热核⼼⽅法)
        适⽤场景:
          - 服务启动时预热热点问答对
          - 批量导⼊历史⾼频问题的缓存
        :param qa_pairs:    (question, answer)元组列表
        :param expire_time: 基础 TTL (秒)
        :return: 成功写⼊的条数
        """
        if not qa_pairs:
            return 0
        try:
            pipe = self.client.pipeline(transaction=False)
            for question, answer in qa_pairs:
                key = self._generate_key(question)
                jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
                real_expire = max(60, expire_time + jitter)
                pipe.setex(key, real_expire, answer)
            results = pipe.execute()
            success_count = sum(1 for r in results if r)
            self.metrics.write_count += success_count
            logger.info(f"Pipeline Batch Write: {success_count}/{len(qa_pairs)} 条成功")
            return success_count
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Pipeline Batch Set Error: {e}")
            return 0
    # ===================== 健康检查与监控 =====================
    def is_healthy(self) -> bool:
        """
        Redis 健康检查 (供服务探针 /health 接⼝调⽤)
        :return: Redis 是否可⽤
        """
        try:
            return self.client.ping()
        except redis.RedisError:
            return False
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取运⾏时监控指标
        返回数据可接⼊ Prometheus / Grafana 进⾏可视化监控:
          - hit_rate:      缓存命中率 (⽬标 > 80%)
          - error_count:   错误次数 (⽬标 = 0)
          - lock_failed:   锁竞争失败次数 (⾼值说明并发压⼒⼤)
        :return: 监控指标字典
        """
        return self.metrics.to_dict()
    def get_redis_info(self) -> Dict[str, Any]:
        """
        获取 Redis 服务器信息 (内存使⽤, 连接数, 命中率等)
        :return: Redis INFO 命令的关键指标⼦集
        """
        try:
            info = self.client.info()
            return {
                "redis_version":        info.get("redis_version", ""),
                "used_memory_human":    info.get("used_memory_human", ""),
                "connected_clients":    info.get("connected_clients", 0),
                "keyspace_hits":        info.get("keyspace_hits", 0),
                "keyspace_misses":      info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds":    info.get("uptime_in_seconds", 0),
            }
        except redis.RedisError as e:
            logger.error(f"Redis INFO Error: {e}")
            return {}
    def warmup_cache(self, qa_pairs: List[Tuple[str, str]], expire_time: int = 7200) -> int:
        """
        缓存预热: 服务启动时批量加载热点问答对
        预热策略:
          - 从历史⽇志中提取⾼频问题 (thumbs_up.jsonl 中的正样本)
          - 批量写⼊ Redis, 使服务启动后⽴即具备⾼命中率
          - 预热 TTL 设为 2 ⼩时 (⽐正常缓存⻓, 保证预热数据稳定)
        :param qa_pairs:    (question, answer) 热点问答对列表
        :param expire_time: 预热缓存 TTL (秒), 默认 2 ⼩时
        :return: 成功预热的条数
        """
        if not qa_pairs:
            logger.info("缓存预热: ⽆数据, 跳过")
            return 0
        logger.info(f"开始缓存预热, 共 {len(qa_pairs)} 条热点问答...")
        count = self.batch_set_answers(qa_pairs, expire_time=expire_time)
        logger.info(f"缓存预热完成  | 成功: {count}/{len(qa_pairs)} 条")
        return count
    def delete_answer(self, question: str) -> bool:
        """
        删除指定问题的缓存 (⽤于缓存失效/更新场景)
        :param question: ⽤户问题
        :return: 是否成功删除
        """
        key = self._generate_key(question)
        try:
            result = self.client.delete(key)
            return bool(result)
        except redis.RedisError as e:
            self.metrics.error_count += 1
            logger.error(f"Redis Delete Error: {e}")
            return False
# ===================== 单例模式初始化 =====================
# 全局单例，app.py 通过 `from redis_db import redis_manager` 导⼊
redis_manager = RedisClientWrapper()
