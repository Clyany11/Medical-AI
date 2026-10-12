"""neo4j_db.py — Neo4j 图数据库连接管理器

设计要点:
  - 官方异步 driver (neo4j.AsyncGraphDatabase), 不阻塞事件循环
  - 模块级单例, 与 milvus_db / redis_db 保持一致的接入约定
  - 只读查询 (execute_read): 写操作会被 Neo4j 服务端拒绝, 防止误改图数据
  - 未配置 NEO4J_URI 时静默降级, 不影响主服务启动

接入约定 (与 redis_db 相同):
    from neo4j_db import neo4j_manager
    await neo4j_manager.initialize()      # 由 app.py lifespan 调用
    rows = await neo4j_manager.run_read("MATCH (n) RETURN n LIMIT 5")
    await neo4j_manager.close()           # 由 app.py lifespan 调用
"""
import os
import logging
from typing import Any, Dict, List, Optional

import neo4j
from dotenv import load_dotenv

# 必须在读取任何环境变量之前调用
load_dotenv()

logger = logging.getLogger("Neo4jManager")

# ===================== 连接配置 (.env) =====================
NEO4J_URI = os.getenv("NEO4J_URI", "")                     # 如 bolt://localhost:7687
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
# 单次查询最多返回的行数, 防止全图扫描把结果撑爆
NEO4J_MAX_ROWS = int(os.getenv("NEO4J_MAX_ROWS", "100"))


class Neo4jManager:
    """
    Neo4j 客户端封装 (单例)

    用法:
        neo4j_manager = Neo4jManager()
        await neo4j_manager.initialize()
        result = await neo4j_manager.run_read("MATCH (d:Disease) RETURN d.name LIMIT 10")
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        # 参数优先, 其次 .env, 最后内置默认值
        self.uri = uri if uri is not None else NEO4J_URI
        self.user = user if user is not None else NEO4J_USER
        self.password = password if password is not None else NEO4J_PASSWORD
        self.database = database if database is not None else NEO4J_DATABASE
        self._driver: Optional[neo4j.AsyncDriver] = None
        self._initialized = False

    @property
    def configured(self) -> bool:
        """是否提供了连接地址 (为空则视为未接入, 直接跳过)"""
        return bool(self.uri)

    async def initialize(self) -> None:
        """创建 driver 并验证连通性 (由 lifespan 启动阶段调用)"""
        if self._initialized:
            return
        if not self.configured:
            logger.warning("Neo4j 未配置 (NEO4J_URI 为空), 跳过初始化, 图查询功能不可用")
            return
        self._driver = neo4j.AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        # 主动验证: 连不上会在启动日志里立刻暴露, 而不是等到第一次查询
        await self._driver.verify_connectivity()
        self._initialized = True
        logger.info(f"Neo4j 连接成功 ✅ ({self.uri} | database={self.database})")

    async def close(self) -> None:
        """释放连接 (由 lifespan 关闭阶段调用)"""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        self._initialized = False
        logger.info("Neo4j 连接已关闭")

    def is_healthy(self) -> bool:
        """健康检查 (供 /health 探针使用)"""
        return self._initialized

    async def run_read(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行只读 Cypher 查询

        :param cypher: Cypher 语句 (只读; 含写操作会被 Neo4j 服务端拒绝)
        :param params: 参数化查询的参数字典 (防注入, 优先用它而不是拼字符串)
        :param limit:  返回行数上限, 缺省用 NEO4J_MAX_ROWS
        :return: {"rows": [...], "count": n, "truncated": bool} 或 {"rows": [], "error": "..."}
        """
        if not self._initialized:
            return {"rows": [], "count": 0, "error": "neo4j not initialized"}
        cypher = (cypher or "").strip().rstrip(";")
        if not cypher:
            return {"rows": [], "count": 0, "error": "empty cypher"}

        max_rows = limit or NEO4J_MAX_ROWS
        try:
            async with self._driver.session(
                database=self.database,
                default_access_mode=neo4j.READ_ACCESS,
            ) as session:
                result = await session.run(cypher, params or {})
                rows: List[Dict[str, Any]] = []
                async for record in result:
                    rows.append(record.data())
                    # 客户端截断: 超出上限就停止消费结果游标
                    if len(rows) >= max_rows:
                        break
                return {
                    "rows": rows,
                    "count": len(rows),
                    "truncated": len(rows) >= max_rows,
                }
        except neo4j.exceptions.Neo4jError as e:
            # 语法错误 / 权限不足 / 只读模式下写入 等, 都归到这里
            logger.error(f"Neo4j 查询失败: {e}")
            return {"rows": [], "count": 0, "error": str(e)}
        except Exception as e:
            logger.error(f"Neo4j 查询异常: {e}")
            return {"rows": [], "count": 0, "error": str(e)}


# ===================== 单例 =====================
# 全局单例, 通过 `from neo4j_db import neo4j_manager` 使用
neo4j_manager = Neo4jManager()
