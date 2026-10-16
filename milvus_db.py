import os
import json
import asyncio
import time
import httpx
from typing import List, Optional, Dict, Any
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
from vectors import OpenAICompatEmbeddings, create_embedding_client
import logging
from dotenv import load_dotenv
# 必须在读取任何环境变量之前调用, 否则 .env 里的 MILVUS_* 不会被加载
load_dotenv()
# ===================== 全局配置 (⾼性能调优参数) =====================
# 1. Milvus检索性能参数 (压测优化版)
MILVUS_PERF_CONFIG = {
    "default_top_k": 30,
    # collection 名可用 MILVUS_COLLECTION 覆盖 (挂载旧库时需指向其实际集合名)
    "collection_name": os.getenv("MILVUS_COLLECTION", "agent_knowledge"),
    "consistency_level": "Bounded",
    # 稠密索引检索参数 (必须与实际建索引时的类型一致, 否则参数被忽略、退回默认值)
    # 注意: Milvus Lite 不支持 HNSW, 实际建的是 IVF_FLAT, 故按 IVF_FLAT 传参
    "dense_search_params": {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nprobe": 16}
    },
    # BM25检索参数
    "bm25_search_params": {
        "metric_type": "BM25",
        "params": {}
    },
    # RRF融合参数
    "ranker_params": {"k": 60},
    "ranker_type": "rrf"}
# 2. 古籍 Collection 配置 (与现代医学知识隔离)
ANCIENT_BOOKS_CONFIG = {
    "default_top_k": 30,
    "collection_name": os.getenv("MILVUS_ANCIENT_COLLECTION", "ancient_books_knowledge"),
    "consistency_level": "Eventually",
    "dense_search_params": {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nprobe": 16}
    },
    "bm25_search_params": {
        "metric_type": "BM25",
        "params": {}
    },
    "ranker_params": {"k": 60},
    "ranker_type": "rrf"}
# 3. Rerank 精排配置
# 作用: 在混合检索 (dense + BM25 → RRF 融合) 之后, 用 Cross-Encoder 对候选文档重新
#       打分排序, 取 top_n 条送入大模型. RRF 只按"名次"融合 (无法判断语义相关性),
#       rerank 则同时看 query 与文档的语义交互, 能显著提升送入 LLM 的资料精度.
# 设计: 默认复用 embedding 厂商 (智谱 GLM) 的 /rerank 接口, 无需新增凭证;
#       换厂商只需用 RERANK_API_KEY / RERANK_BASE_URL / RERANK_MODEL 覆盖 (接口路径统一为 {base}/rerank).
RERANK_CONFIG = {
    # 总开关: 关闭后 search() 行为与接入前一致 (直接返回 default_top_k 条)
    "enabled": os.getenv("RERANK_ENABLED", "true").lower() == "true",
    # 粗召回候选数 (送入 rerank 的条数): 应 >= top_n, 稍大些才能让精排有选择余地
    "candidate_k": int(os.getenv("RERANK_CANDIDATE_K", "60")),
    # 精排后保留条数 (最终送入大模型的文档数)
    "top_n": int(os.getenv("RERANK_TOP_N", "10")),
    "model": os.getenv("RERANK_MODEL", "rerank"),
    # 复用嵌入凭证; 若厂商不同, 用 RERANK_API_KEY 单独指定
    "api_key": os.getenv("RERANK_API_KEY") or os.getenv("EMBEDDING_API_KEY", ""),
    "base_url": (os.getenv("RERANK_BASE_URL")
                 or os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")).rstrip("/"),
    "timeout": float(os.getenv("RERANK_TIMEOUT", "10")),
}
# 配置⽇志 (添加耗时, 请求ID, 便于压测排查)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger("MilvusClientWrapper")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 环境变量加载
# 注意: 变量名不用 MILVUS_URI, 因为 pymilvus 会读取同名环境变量并强制要求 http(s) 格式
milvus_uri = os.getenv("MILVUS_ENDPOINT")
milvus_user = os.getenv("MILVUS_USER", "root")
milvus_passwd = os.getenv("MILVUS_PASSWORD", "Milvus")
# 初始化 Embedding 客户端 (OpenAI 兼容, 单例复⽤, 厂商由 EMBEDDING_BASE_URL 决定)
client = create_embedding_client()


def build_connection_args(uri: str, user: str = "", password: str = "") -> Dict[str, Any]:
    """
    构建 Milvus 连接参数 (兼容 远程服务 与 本地文件 Milvus Lite)

    Milvus Lite (uri 为本地 .db 文件路径) 不支持账号密码,
    传入 user/password 会直接报错, 因此仅在 http(s) 远程地址时才附加认证信息.
    """
    args: Dict[str, Any] = {"uri": uri}
    if uri.startswith("http://") or uri.startswith("https://"):
        if user:
            args["user"] = user
        if password:
            args["password"] = password
    return args
class MilvusClientWrapper:
    """
    ⾼性能Milvus客户端封装 (适配⾼并发压测 + 多 Collection ⽀持)
    核⼼优化: 连接复⽤, 检索参数调优, 并发安全, 耗时监控, 资源管理
    多 Collection ⽀持:
      - 通过 collection_config 参数指定不同的 collection 配置
      - 每个实例独⽴管理⼀个 collection (不再使⽤单例模式)
      - ⽀持现代医学知识 (agent_knowledge) 和古籍知识 (ancient_books_knowledge) 隔离
    """
    _init_lock: asyncio.Lock = asyncio.Lock()                    # 异步锁确保单例安全
    _search_params_cache: Optional[List[Dict[str, Any]]] = None  # 缓存检索参数, 避免重复创建
    def __init__(self, client_embedding, uri: str, collection_config: Optional[Dict[str, Any]] = None):
        """
        初始化 Milvus 客户端
        :param client_embedding: OpenAI 兼容的 embedding 客户端 (⽤于向量化)
        :param uri: Milvus 连接地址
        :param collection_config: Collection 配置字典 (默认使⽤ MILVUS_PERF_CONFIG)
        """
        # 基础配置
        self.uri = uri
        self.config = collection_config or MILVUS_PERF_CONFIG
        self.collection_name = self.config["collection_name"]
        self.embedding_model = OpenAICompatEmbeddings(client=client_embedding)
        self.vectorstore: Optional[Milvus] = None
        self.user = milvus_user
        self.passwd = milvus_passwd
        # 状态标记
        self._initialized: bool = False
        self._collection_loaded: bool = False  # 标记集合是否加载到内存
        self._warmup_done: bool = False        # 标记预热完成
    async def initialize(self):
        """
        异步初始化(含连接创建 + 集合预热, 适配服务⽣命周期)
        优化点: 双重检查锁 + 预热集合 + 缓存检索参数
        """
        if self._initialized:
            logger.debug("Milvus已初始化, 跳过重复初始化...")
            return
        async with self._init_lock:
            if self._initialized:  # 双重检查, 防⽌并发初始化
                return
            start_time = time.time()
            logger.info(f"开始初始化 Milvus 连接(URI: {self.uri})...")
            try:
                # 1. 创建Langchain Milvus客户端(异步上下⽂执⾏同步构造)
                self.vectorstore = Milvus(
                    embedding_function=self.embedding_model,
                    builtin_function=BM25BuiltInFunction(),
                    vector_field=["dense", "sparse"],
                    connection_args=build_connection_args(self.uri, self.user, self.passwd),
                    collection_name=self.collection_name,
                    auto_id=True,
                    consistency_level=self.config["consistency_level"],
                    drop_old=False  # 关键: 不删除旧集合, 避免重建开销
                )
                # 2. 缓存检索参数 (避免每次search重复创建字典)
                self._cache_search_params()
                # 3. 预热集合 (加载到内存, 压测核⼼优化)
                await self._warmup_collection()
                # 4. 标记初始化完成
                self._initialized = True
                self._warmup_done = True
                init_cost = round(time.time() - start_time, 3)
                logger.info(f"Milvus初始化完成  | 耗时: {init_cost}s | 集合: {self.collection_name}")
            except Exception as e:
                init_cost = round(time.time() - start_time, 3)
                logger.error(f"Milvus初始化失败  | 耗时: {init_cost}s | 错误: {str(e)}")
                raise e
    def _cache_search_params(self):
        """缓存检索参数, 减少运⾏时对象创建开销"""
        self._search_params_cache = [
            self.config["dense_search_params"],
            self.config["bm25_search_params"]
        ]
        logger.debug(f"Milvus检索参数已缓存! (集合: {self.collection_name})")
    async def _warmup_collection(self):
        """
        预热集合 (加载到内存)
        优化点: 避免每次检索时动态加载集合, 减少⾸次检索耗时
        """
        if self._collection_loaded or not self.vectorstore:
            return
        start_time = time.time()
        logger.info(f"开始预热 Milvus 集合({self.collection_name})...")
        try:
            # Langchain Milvus封装了底层加载逻辑, 显式调⽤load(若底层⽀持)
            if hasattr(self.vectorstore, "load"):
                await asyncio.to_thread(self.vectorstore.load)  # 同步操作转异步
            self._collection_loaded = True
            warmup_cost = round(time.time() - start_time, 3)
            logger.info(f"Milvus集合预热完成  | 耗时: {warmup_cost}s")
        except Exception as e:
            logger.warning(f"Milvus集合预热失败 (不影响功能, ⾸次检索会⾃动加载) | 错误: {str(e)}")
            self._collection_loaded = False
    async def _rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """
        Rerank 精排: 用 Cross-Encoder 对候选文档按语义相关性重新排序

        - 成功: 按 relevance_score 降序返回 top_n 条 (复用原 Document, metadata 不丢)
        - 失败/未启用: 降级为原始顺序的前 top_n 条, 保证检索链路不中断 (不抛异常)
        :param query: 用户问句
        :param docs: RRF 融合后的候选文档
        :param top_n: 精排后保留条数
        """
        if not RERANK_CONFIG["enabled"] or not docs:
            return docs[:top_n]
        url = f"{RERANK_CONFIG['base_url']}/rerank"
        payload = {
            "model": RERANK_CONFIG["model"],
            "query": query,
            "documents": [d.page_content for d in docs],
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {RERANK_CONFIG['api_key']}",
            "Content-Type": "application/json",
        }
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=RERANK_CONFIG["timeout"]) as http:
                resp = await http.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                results = resp.json().get("results", [])
            # results 里的 index 指向候选文档下标, 需据此映射回原 Document
            reranked = [
                docs[r["index"]]
                for r in results
                if isinstance(r.get("index"), int) and 0 <= r["index"] < len(docs)
            ]
            if not reranked:
                return docs[:top_n]
            cost = round(time.time() - start_time, 3)
            logger.info(
                f"Rerank完成  | query: {query[:20]}... | "
                f"候选: {len(docs)} → 精排: {len(reranked)} | "
                f"耗时: {cost}s | 模型: {RERANK_CONFIG['model']}"
            )
            return reranked
        except Exception as e:
            cost = round(time.time() - start_time, 3)
            logger.warning(
                f"Rerank失败, 降级为原始顺序  | query: {query[:20]}... | "
                f"耗时: {cost}s | 错误: {str(e)}"
            )
            return docs[:top_n]
    async def _fetch_parents(self, docs: List[Document]) -> List[Document]:
        """
        父子回表: 用命中的 child 的 parent_id 去同一 Collection 里取回父块正文.

        建库时父块与子块都写入了 Collection (doc_type 区分):
          - 子块 doc_type="child", parent_id 指向所属父块 doc_id
          - 父块 doc_type="parent", doc_id 即子块要回表的 parent_id
        检索只召回子块 (见 search 里的 expr 过滤), 命中后在此按 parent_id 回表取父块正文,
        多个子块命中同一父块时按首次出现顺序去重 (保留最优名次).
        JSONL 等无父块的文档 (parent_id 为空) 原样保留; 回表失败的子块降级保留其自身内容.
        """
        if not docs:
            return []
        # 1: 收集需要回表的 parent_id (去重, 保持顺序)
        parent_ids: List[str] = []
        for d in docs:
            pid = d.metadata.get("parent_id") or ""
            if pid and pid not in parent_ids:
                parent_ids.append(pid)
        if not parent_ids:
            return docs
        # 2: 按 doc_id 批量查询父块 (doc_id in [...])
        row_map: Dict[str, Dict[str, Any]] = {}
        try:
            expr = f"doc_id in {json.dumps(parent_ids, ensure_ascii=False)}"
            rows = self.vectorstore.col.query(
                expr=expr,
                output_fields=["doc_id", "text", "source", "image_url", "pdf_url", "chunk_index"],
            )
            row_map = {r["doc_id"]: r for r in rows}
        except Exception as e:
            logger.warning(f"父子回表查询失败, 降级返回子块内容 | 错误: {str(e)}")
            return docs
        # 3: 按原召回顺序, 用父块正文替换子块, 同一父块只保留一次
        result: List[Document] = []
        seen_parents = set()
        for d in docs:
            pid = d.metadata.get("parent_id") or ""
            if not pid:
                result.append(d)          # 无父块 (如 JSONL), 原样返回
                continue
            if pid in seen_parents:
                continue                  # 该父块已输出过, 去重
            row = row_map.get(pid)
            if not row:
                result.append(d)          # 回表未命中, 降级保留子块
                continue
            seen_parents.add(pid)
            result.append(Document(
                page_content=row.get("text", "") or d.page_content,
                metadata={
                    "doc_id":      row.get("doc_id", ""),
                    "parent_id":   pid,
                    "doc_type":    "parent",
                    "source":      row.get("source", ""),
                    "image_url":   row.get("image_url", ""),
                    "pdf_url":     row.get("pdf_url", ""),
                    "chunk_index": d.metadata.get("chunk_index"),
                },
            ))
        return result
    async def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        ⾼性能异步检索 (混合检索 + Rerank 精排)
        :param query: 检索问句
        :param k: 最终返回条数
                  - 未启用 rerank: 默认 default_top_k (30)
                  - 启用 rerank:   默认 RERANK_CONFIG["top_n"] (10)
        :return: 召回的 Document 列表
        """
        # 1: 前置检查: 确保初始化和预热完成
        if not self._initialized:
            await self.initialize()
        if not self._warmup_done:
            await self._warmup_collection()
        # 2: 参数处理
        rerank_on = RERANK_CONFIG["enabled"]
        # 最终返回条数: 未启用 rerank 时保持接入前语义 (default_top_k)
        final_k = k or (RERANK_CONFIG["top_n"] if rerank_on else self.config["default_top_k"])
        # 粗召回条数: 启用 rerank 时多召回候选 (candidate_k) 交给精排挑选, 否则等于 final_k
        recall_k = max(final_k, RERANK_CONFIG["candidate_k"]) if rerank_on else final_k
        search_params = self._search_params_cache or [
            self.config["dense_search_params"],
            self.config["bm25_search_params"]
        ]
        # 每个向量分支 (dense / sparse) 的预取条数.
        # langchain-milvus 的默认值 fetch_k=4 只声明在 _acollection_hybrid_search 的签名里,
        # 调用 asimilarity_search 时不显式传 fetch_k 就会沿用该默认值,
        # 导致混合检索候选池仅 4+4 条, k 再大也只返回 4~8 条 (RRF 融合后的上限),
        # 因此这里显式按粗召回条数的 2 倍预取.
        fetch_k = recall_k * 2
        # 3: 异步检索 (带耗时监控)
        start_time = time.time()
        logger.debug(f"开始 Milvus 检索 | query: {query[:20]}... | 粗召回: {recall_k} | 最终: {final_k}")
        try:
            # 核⼼检索逻辑 (粗召回候选池)
            # 方案B(父子回表): 只让 child 块参与匹配 (expr 过滤 doc_type), 父块不参与;
            # 命中子块后再在下方用 parent_id 回表取父块正文送 LLM (见 _fetch_parents)
            docs = await self.vectorstore.asimilarity_search(
                query=query,
                k=recall_k,
                fetch_k=fetch_k,
                expr="doc_type == 'child'",
                ranker_type=self.config["ranker_type"],
                ranker_params=self.config["ranker_params"],
                search_params=search_params
            )
            # 兼容保护: 万一底层未应用 doc_type 过滤, 这里再剔除父块
            # (老库无 doc_type 字段时按 child 处理, 不影响原有行为)
            docs = [d for d in docs if d.metadata.get("doc_type", "child") != "parent"]
            # 4: Rerank 精排 (粗召回 → 精排取 top_n; 未启用或失败时自动降级为原始顺序)
            #    注意: 精排对子块打分 (细粒度命中), 回表取父块是为了给 LLM 更完整上下文
            if rerank_on:
                docs = await self._rerank(query, docs, final_k)
            # 4.5: 父子回表 — 用命中子块的 parent_id 取回父块正文 (同一父块自动去重)
            docs = await self._fetch_parents(docs)
            # 5. 耗时统计 (压测关键指标)
            search_cost = round(time.time() - start_time, 3)
            logger.info(
                f"Milvus检索完成  | "
                f"query: {query[:20]}... | "
                f"召回数: {len(docs)} | "
                f"耗时: {search_cost}s | "
                f"粗召回: {recall_k} | "
                f"最终: {final_k} | "
                f"rerank: {'开' if rerank_on else '关'}"
            )
            return docs
        except Exception as e:
            search_cost = round(time.time() - start_time, 3)
            logger.error(
                f"Milvus检索失败  | "
                f"query: {query[:20]}... | "
                f"耗时: {search_cost}s | "
                f"错误: {str(e)}"
            )
            # 降级逻辑: 返回空列表, 避免压测时服务崩溃
            return []
    def format_docs(self, docs: List[Document]) -> str:
        """
        格式化召回结果 (轻量优化: 跳过空⽂档, 减少拼接开销)
        """
        if not docs:
            return ""
        # 优化: 过滤空内容⽂档, 减少⽆效⽂本拼接
        valid_docs = [doc for doc in docs if doc.page_content.strip()]
        return "\n\n".join(doc.page_content for doc in valid_docs)
    async def close(self):
        """
        关闭Milvus连接 (适配服务优雅退出)
        优化点: 释放连接资源, 避免压测后连接泄漏
        """
        if not self._initialized or not self.vectorstore:
            logger.debug("Milvus未初始化, ⽆需关闭!")
            return
        start_time = time.time()
        logger.info(f"开始关闭 Milvus 连接 (集合: {self.collection_name})...")
        try:
            # 卸载集合 (释放内存)
            if hasattr(self.vectorstore, "release") and self._collection_loaded:
                await asyncio.to_thread(self.vectorstore.release)
            # 关闭底层连接 (Langchain 封装的 Milvus 客户端)
            if hasattr(self.vectorstore, "close"):
                await asyncio.to_thread(self.vectorstore.close)
            # 重置状态
            self._initialized = False
            self._collection_loaded = False
            self._warmup_done = False
            close_cost = round(time.time() - start_time, 3)
            logger.info(f"Milvus连接关闭完成  | 耗时: {close_cost}s")
        except Exception as e:
            logger.error(f"Milvus连接关闭失败  | 错误: {str(e)}")
# ===================== 实例化 (全局复⽤) =====================
# 现代医学知识 Collection (主 collection)
milvus_manager = MilvusClientWrapper(client_embedding=client, uri=milvus_uri)
# 中医古籍 Collection (独⽴ collection, 在 app.py lifespan 中初始化)
ancient_books_manager = MilvusClientWrapper(
    client_embedding=client, uri=milvus_uri, collection_config=ANCIENT_BOOKS_CONFIG)
