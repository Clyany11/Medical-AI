"""milvus_schema.py — Milvus Collection Schema 设计与数据写⼊⼯具功能概述:
  1. 定义⽀持图⽚URL的 Milvus Collection Schema
  2. 向量字段 (dense / sparse) 与 image_url 字段完全分离, 不⼲扰语义相似度计算
  3. 采⽤ IVF_FLAT 索引 + BM25 混合检索, 最⼤化检索精度与速度
     (Milvus Lite 不支持 HNSW, 稠密向量实际使用 IVF_FLAT)
  4. 提供带 image_url 的⽂档写⼊函数, 供数据⼊库时调⽤
  5. 查询时将 image_url 随召回结果⼀并返回给前端Collection 字段设计:
  │ 字段名           │ 类型              │ 说明                                 │
  │ pk              │ INT64 (auto_id)  │ Milvus ⾃动⽣成主键                     │
  │ text            │ VARCHAR(65535)   │ 原始⽂本内容 (⽤于 BM25 分析)            │
  │ dense           │ FLOAT_VECTOR     │ 稠密向量 (语义相似度，不含URL)            │
  │ sparse          │ SPARSE_FLOAT_VEC │ 稀疏向量 (BM25 关键词匹配)               │
  │ doc_id          │ VARCHAR(64)      │ ⽂档 UUID (溯源⽤)                      │
  │ source          │ VARCHAR(512)     │ 原始⽂件路径                            │
  │ chunk_index     │ INT64            │ 分块序号                               │
  │ image_url       │ VARCHAR(2048)    │ 图⽚ URL (不参与向量化, 仅存储)          │
  │ pdf_url         │ VARCHAR(2048)    │ 原始 PDF ⽂件 OSS URL (溯源⽤, 仅存储)   │
  │ parent_id       │ VARCHAR(64)      │ 归属⽗块 doc_id (⼦块回表⽤, 仅存储)      │
  │ doc_type        │ VARCHAR(16)      │ "parent" / "child" (区分父子记录)        │与 milvus_db.py 的关系:
  - milvus_db.py 负责查询 (search), 本⽂件负责 Schema 定义和数据写⼊
  - 两者共⽤ collection_name = "agent_knowledge"
  - image_url 通过 Document.metadata["image_url"] 传递, 不影响向量检索逻辑使⽤示例:
  from milvus_schema import MilvusSchemaManager
  manager = MilvusSchemaManager(uri="http://localhost:19530")
  manager.create_collection_if_not_exists()
  manager.insert_documents_with_images(docs_with_images)"""
import os
import uuid
import time
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    connections,
    utility, )
from vectors import OpenAICompatEmbeddings, create_embedding_client, get_embedding_dim
# 复用连接参数构造逻辑 (本地文件 Milvus Lite 不支持账号密码)
from milvus_db import build_connection_args
from dotenv import load_dotenv
load_dotenv()
# 环境变量加载
# 注意: 变量名不用 MILVUS_URI, 因为 pymilvus 会读取同名环境变量并强制要求 http(s) 格式
milvus_uri = os.getenv("MILVUS_ENDPOINT")
milvus_user = os.getenv("MILVUS_USER")
# 兼容历史变量名: 优先 MILVUS_PASSWORD, 回退 MILVUS_PASSWD
milvus_passwd = os.getenv("MILVUS_PASSWORD") or os.getenv("MILVUS_PASSWD")
# ===================== ⽇志配置 =====================
logger = logging.getLogger("MilvusSchemaManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# ===================== 全局常量 =====================
COLLECTION_NAME = "agent_knowledge"    # 与 milvus_db.py 保持⼀致
DENSE_DIM = get_embedding_dim()         # 稠密向量维度, 由 EMBEDDING_DIM 决定 (须与 embedding 模型输出一致)
MAX_TEXT_LEN = 65535                    # ⽂本字段最⼤⻓度
MAX_IMAGE_URL_LEN = 2048               # 图⽚ URL 最⼤⻓度
MAX_PDF_URL_LEN = 2048                 # PDF URL 最⼤⻓度
MAX_SOURCE_LEN = 512                   # 来源路径最⼤⻓度
MAX_DOC_ID_LEN = 64                    # doc_id UUID 最⼤⻓度
MAX_PARENT_ID_LEN = 64                 # parent_id UUID 最⼤⻓度 (与 doc_id 同规格)
# 写⼊参数默认值 (可⽤ BUILD_BATCH_SIZE / BUILD_BATCH_SLEEP 覆盖)
DEFAULT_INSERT_BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", 32))
DEFAULT_INSERT_SLEEP = float(os.getenv("BUILD_BATCH_SLEEP", 0.1))
class MilvusSchemaManager:
    """
    Milvus Collection Schema 管理器
    职责：
      1. 创建/检查 Collection (含图⽚URL字段)
      2. 批量写⼊带 image_url 的⽂档数据
      3. 提供 Schema 信息查询接⼝
    设计原则：
      - image_url 字段类型为 VARCHAR, 不参与任何向量化操作
      - dense 向量字段仅存储⽂本语义向量, 与 image_url 完全隔离
      - 查询时通过 output_fields 将 image_url ⼀并返回
    """
    def __init__(
        self,
        uri: str,
        embedding_client: Optional[Any] = None,
        collection_name: str = COLLECTION_NAME,
        drop_old: bool = False,
    ):
        """
        初始化 Schema 管理器
        :param uri:              Milvus 连接地址（如 "http://localhost:19530"）
        :param embedding_client: OpenAI 兼容的 embedding 客户端 (⽤于⽂档写⼊时⽣成向量)
        :param collection_name:  目标 Collection 名 (默认 agent_knowledge; 挂载旧库时需指向其实际集合名)
        :param drop_old:         是否先删除同名旧 Collection (新增字段/重建时必须置 True, 否则旧 schema 不兼容)
        """
        self.uri = uri
        self.user = milvus_user
        self.passwd = milvus_passwd
        self.collection_name = collection_name
        self.drop_old = drop_old
        # 嵌⼊模型 (写⼊时使⽤), 未传入客户端时按 .env 配置自动创建
        self.embeddings = OpenAICompatEmbeddings(client=embedding_client or create_embedding_client())
        # Langchain Milvus 实例 (延迟初始化)
        self._vectorstore: Optional[Milvus] = None
    def _get_vectorstore(self) -> Milvus:
        """
        获取或创建 Langchain Milvus 实例 (懒加载)
        Langchain Milvus 会⾃动处理 Collection 的创建和字段映射:
          - text_key="text"               → ⽂本内容字段
          - metadata_field中的image_url    → 存储为 VARCHAR, 不参与向量化
          - auto_id=True                  → Milvus ⾃动⽣成 pk
        """
        if self._vectorstore is None:
            self._vectorstore = Milvus(
                embedding_function=self.embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                # index_params 与 vector_field 按位置一一对应:
                #   dense  -> IVF_FLAT / IP   (Milvus Lite 下稠密向量索引)
                #   sparse -> SPARSE_INVERTED_INDEX / BM25
                # 不传则默认 AUTOINDEX, 会使 Milvus Lite 无法 load 稀疏列
                index_params=[
                    {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
                    {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
                ],
                connection_args=build_connection_args(self.uri, self.user, self.passwd),
                collection_name=self.collection_name,
                auto_id=True,
                # Eventually ⼀致性: 写⼊后不⽴即可查, 但性能最优
                consistency_level="Eventually",
                drop_old=self.drop_old,
            )
        return self._vectorstore
    def insert_documents_with_images(
        self,
        docs: List[Document],
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        sleep_between_batches: float = DEFAULT_INSERT_SLEEP,
    ) -> int:
        """
        批量写⼊带图⽚URL的⽂档到 Milvus
        ⽂档格式要求 (metadata 字段):
          doc.metadata = {
              "doc_id":      str(uuid.uuid4()),    # ⽂档唯⼀标识 (溯源⽤)
              "source":      "data/medical.jsonl", # 原始⽂件路径
              "chunk_index": 0,                    # 分块序号
              "image_url":   "https://...",        # 图⽚ URL (可为空字符串)
              "pdf_url":     "https://...",        # 原始 PDF OSS URL (可为空字符串)
              "parent_id":   "…",                  # ⼦块归属的⽗块 doc_id (回表⽤; 空字符串表示⽆⽗块)
              "doc_type":    "child",              # "parent" 父块 / "child" ⼦块
          }
        image_url 处理逻辑:
          - ⾮空 URL  →  存⼊ Milvus VARCHAR 字段, 查询时随结果返回
          - 空字符串   →  存⼊空字符串, 前端判断是否展示图⽚
          - URL 不参与 embed_documents(), 不影响语义向量
        :param docs:                   Document 列表
        :param batch_size:             每批写⼊数量  (避免单次写⼊过多导致超时)
        :param sleep_between_batches:  批次间休眠时间 (秒), 避免 Milvus 过载
        :return: 成功写⼊的⽂档总数
        """
        vs = self._get_vectorstore()
        total = 0
        batch = []
        logger.info(f"开始写⼊ {len(docs)} 条⽂档 (含图⽚URL) 到 Milvus...")
        for i, doc in enumerate(docs):
            # 确保 metadata 包含必要字段
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = str(uuid.uuid4())
            if "image_url" not in doc.metadata:
                doc.metadata["image_url"] = ""
            if "pdf_url" not in doc.metadata:
                doc.metadata["pdf_url"] = ""
            if "chunk_index" not in doc.metadata:
                doc.metadata["chunk_index"] = i
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""
            # 父子回表字段: 父块 parent_id 为空, 子块 parent_id 指向父块 doc_id
            if "parent_id" not in doc.metadata:
                doc.metadata["parent_id"] = ""
            if "doc_type" not in doc.metadata:
                doc.metadata["doc_type"] = "child"
            # start_index 由 RecursiveCharacterTextSplitter(add_start_index=True) ⾃动注⼊
            # JSONL 数据没有该字段, 统⼀补 0, 确保所有批次字段⼀致, 避免 Milvus DataNotMatchException
            if "start_index" not in doc.metadata:
                doc.metadata["start_index"] = 0
            # image_url 截断保护
            if len(doc.metadata["image_url"]) > MAX_IMAGE_URL_LEN:
                logger.warning(f"image_url 超⻓, 已截断: {doc.metadata['image_url'][:50]}...")
                doc.metadata["image_url"] = doc.metadata["image_url"][:MAX_IMAGE_URL_LEN]
            # pdf_url 截断保护
            if len(doc.metadata["pdf_url"]) > MAX_PDF_URL_LEN:
                logger.warning(f"pdf_url 超⻓, 已截断: {doc.metadata['pdf_url'][:50]}...")
                doc.metadata["pdf_url"] = doc.metadata["pdf_url"][:MAX_PDF_URL_LEN]
            # parent_id 截断保护
            if len(doc.metadata["parent_id"]) > MAX_PARENT_ID_LEN:
                logger.warning(f"parent_id 超⻓, 已截断: {doc.metadata['parent_id'][:50]}...")
                doc.metadata["parent_id"] = doc.metadata["parent_id"][:MAX_PARENT_ID_LEN]
            batch.append(doc)
            if len(batch) >= batch_size:
                try:
                    vs.add_documents(batch)
                    total += len(batch)
                    logger.info(f"已写⼊ {total}/{len(docs)} 条⽂档...")
                    time.sleep(sleep_between_batches)
                except Exception as e:
                    logger.error(f"批次写⼊失败 (第 {total} 条附近): {e}")
                batch = []
        # 写⼊剩余⽂档
        if batch:
            try:
                vs.add_documents(batch)
                total += len(batch)
                logger.info(f"已写⼊ {total}/{len(docs)} 条⽂档(最后⼀批)")
            except Exception as e:
                logger.error(f"最后批次写⼊失败: {e}")
        logger.info(f"⽂档写⼊完成  | 总计: {total} 条")
        return total
    def prepare_medical_doc_with_image(
        self,
        text: str,
        image_url: str = "",
        source: str = "",
        chunk_index: int = 0,
    ) -> Document:
        """
        构建带图⽚URL的医疗⽂档对象 (⼯⼚⽅法)
        使⽤示例:
            doc = manager.prepare_medical_doc_with_image(
                text="⾼⾎压患者应注意低盐饮⻝...",
                image_url="https://xiaozhuteacher.oss-cn-beijing.aliyuncs.com/xiaozhuteacher/1.jpg",
                source="data/hypertension.jsonl",
                chunk_index=0,
            )
        :param text:        ⽂档⽂本内容   (将被向量化)
        :param image_url:   配套图⽚ URL  (不参与向量化, 仅存储)
        :param source:      原始数据来源路径
        :param chunk_index: 分块序号
        :return: LangChain Document 对象
        """
        return Document(
            page_content=text,
            metadata={
                "doc_id":      str(uuid.uuid4()),
                "source":      source,
                "chunk_index": chunk_index,
                # image_url 存储在 metadata 中, Langchain Milvus 会将其映射为 VARCHAR 字段
                # 关键: 此字段不会被 embed_documents() 处理, 不影响语义向量
                "image_url":   image_url,
            },
        )
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取 Collection 基本信息 (⽤于运维检查)
        :return: 包含 collection_name, entity_count 等信息的字典
        """
        try:
            vs = self._get_vectorstore()
            info = {
                "collection_name": self.collection_name,
                "uri": self.uri,
                "status": "connected",
            }
            # 尝试获取实体数量
            if hasattr(vs, "col") and vs.col:
                info["entity_count"] = vs.col.num_entities
            return info
        except Exception as e:
            return {"collection_name": self.collection_name, "status": "error", "error": str(e)}
# ===================== 数据准备⼯具函数 =====================
def prepare_documents_from_jsonl_with_images(
    file_path: str,
    image_url_map: Optional[Dict[str, str]] = None, ) -> List[Document]:
    """
    从 JSONL ⽂件加载医疗问答数据, ⽀持关联图⽚URL
    JSONL 格式 (每⾏⼀个 JSON 对象):
        {"query": "⾼⾎压怎么治疗?", "response": "⾼⾎压治疗包括...", "image_url": "https://..."}
        或
        {"query": "...", "response": "..."}  # ⽆图⽚时 image_url 为空
    :param file_path:     JSONL ⽂件路径
    :param image_url_map: 可选的 query → image_url 映射字典 (覆盖⽂件中的 image_url)
    :return: Document 列表
    """
    import json
    docs = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            content = json.loads(line)
            query = content.get("query", "")
            response = content.get("response", "")
            text = f"{query}\n{response}"
            # 优先使⽤外部映射的 image_url, 其次使⽤⽂件中的, 最后为空
            image_url = ""
            if image_url_map and query in image_url_map:
                image_url = image_url_map[query]
            elif "image_url" in content:
                image_url = content["image_url"]
            doc = Document(
                page_content=text,
                metadata={
                    "doc_id":      str(uuid.uuid4()),
                    "source":      file_path,
                    "chunk_index": i,
                    # image_url 字段: 存储图⽚地址, 不参与向量化
                    # 样例: https://xiaozhuteacher.oss-cn-beijing.aliyuncs.com/xiaozhuteacher/1.jpg
                    "image_url":   image_url,
                },
            )
            docs.append(doc)
            count += 1
    logger.info(f"从 {file_path} 加载了 {count} 条⽂档 (含图⽚URL映射)")
    return docs
def extract_image_urls_from_search_results(docs: List[Document]) -> List[Dict[str, str]]:
    """
    从 Milvus 召回结果中提取图⽚URL和PDF URL, 供前端展示使⽤
    :param docs: milvus_manager.search() 返回的 Document 列表
    :return: [{"rank": 0, "image_url": "...", "pdf_url": "...", "text_preview": "..."}, ...]
    """
    result = []
    for rank, doc in enumerate(docs):
        image_url = doc.metadata.get("image_url", "")
        pdf_url = doc.metadata.get("pdf_url", "")
        result.append({
            "rank": rank,
            "image_url": image_url,
            "pdf_url": pdf_url,
            "text_preview": (doc.page_content or "")[:500],
        })
    return result
