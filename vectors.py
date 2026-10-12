"""vectors.py — 通用 OpenAI 兼容 Embedding 封装

支持任意提供 OpenAI 兼容 `/v1/embeddings` 接口的服务, 例如:
  - 硅基流动 SiliconFlow  (https://api.siliconflow.cn/v1)
  - 智谱 GLM             (https://open.bigmodel.cn/api/paas/v4)
  - 阿里云百炼 DashScope  (https://dashscope.aliyuncs.com/compatible-mode/v1)
  - OpenAI              (https://api.openai.com/v1)

通过 .env 配置 (切换厂商只需改这三个变量):
  EMBEDDING_API_KEY   必填
  EMBEDDING_BASE_URL  默认 https://api.siliconflow.cn/v1
  EMBEDDING_MODEL     默认 BAAI/bge-m3
  EMBEDDING_DIM       默认 1024, 必须与该模型实际输出维度一致,
                      且必须与 Milvus collection 的 dense 字段维度一致
  EMBEDDING_BATCH_SIZE 默认 32, 单次请求提交的文本条数 (上限由厂商决定)

设计要点:
  - 继承 LangChain Embeddings 接口, 可直接作为 embedding_function 传给 langchain_milvus.Milvus
  - 批量提交 (batch_size) 以减少网络往返
  - 返回 numpy.ndarray (与项目内 rag_evaluation 的余弦相似度计算保持一致)
"""
import logging
import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("Vectors")

# 默认配置 (可用 .env 覆盖)
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024
# 单次请求提交的文本条数 (可用 EMBEDDING_BATCH_SIZE 覆盖; 各厂商上限不同, 超限会报错)
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))


def create_embedding_client() -> OpenAI:
    """创建 OpenAI 兼容的 embedding 客户端 (厂商由 EMBEDDING_BASE_URL 决定)"""
    return OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL", DEFAULT_BASE_URL),
    )


def get_embedding_dim() -> int:
    """读取 embedding 输出维度 (Milvus dense 字段维度须与此一致)"""
    return int(os.getenv("EMBEDDING_DIM", DEFAULT_DIM))


class OpenAICompatEmbeddings(Embeddings):
    """OpenAI 兼容的文本嵌入

    :param client: OpenAI 兼容客户端, 为 None 时按 .env 自动创建
    :param model:  嵌入模型 ID, 为 None 时读 EMBEDDING_MODEL
    :param batch_size: 单次请求的文本条数
    """

    def __init__(self, client: OpenAI = None, model: str = None, batch_size: int = BATCH_SIZE):
        self.client = client or create_embedding_client()
        self.model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size

    def _embed_batch(self, texts):
        """调用一次嵌入接口, 返回 numpy 向量列表"""
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]

    def embed_documents(self, texts):
        """批量嵌入 (文档入库时使用)"""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[i: i + self.batch_size]))
        return embeddings

    def embed_query(self, text):
        """单条嵌入 (检索时使用)"""
        return self._embed_batch([text])[0]