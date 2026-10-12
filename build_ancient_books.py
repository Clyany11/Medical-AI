"""build_ancient_books.py — 中医古籍数据⼊库脚本功能:
  1. 从 data/Ancient-Books/*.txt 读取中医古籍⽂本
  2. 解析元数据 (书名, 作者, 朝代)
  3. 将原始 .txt ⽂件上传到阿⾥云 OSS，OSS URL 存⼊每个 chunk 的 pdf_url 字段
  4. 按 parent(2000字) → child(400字) 两级切分 (与 build_milvus_db.py 保持⼀致)
  5. 调⽤ OpenAI 兼容的 embedding API ⽣成向量
  6. 写⼊ Milvus ancient_books_knowledge collection运⾏⽅式:
  python3 build_ancient_books.py                    # 全量建库
  python3 build_ancient_books.py --limit 10         # 仅处理前 10 个⽂件 (测试⽤)
  python3 build_ancient_books.py --book "伤寒论"     # 仅处理指定书籍
注意：
  - Milvus URI 从 .env 读取
  - 所有外部依赖 (EMBEDDING_API_KEY / MILVUS_URI 等) 从 .env 读取
  - 本脚本复⽤已有模块 (milvus_schema, vectors, milvus_db, testAliyunfiles)"""
import os
import sys
import re
import uuid
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 复⽤已有模块
from vectors import OpenAICompatEmbeddings, create_embedding_client
from milvus_schema import MilvusSchemaManager
from milvus_db import ANCIENT_BOOKS_CONFIG
from testAliyunfiles import upload_to_oss
load_dotenv()
# ===================== ⽇志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger("BuildAncientBooks")
# ===================== 全局配置 =====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
ANCIENT_BOOKS_DIR = "./data/Ancient-Books"
# 写⼊批次⼤⼩ (Milvus 单批写入条数; embedding 请求的批大小由 EMBEDDING_BATCH_SIZE 控制)
BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", 32))
BATCH_SLEEP = float(os.getenv("BUILD_BATCH_SLEEP", 0.1))   # 批次间休眠 (秒)
# ⽗⼦⽂档切分参数 (与 build_milvus_db.py 保持⼀致)
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
def parse_metadata(file_path: str, content: str) -> Dict[str, str]:
    """
    从古籍⽂本中解析元数据 (书名, 作者, 朝代)
    ⽂本格式示例:
      <篇名>伤寒论书名: 伤寒论作者: 张仲景朝代: 东汉年份: 公元25-220年
    :param file_path: ⽂件路径 (⽤于提取书名)
    :param content: ⽂件内容
    :return: {"book_name": "伤寒论", "author": "张仲景", "dynasty": "东汉"}
    """
    metadata = {
        "book_name": Path(file_path).stem,  # 默认使⽤⽂件名作为书名
        "author": "",
        "dynasty": ""
    }
    # 尝试从第⼀⾏提取元数据
    first_line = content.split("\n")[0] if content else ""
    # 提取书名 (优先从⽂本中提取, 失败则使⽤⽂件名)
    book_match = re.search(r"书名[：:](.*?)(?:作者|朝代|年份|<|$)", first_line)
    if book_match:
        metadata["book_name"] = book_match.group(1).strip()
    # 提取作者
    author_match = re.search(r"作者[：:](.*?)(?:朝代|年份|<|$)", first_line)
    if author_match:
        metadata["author"] = author_match.group(1).strip()
    # 提取朝代
    dynasty_match = re.search(r"朝代[：:](.*?)(?:年份|<|$)", first_line)
    if dynasty_match:
        metadata["dynasty"] = dynasty_match.group(1).strip()
    return metadata
# ======= ⽗⼦⽂档两级切分 (与 build_milvus_db.py 保持⼀致) =========
def split_parent_child(raw_doc: Document, txt_url: str) -> List[Document]:
    """
    对原始⽂档进⾏⽗⼦两级切分 (parent 2000 → child 400)
    所有 child 块共享同⼀ txt_url (存⼊ pdf_url 字段)
    :param raw_doc: 原始 Document (整个⽂件内容)
    :param txt_url: 原始 .txt ⽂件的 OSS URL (存⼊ pdf_url 字段)
    :return: child Document 列表
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        add_start_index=True,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        add_start_index=True,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    # 第⼀级: parent 切分
    parent_chunks = parent_splitter.split_documents([raw_doc])
    logger.info(f"  parent 切分: 1 条原始⽂档 → {len(parent_chunks)} 个 parent 块")
    # 第⼆级: child 切分
    child_docs: List[Document] = []
    chunk_idx = 0
    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            # 继承原始⽂档的元数据
            child.metadata["doc_id"] = str(uuid.uuid4())
            child.metadata["source"] = raw_doc.metadata.get("source", "")
            child.metadata["book_name"] = raw_doc.metadata.get("book_name", "")
            child.metadata["author"] = raw_doc.metadata.get("author", "")
            child.metadata["dynasty"] = raw_doc.metadata.get("dynasty", "")
            child.metadata["image_url"] = ""
            child.metadata["pdf_url"] = txt_url  # 存⼊ .txt ⽂件的 OSS URL
            child.metadata["chunk_index"] = chunk_idx
            child_docs.append(child)
            chunk_idx += 1
    logger.info(f"  child 切分完成: {len(parent_chunks)} 个 parent 块 → {len(child_docs)} 个 child 块")
    return child_docs
# ===================== ⽂档构建 =====================
def build_documents_from_file(file_path: str) -> List[Document]:
    """
    从单个古籍⽂件构建 Document 列表:
      1. 读取⽂件内容
      2. 解析元数据 (书名, 作者, 朝代)
      3. 将原始 .txt ⽂件上传到阿⾥云 OSS，URL 存⼊ pdf_url 字段
      4. 按 parent(2000) → child(400) 两级切分
    :param file_path: 古籍⽂件路径
    :return: child Document 列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logger.error(f"读取⽂件失败 ({file_path}): {e}")
        return []
    if not content:
        logger.warning(f"⽂件内容为空: {file_path}")
        return []
    # 1: 解析元数据
    metadata = parse_metadata(file_path, content)
    logger.info(f"解析元数据: {metadata['book_name']} | 作者: {metadata['author']} | 朝代: {metadata['dynasty']}")
    # 2: 上传原始 .txt ⽂件到阿⾥云 OSS，获取 txt_url 存⼊ pdf_url 字段
    txt_url = upload_to_oss(file_path) or ""
    if txt_url:
        logger.info(f"  .txt ⽂件上传 OSS 成功: {txt_url}")
    else:
        logger.warning(f"  .txt ⽂件上传 OSS 失败，pdf_url 置空")
    # 3: 构建原始 Document (整个⽂件作为⼀条原始⽂档)
    source = f"Ancient-Books/{Path(file_path).name}"
    raw_doc = Document(
        page_content=content,
        metadata={
            "doc_id":      str(uuid.uuid4()),
            "source":      source,
            "chunk_index": 0,
            "book_name":   metadata["book_name"],
            "author":      metadata["author"],
            "dynasty":     metadata["dynasty"],
            "image_url":   "",
            "pdf_url":     txt_url,
        },
    )
    # 4: ⽗⼦两级切分
    child_docs = split_parent_child(raw_doc, txt_url)
    logger.info(f"构建⽂档完成: {len(child_docs)} 个 child chunk")
    return child_docs
# ===================== 主⼊⼝ =====================
def parse_args():
    parser = argparse.ArgumentParser(description="中医古籍数据⼊库脚本")
    parser.add_argument("--limit", type=int, help="仅处理前 N 个⽂件 (测试⽤)")
    parser.add_argument("--book", type=str, help="仅处理指定书籍 (⽂件名, 不含扩展名)")
    return parser.parse_args()
def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("中医古籍数据⼊库脚本启动")
    logger.info(f"  MILVUS_URI  : {MILVUS_URI}")
    logger.info(f"  古籍⽬录    : {ANCIENT_BOOKS_DIR}")
    logger.info(f"  limit       : {args.limit}")
    logger.info(f"  book        : {args.book}")
    logger.info("=" * 60)
    # 初始化 Embedding 客户端 (OpenAI 兼容)
    if not EMBEDDING_API_KEY:
        logger.error("EMBEDDING_API_KEY 未配置, 请检查 .env ⽂件")
        sys.exit(1)
    embedding_client = create_embedding_client()
    # 初始化 MilvusSchemaManager（使⽤古籍 collection 配置）
    schema_manager = MilvusSchemaManager(uri=MILVUS_URI, embedding_client=embedding_client)
    # 获取所有 .txt ⽂件
    ancient_books_path = Path(ANCIENT_BOOKS_DIR)
    if not ancient_books_path.exists():
        logger.error(f"古籍⽬录不存在: {ANCIENT_BOOKS_DIR}")
        sys.exit(1)
    txt_files = sorted(ancient_books_path.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name not in ["README.md", "CLAUDE.md"]]
    # 过滤: 仅处理指定书籍
    if args.book:
        txt_files = [f for f in txt_files if args.book in f.stem]
        if not txt_files:
            logger.error(f"未找到书籍: {args.book}")
            sys.exit(1)
    # 限制⽂件数量 (测试⽤)
    if args.limit:
        txt_files = txt_files[:args.limit]
    logger.info(f"共找到 {len(txt_files)} 个古籍⽂件")
    # 逐个⽂件处理
    total_written = 0
    for txt_file in tqdm(txt_files, desc="处理古籍⽂件"):
        logger.info(f"\n处理⽂件: {txt_file.name}")
        # 1: 构建 Document 列表
        docs = build_documents_from_file(str(txt_file))
        if not docs:
            logger.warning(f"跳过⽂件(⽆有效⽂档): {txt_file.name}")
            continue
        # 2: 写⼊ Milvus (使⽤古籍 collection)
        try:
            # 修改collection_name (因为MilvusSchemaManager默认使⽤agent_knowledge)
            # 这⾥直接使⽤ Langchain Milvus 写⼊, 指定collection_name
            from langchain_milvus import Milvus, BM25BuiltInFunction
            vectorstore = Milvus(
                embedding_function=OpenAICompatEmbeddings(client=embedding_client),
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={
                    "uri": MILVUS_URI,
                    "user": os.getenv("MILVUS_USER", "root"),
                    "password": os.getenv("MILVUS_PASSWORD") or os.getenv("MILVUS_PASSWD", "Milvus")
                },
                collection_name=ANCIENT_BOOKS_CONFIG["collection_name"],
                auto_id=True,
                consistency_level=ANCIENT_BOOKS_CONFIG["consistency_level"],
                drop_old=False,
            )
            # 批量写⼊
            batch = []
            for i, doc in enumerate(docs):
                batch.append(doc)
                if len(batch) >= BATCH_SIZE:
                    vectorstore.add_documents(batch)
                    total_written += len(batch)
                    logger.info(f"  已写⼊ {total_written} 个 chunk...")
                    time.sleep(BATCH_SLEEP)
                    batch = []
            # 写⼊剩余⽂档
            if batch:
                vectorstore.add_documents(batch)
                total_written += len(batch)
                logger.info(f"  已写⼊ {total_written} 个 chunk（最后⼀批）")
        except Exception as e:
            logger.error(f"写⼊ Milvus 失败 ({txt_file.name}): {e}")
            continue
    logger.info("=" * 60)
    logger.info(f"建库完成！本次共写⼊ {total_written} 条数据到 Milvus")
    logger.info(f"Collection: {ANCIENT_BOOKS_CONFIG['collection_name']}")
    logger.info("=" * 60)
if __name__ == "__main__":
    main()
