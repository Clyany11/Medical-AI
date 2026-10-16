"""
build_milvus_db.py — Milvus 数据库建库脚本
功能:
  1. 从 data/data.jsonl 读取医疗问答数据, 为每条文本:
     - 调用 OpenAI 兼容的 embedding API 生成向量 (维度由 EMBEDDING_DIM 决定)
     - 调用豆包文生图模型生成配套图片
     - 将图片从火山云 URL 下载后上传到阿里云 OSS
     - 将文本 + embedding + OSS 图片 URL 写入 Milvus
  2. 从 data/pdf_output/ 下的各子文件夹读取 .txt 文件, 按父子文档切分后写入 Milvus
     - 子文件夹名 + ".pdf" 作为 source 字段
     - 对应的原始 PDF 文件上传到阿里云 OSS pdf_documents/ 文件夹, pdf_url 写入 Milvus
     - 按 parent 块（chunk_size=2000）→ child 块（chunk_size=400）两级切分
     - 父子回表: 父块与子块都写入 Milvus (doc_type 区分), 子块用 parent_id 指向父块;
       检索命中子块后回表取父块正文送 LLM
     - image_url 为空字符串, pdf_url 为 OSS 上的原始 PDF 访问地址
运行方式:
  python build_milvus_db.py                    # 全量建库 (pdf + jsonl + txt)
  python build_milvus_db.py --only-jsonl       # 仅处理 jsonl
  python build_milvus_db.py --only-pdf         # 仅处理 pdf
  python build_milvus_db.py --only-txt         # 仅处理 txt
  python build_milvus_db.py --rebuild          # 先删除同名旧集合再重建 (新增字段后必须加此参数)
  python build_milvus_db.py --skip-image       # 跳过图片生成 (jsonl 数据 image_url 为空)
注意：
  - Milvus 采用本地文件模式 (Milvus Lite), 连接地址读 MILVUS_ENDPOINT; 目标集合名读 MILVUS_COLLECTION
  - 所有外部依赖 (EMBEDDING_API_KEY, ARK_API_KEY[可选], Alibaba_ACCESS_KEY_ID 等) 从 .env 读取
  - 本脚本不修改任何已有代码文件, 仅复用已有模块
"""
import os
import sys
import json
import uuid
import time
import logging
import argparse
import tempfile
import requests
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 复用已有模块
from vectors import OpenAICompatEmbeddings, create_embedding_client
from milvus_schema import MilvusSchemaManager, COLLECTION_NAME
from testAliyunfiles import upload_to_oss, upload_pdf_to_oss, upload_txt_to_oss

load_dotenv()

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BuildMilvusDB")

# ===================== 全局配置 =====================
# 与运行时保持一致: 本地文件模式读 MILVUS_ENDPOINT, 目标集合名读 MILVUS_COLLECTION
# 变量名不用 MILVUS_URI, 因为 pymilvus 会读取同名环境变量并强制要求 http(s) 格式
MILVUS_URI = os.getenv("MILVUS_ENDPOINT") or os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", COLLECTION_NAME)
# 仅文生图需要 Ark (可选); embedding 走 OpenAI 兼容接口
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# 下面是不同批次入库的.jsonl数据,需要生成对应的图片
JSONL_FILE = "./data/data_20260704.jsonl"
# 下面是不同批次入库的PDF文件数据
PDF_OUTPUT_DIR = "./data/pdf_output"
PDF_DOCUMENTS_DIR = "./data/pdf_documents"
# 下面是不同批次入库的.txt数据
# MEDICAL_TEXTS_DIR = "./data/medical_texts"
MEDICAL_TEXTS_DIR = "./data/Ancient-Books"

# 图片生成模型
IMAGE_GEN_MODEL = "doubao-seedream-5-0-260128"
# IMAGE_GEN_MODEL = "doubao-seedream-4-5-251128"
# IMAGE_GEN_MODEL = "doubao-seedream-4-0-250828"
IMAGE_SIZE = "2K"

# 写入批次大小 (Milvus 单批写入条数; embedding 请求的批大小由 EMBEDDING_BATCH_SIZE 控制)
BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", 32))
BATCH_SLEEP = float(os.getenv("BUILD_BATCH_SLEEP", 0.1))   # 批次间休眠(秒)

# PDF 文本分割参数
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# ===================== 豆包文生图 (可选, 需 ARK_API_KEY) =====================
def generate_image_from_text(ark_client, text: str) -> Optional[str]:
    """
    调用豆包文生图模型，根据文本生成图片，返回火山云图片 URL。
    失败时返回 None，不中断主流程。
    """
    # 构造医学图片生成提示词（与 volc_embed.py 保持一致）
    template = (
        f"请根据以下文本描述生成一张医学展示图片:\n\n{text}\n\n"
        "图片内容应与描述高度相关，风格属于科学性质，用于医疗领域的展示，"
        "细节丰富，内容严谨，尽量少的在图片中插入文字，"
        "需要确保只写大号文字，不超过10个字，清晰可读。"
    )
    try:
        response = ark_client.images.generate(
            model=IMAGE_GEN_MODEL,
            prompt=template,
            size=IMAGE_SIZE,
            response_format="url",
            output_format="png",
            watermark=False,
            # guidance_scale=9.0,
        )
        image_url = response.data[0].url
        logger.debug(f"文生图成功: {image_url[:60]}...")
        return image_url
    except Exception as e:
        logger.warning(f"文生图失败: {e}")
        return None


# ===================== 图片转存：火山云 → 本地临时文件 → 阿里云 OSS =====================
def download_image_to_temp(volc_url: str) -> Optional[str]:
    """
    从火山云 URL 下载图片到本地临时文件，返回临时文件路径。
    失败时返回 None。
    """
    try:
        resp = requests.get(volc_url, timeout=30)
        resp.raise_for_status()
        # 从 URL 推断扩展名，默认 .jpeg 或者 .png
        url_path = volc_url.split("?")[0]  # 去掉查询参数
        suffix = Path(url_path).suffix or ".png"
        # 写入临时文件 (不自动删除, 由调用方负责清理)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        logger.debug(f"图片已下载到临时文件: {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.warning(f"图片下载失败 ({volc_url[:60]}...): {e}")
        return None


def transfer_image_to_oss(volc_url: str) -> Optional[str]:
    """
    将火山云图片 URL 对应的图片完整转存到阿里云 OSS。
    流程：下载到本地临时文件 → 上传 OSS → 删除临时文件 → 返回 OSS URL。
    失败时返回 None，不中断主流程。
    """
    tmp_path = None
    try:
        # 1: 下载到本地临时文件
        tmp_path = download_image_to_temp(volc_url)
        if not tmp_path:
            return None
        # 2: 上传到阿里云 OSS (严格复用 testAliyunfiles.upload_to_oss)
        oss_url = upload_to_oss(tmp_path)
        if oss_url:
            logger.debug(f"OSS 上传成功: {oss_url}")
        else:
            logger.warning("OSS 上传返回 None")
        return oss_url
    except Exception as e:
        logger.warning(f"图片转存失败: {e}")
        return None
    finally:
        # 3: 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ===================== Part 1: 处理 data.jsonl =====================
def process_jsonl(
    ark_client,
    schema_manager: MilvusSchemaManager,
    skip_image: bool = False,
) -> int:
    """
    读取 data/data.jsonl, 为每条数据生成图片并写入 Milvus
    :param ark_client:      豆包 Ark 客户端 (仅文生图使用, skip_image=True 时可为 None)
    :param schema_manager:  MilvusSchemaManager 实例
    :param skip_image:      True 时跳过图片生成, image_url 为空字符串
    :return: 成功写入的文档总数
    """
    logger.info(f"=== 开始处理 JSONL 数据: {JSONL_FILE} ===")
    if not os.path.exists(JSONL_FILE):
        logger.error(f"文件不存在: {JSONL_FILE}")
        return 0

    # 1: 读取所有行
    raw_items = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败, 跳过该行: {e}")
    logger.info(f"共读取 {len(raw_items)} 条 JSONL 数据")

    # 2: 逐条处理：生成图片 → 转存 OSS → 构建 Document
    docs: List[Document] = []
    for i, item in enumerate(tqdm(raw_items, desc="处理 JSONL")):
        query = item.get("query", "")
        response = item.get("response", "")
        text = f"{query}\n{response}"

        oss_url = ""
        if not skip_image and ark_client:
            # 2a: 文生图 (使用 response 文本，内容更丰富)
            volc_url = generate_image_from_text(ark_client, response or text)
            if volc_url:
                # 2b: 转存到阿里云 OSS
                oss_url = transfer_image_to_oss(volc_url) or ""
            if oss_url:
                logger.info(f"[{i+1}/{len(raw_items)}] 图片已上传 OSS: {oss_url}")
            else:
                logger.warning(f"[{i+1}/{len(raw_items)}] 图片生成/上传失败, image_url 置空")

        doc = Document(
            page_content=text,
            metadata={
                "doc_id":      str(uuid.uuid4()),
                "source":      JSONL_FILE,
                "chunk_index": i,
                "image_url":   oss_url,
                "pdf_url":     "",
            },
        )
        docs.append(doc)

    # 3: 批量写入 Milvus
    logger.info(f"开始将 {len(docs)} 条 JSONL 文档写入 Milvus...")
    total = schema_manager.insert_documents_with_images(
        docs,
        batch_size=BATCH_SIZE,
        sleep_between_batches=BATCH_SLEEP,
    )
    logger.info(f"=== JSONL 数据写入完成，共 {total} 条 ===")
    return total


# ===================== Part 2: 处理 PDF txt 文件 =====================
def load_txt_docs(txt_path: str, source: str) -> List[Document]:
    """
    读取单个 .txt 文件，返回单个 Document (整个文件作为一条原始文档)
    source 字段为子文件夹名 + ".pdf"
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
    except Exception as e:
        logger.error(f"读取 txt 失败 ({txt_path}): {e}")
        return []
    if not text_content:
        logger.warning(f"txt 文件内容为空: {txt_path}")
        return []
    doc = Document(
        page_content=text_content,
        metadata={
            "doc_id":      str(uuid.uuid4()),
            "source":      source,
            "chunk_index": 0,
            "image_url":   "",
            "pdf_url":     "",
        },
    )
    logger.info(f"从 {txt_path} 加载了 1 条原始文档 (字符数: {len(text_content)})")
    return [doc]


def split_pdf_docs(docs: List[Document], pdf_url: str = "") -> List[Document]:
    """
    对 PDF 原始文档进行两级切分 (parent 2000 → child 400), 并保留父块以支持"父子回表".
    返回的 Document 列表同时包含:
      - 父块 (doc_type="parent"): text 为父块正文, 自身 doc_id 作为子块 parent_id 的取值
      - 子块 (doc_type="child") : text 为子块正文, parent_id 指向其所属父块 doc_id
    image_url 为空, pdf_url 由调用方传入, 父子块共享同一 pdf_url
    :param docs:    load_txt_docs 返回的原始 Document 列表
    :param pdf_url: 原始 PDF 文件的 OSS URL (所有块共享)
    :return: 父块 + 子块 的 Document 列表
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
    parent_chunks = parent_splitter.split_documents(docs)
    logger.info(f"  parent 切分: {len(docs)} 条原始文档 → {len(parent_chunks)} 个 parent 块")

    parent_docs: List[Document] = []
    child_docs: List[Document] = []
    child_idx = 0
    for parent_idx, parent in enumerate(parent_chunks):
        # 父块: 单独成一条记录, doc_type="parent", 无上级 (parent_id 为空)
        parent_id = str(uuid.uuid4())
        source = parent.metadata.get("source", "")
        parent_docs.append(Document(
            page_content=parent.page_content,
            metadata={
                "doc_id":      parent_id,
                "parent_id":   "",
                "doc_type":    "parent",
                "source":      source,
                "image_url":   "",
                "pdf_url":     pdf_url,
                "chunk_index": parent_idx,
                "start_index": parent.metadata.get("start_index", 0),
            },
        ))
        # 子块: doc_type="child", parent_id 指向父块
        for child in child_splitter.split_documents([parent]):
            child.metadata["doc_id"] = str(uuid.uuid4())
            child.metadata["parent_id"] = parent_id
            child.metadata["doc_type"] = "child"
            child.metadata["source"] = source
            child.metadata["image_url"] = ""
            child.metadata["pdf_url"] = pdf_url
            child.metadata["chunk_index"] = child_idx
            child_docs.append(child)
            child_idx += 1
    logger.info(
        f"  child 切分完成: {len(parent_chunks)} 个 parent 块 → "
        f"{len(parent_docs)} 个 parent 记录 + {len(child_docs)} 个 child 块"
    )
    return parent_docs + child_docs


def process_pdf_output(schema_manager: MilvusSchemaManager) -> int:
    """
    遍历 data/pdf_output/ 下的所有子文件夹，读取 .txt 文件,
    将对应的原始 PDF 上传到阿里云 OSS, 按父子两级切分后写入 Milvus,
    每个 child chunk 的 pdf_url 指向 OSS 上的原始 PDF, image_url 为空
    :param schema_manager: MilvusSchemaManager 实例
    :return: 成功写入的文档总数
    """
    logger.info(f"=== 开始处理 PDF 数据: {PDF_OUTPUT_DIR} ===")
    pdf_root = Path(PDF_OUTPUT_DIR)
    if not pdf_root.exists():
        logger.error(f"目录不存在: {PDF_OUTPUT_DIR}")
        return 0

    pdf_docs_root = Path(PDF_DOCUMENTS_DIR)
    if not pdf_docs_root.exists():
        logger.warning(f"PDF 原文件目录不存在: {PDF_DOCUMENTS_DIR}, pdf_url 将为空")

    total_written = 0
    subdirs = [d for d in sorted(pdf_root.iterdir()) if d.is_dir()]
    if not subdirs:
        logger.warning(f"pdf_output 目录下没有子文件夹: {PDF_OUTPUT_DIR}")
        return 0

    for subdir in subdirs:
        source = subdir.name + ".pdf"
        logger.info(f"处理子文件夹: {subdir.name}  →  source={source}")

        # 1: 上传对应的原始 PDF 到阿里云 OSS, 获取 pdf_url
        # 先精确匹配, 再以子文件夹名为前缀模糊匹配 (处理文件名后缀不一致的情况)
        pdf_url = ""
        pdf_file = pdf_docs_root / source
        if not pdf_file.exists():
            # 模糊匹配: 找pdf_documents/下以子文件夹名开头的.pdf 文件
            candidates = list(pdf_docs_root.glob(f"{subdir.name}*.pdf"))
            if candidates:
                pdf_file = candidates[0]
                logger.info(f"  精确匹配失败, 使用模糊匹配: {pdf_file.name}")

        if pdf_file.exists():
            logger.info(f"  上传原始 PDF: {pdf_file.name}")
            pdf_url = upload_pdf_to_oss(str(pdf_file)) or ""
            if pdf_url:
                logger.info(f"  PDF 上传成功: {pdf_url}")
            else:
                logger.warning(f"  PDF 上传失败, pdf_url 置空")
        else:
            logger.warning(f"  未找到原始 PDF 文件: {subdir.name}.pdf, pdf_url 置空")

        # 2: 找到该子文件夹下的 .txt 文件
        txt_files = sorted(subdir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"子文件夹 {subdir.name} 下没有 .txt 文件, 跳过")
            continue

        for txt_path in txt_files:
            logger.info(f"  读取: {txt_path.name}")
            # 3: 加载原始文档
            raw_docs = load_txt_docs(str(txt_path), source=source)
            if not raw_docs:
                continue
            # 4: 两级切分 (含父块), 传入 pdf_url (所有块共享)
            docs_to_insert = split_pdf_docs(raw_docs, pdf_url=pdf_url)
            # 5: 写入 Milvus
            written = schema_manager.insert_documents_with_images(
                docs_to_insert,
                batch_size=BATCH_SIZE,
                sleep_between_batches=BATCH_SLEEP,
            )
            total_written += written
            logger.info(f"  已写入 {written} 个 parent/child 块")

    logger.info(f"=== PDF 数据写入完成, 共 {total_written} 个 parent/child 块 ===")
    return total_written


# ===================== Part 3: 处理 medical_texts .txt 文件 =====================
def process_medical_texts(schema_manager: MilvusSchemaManager) -> int:
    """
    遍历 data/medical_texts/ 下的所有 .txt 文件,
    将每个 .txt 文件上传到阿里云 OSS (medical_texts/ 文件夹),
    按父子两级切分后写入 Milvus (增量写入, 不影响已有数据),
    每个 child chunk 的 pdf_url 指向 OSS 上的原始 .txt 文件, image_url 为空
    :param schema_manager: MilvusSchemaManager 实例
    :return: 成功写入的文档总数
    """
    logger.info(f"=== 开始处理 medical_texts 数据: {MEDICAL_TEXTS_DIR} ===")
    txt_root = Path(MEDICAL_TEXTS_DIR)
    if not txt_root.exists():
        logger.error(f"目录不存在: {MEDICAL_TEXTS_DIR}")
        return 0

    txt_files = sorted(txt_root.glob("*.txt"))
    if not txt_files:
        logger.warning(f"medical_texts 目录下没有 .txt 文件: {MEDICAL_TEXTS_DIR}")
        return 0

    total_written = 0
    for txt_path in txt_files:
        source = txt_path.name  # 例如 "内科学.txt"
        logger.info(f"处理文件: {source}")

        # 1: 上传 .txt 文件到阿里云 OSS, 获取 txt_url (存入 pdf_url 字段)
        txt_url = ""
        logger.info(f"  上传 txt 文件: {txt_path.name}")
        txt_url = upload_txt_to_oss(str(txt_path)) or ""
        if txt_url:
            logger.info(f"  txt 文件上传成功: {txt_url}")
        else:
            logger.warning(f"  txt 文件上传失败，pdf_url 置空")

        # 2: 加载原始文档 (整个文件作为一条原始文档)
        raw_docs = load_txt_docs(str(txt_path), source=source)
        if not raw_docs:
            continue
        # 3: 两级切分 (含父块), 传入 txt_url (存入 pdf_url 字段, 所有块共享)
        docs_to_insert = split_pdf_docs(raw_docs, pdf_url=txt_url)
        # 4: 增量写入 Milvus
        written = schema_manager.insert_documents_with_images(
            docs_to_insert,
            batch_size=BATCH_SIZE,
            sleep_between_batches=BATCH_SLEEP,
        )
        total_written += written
        logger.info(f"  已写入 {written} 个 parent/child 块")

    logger.info(f"=== medical_texts 数据写入完成, 共 {total_written} 个 parent/child 块 ===")
    return total_written


# ===================== 主入口 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Milvus 建库脚本")
    parser.add_argument("--only-jsonl", action="store_true", help="仅处理 data.jsonl")
    parser.add_argument("--only-pdf",   action="store_true", help="仅处理 pdf_output")
    parser.add_argument("--only-txt", action="store_true", help="仅处理 txt (medical_texts/Ancient-Books)")
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="跳过图片生成 (jsonl 数据 image_url 置空, 加快建库速度)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先删除同名旧集合再重建 (新增 parent_id/doc_type 等字段后必须加此参数)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Milvus 建库脚本启动")
    logger.info(f"  MILVUS_URI   : {MILVUS_URI}")
    logger.info(f"  COLLECTION   : {MILVUS_COLLECTION}")
    logger.info(f"  only-jsonl   : {args.only_jsonl}")
    logger.info(f"  only-pdf     : {args.only_pdf}")
    logger.info(f"  only-txt     : {args.only_txt}")
    logger.info(f"  skip-image   : {args.skip_image}")
    logger.info(f"  rebuild      : {args.rebuild}")
    logger.info("=" * 60)

    # 初始化 Embedding 客户端 (OpenAI 兼容)
    if not EMBEDDING_API_KEY:
        logger.error("EMBEDDING_API_KEY 未配置, 请检查 .env 文件")
        sys.exit(1)
    embedding_client = create_embedding_client()

    # 文生图客户端 (可选): 仅在需要生成图片时才要求 ARK_API_KEY
    ark_client = None
    if not args.skip_image:
        if not ARK_API_KEY:
            logger.warning("ARK_API_KEY 未配置, 将跳过文生图 (image_url 置空)")
        else:
            from volcenginesdkarkruntime import Ark
            ark_client = Ark(api_key=ARK_API_KEY)

    # 初始化 MilvusSchemaManager (复用已有模块, 传入 OpenAI 兼容 embedding 客户端)
    # collection_name 与运行时保持一致; --rebuild 时先删除旧集合 (旧 schema 无 parent_id/doc_type)
    schema_manager = MilvusSchemaManager(
        uri=MILVUS_URI,
        embedding_client=embedding_client,
        collection_name=MILVUS_COLLECTION,
        drop_old=args.rebuild,
    )

    # 各数据源是否处理: 指定了 --only-xxx 时只处理对应源, 否则全部处理
    run_pdf = args.only_pdf or not (args.only_jsonl or args.only_txt)
    run_jsonl = args.only_jsonl or not (args.only_pdf or args.only_txt)
    run_txt = args.only_txt or not (args.only_jsonl or args.only_pdf)

    total = 0

    # Part 1: 处理 PDF
    if run_pdf:
        total += process_pdf_output(schema_manager=schema_manager)

    # Part 2: 处理 JSONL
    if run_jsonl:
        total += process_jsonl(
            ark_client=ark_client,
            schema_manager=schema_manager,
            skip_image=args.skip_image,
        )

    # Part 3: 处理 txt
    if run_txt:
        total += process_medical_texts(schema_manager=schema_manager)

    logger.info("=" * 60)
    logger.info(f"建库完成！本次共写入 {total} 条数据到 Milvus")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()