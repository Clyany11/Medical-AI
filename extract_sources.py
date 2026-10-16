"""
extract_sources.py — 从原始 PDF 重新抽取文本, 生成建库脚本所需的源文本目录结构

背景: data/ 目录曾整体丢失, 源文本 (.txt) 不存在; 原始 PDF 仍在.
本脚本用 pdfplumber 从原始 PDF 重新抽取 (多进程并行), 产出
build_medical_milvus.py / build_ancient_books.py 所需的目录结构:

  输入:
    - 现代医学 PDF: E:\\AI\\35agent\\现代医学\\*.pdf   (医学教材)
    - 古籍 PDF:     E:\\AI\\35agent\\古籍\\*.pdf        (古籍)

  输出 (相对本脚本所在目录):
    - data/pdf_output/<书名>/<书名>.txt     PDF 分支的正文 (source = <书名>.pdf)
    - data/pdf_documents/<书名>.pdf         原始 PDF 副本 (供上传 OSS)
    - data/Ancient-Books/<书名>.txt         古籍正文 (纯 txt 分支)

用法:
  python extract_sources.py                # 医学 + 古籍 全部抽取
  python extract_sources.py --only-medical
  python extract_sources.py --only-ancient
"""
import argparse
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber

# ===================== 路径配置 =====================
MEDICAL_PDF_DIR = Path(r"E:\AI\35agent\现代医学")   # 医学教材原始 PDF
ANCIENT_PDF_DIR = Path(r"E:\AI\35agent\古籍")        # 古籍原始 PDF
DATA_ROOT       = Path(__file__).resolve().parent / "data"
PDF_OUTPUT_DIR  = DATA_ROOT / "pdf_output"          # -> PDF_OUTPUT_DIR
PDF_DOCS_DIR    = DATA_ROOT / "pdf_documents"       # -> PDF_DOCUMENTS_DIR
ANCIENT_TXT_DIR = DATA_ROOT / "Ancient-Books"       # -> ANCIENT_BOOKS_DIR / MEDICAL_TEXTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractSources")


def extract_pdf_text(pdf_path: Path) -> str:
    """按页抽取 PDF 文本, 用换行拼接; 单页失败时跳过."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_one(task: Tuple[str, str, Optional[str]]) -> Tuple[str, int]:
    """子进程任务: 抽取单个 PDF 并落盘. task = (pdf_path, out_txt, copy_to)."""
    # 抑制 pdfminer 的字体告警 (字体缺 FontBBox 时刷屏, 不影响文本)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    pdf_path, out_txt, copy_to = task
    pdf_path = Path(pdf_path)
    out_txt = Path(out_txt)

    text = extract_pdf_text(pdf_path)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text, encoding="utf-8")

    if copy_to:
        Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, copy_to)

    return pdf_path.name, len(text)


def _run_tasks(tasks: List[Tuple[str, str, Optional[str]]], tag: str, workers: int) -> None:
    if not tasks:
        logger.error(f"[{tag}] 没有可处理的 PDF")
        return
    workers = max(1, min(len(tasks), workers))
    logger.info(f"[{tag}] {len(tasks)} 本, 并行进程数 {workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_one, t): t for t in tasks}
        for fut in as_completed(futures):
            name, chars = fut.result()
            flag = "  (无文本层!)" if chars == 0 else ""
            logger.info(f"[{tag}] 完成 {name}: {chars} 字符{flag}")


def process_medical(workers: int) -> None:
    """医学教材: 抽成 data/pdf_output/<书名>/<书名>.txt, 并复制 PDF 到 data/pdf_documents/."""
    pdfs = sorted(MEDICAL_PDF_DIR.glob("*.pdf"))
    tasks = []
    for p in pdfs:
        stem = p.stem
        out_txt = PDF_OUTPUT_DIR / stem / f"{stem}.txt"
        tasks.append((str(p), str(out_txt), str(PDF_DOCS_DIR / p.name)))
    _run_tasks(tasks, "医学", workers)


def process_ancient(workers: int) -> None:
    """古籍: 抽成 data/Ancient-Books/<书名>.txt."""
    pdfs = sorted(ANCIENT_PDF_DIR.glob("*.pdf"))
    tasks = [(str(p), str(ANCIENT_TXT_DIR / f"{p.stem}.txt"), None) for p in pdfs]
    _run_tasks(tasks, "古籍", workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从原始 PDF 抽取源文本")
    parser.add_argument("--only-medical", action="store_true", help="仅抽取医学教材")
    parser.add_argument("--only-ancient", action="store_true", help="仅抽取古籍")
    parser.add_argument("--workers", type=int, default=3,
                        help="并行进程数 (PDF 很大, 默认 3 以控制内存)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only = args.only_medical or args.only_ancient
    if args.only_medical or not only:
        process_medical(args.workers)
    if args.only_ancient or not only:
        process_ancient(args.workers)
    logger.info("=== 抽取完成 ===")


if __name__ == "__main__":
    main()
