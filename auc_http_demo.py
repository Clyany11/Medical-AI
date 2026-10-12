"""
auc_http_demo.py — ASR 语音识别 (音频 URL → 文本)

被 app.py 的 /chat_stream 接口同步调用:
    query = auc_http_demo.main_voice(voiceAdr)

流程: 下载音频 URL → 复用 asr_websocket.AsrWsClient 调豆包 ASR → 返回完整文本

[缺口标注] 本模块为接入 app.py 而补充的实现:
  - 底层复用项目已有的 asr_websocket.py, 不重复实现豆包 ASR 二进制协议。
  - 豆包凭证仍取 asr_websocket.Config 中的 app_key / access_key (占位符, 需填真实值)。
  - 非 WAV 音频依赖系统安装 ffmpeg (asr_websocket 转换逻辑)。
  - main_voice 为同步接口, 内部在线程中新建事件循环运行, 以兼容
    在 FastAPI async 路由内被同步调用 (不可在运行中的事件循环里直接 asyncio.run)。
"""

import asyncio
import concurrent.futures
import logging
import os
import tempfile
import uuid

import aiohttp

from asr_websocket import AsrWsClient

logger = logging.getLogger("AucHttpDemo")

# 豆包 ASR WebSocket 地址 (非流式, 一次性返回)
ASR_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"

# 单个音频分包时长 (毫秒)
ASR_SEGMENT_DURATION = 200


def main_voice(voice_url: str) -> str:
    """
    同步入口: 将语音 URL 识别为文本。

    :param voice_url: 音频文件 URL (如 OSS 链接)
    :return: 识别出的完整文本; 失败或无输入时返回 ""
    """
    if not voice_url:
        logger.warning("[ASR] voice_url 为空, 跳过识别")
        return ""

    # 在独立线程中新建事件循环运行, 兼容 sync / async 调用场景
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(_main_voice_async(voice_url)))
        try:
            return future.result()
        except Exception as e:
            logger.error("[ASR] 语音识别失败: %s", e)
            return ""


async def _main_voice_async(voice_url: str) -> str:
    """下载音频 → ASR 识别 → 清理临时文件。"""
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"asr_url_{uuid.uuid4().hex}.wav"
    )
    try:
        await _download_file(voice_url, tmp_path)
        return await _recognize(tmp_path)
    finally:
        # asr_websocket 在格式转换时可能已删除原文件, 故忽略不存在的情况
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


async def _download_file(url: str, dest: str) -> None:
    """从 URL 下载音频到本地临时文件。"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
            with open(dest, "wb") as f:
                f.write(content)
    logger.info("[ASR] 音频已下载: %s (%d bytes)", dest, len(content))


async def _recognize(file_path: str) -> str:
    """调用豆包 ASR, 返回完整识别文本。"""
    full_text = ""
    async with AsrWsClient(ASR_WS_URL, ASR_SEGMENT_DURATION) as client:
        async for response in client.execute(file_path):
            if response.code != 0:
                logger.error("[ASR] 错误码: %s", response.code)
                break

            payload = response.payload_msg or {}
            result = payload.get("result", {})
            text = result.get("text")
            if text:
                full_text = text

            if response.is_last_package:
                break

    full_text = full_text.strip()
    logger.info("[ASR] 识别完成: %s", full_text)
    return full_text
