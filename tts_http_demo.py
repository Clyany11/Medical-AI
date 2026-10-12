"""
tts_http_demo.py — TTS 文本转语音 (生成音频文件)

被 app.py 的 /tts 接口调用:
    local_path = await tts_http_demo.tts_http_stream_async(text)
    随后 app.py 会将该文件上传 OSS 并清理本地临时文件。

流程: 文本 → 复用 protocols 双流协议调豆包 TTS → 聚合音频 → 写本地文件 → 返回路径

[缺口标注] 本模块为接入 app.py 而补充的实现:
  - 复用 tts_bidirection.py 依赖的 protocols 包 (同一套双向流协议), 不重复实现。
  - 音色/凭证从环境变量读取, 缺省取 asr_websocket.Config 的 app_key/access_key
    (占位符, 需填火山引擎真实凭证)。
  - 返回的是本地音频文件路径; 调用方负责后续上传与清理。
"""

import asyncio
import copy
import json
import logging
import os
import tempfile
import uuid

import websockets

from asr_websocket import Config as AsrConfig
from protocols import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

logger = logging.getLogger("TtsHttpDemo")

# 豆包 TTS 双向流 WebSocket 地址
TTS_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# 凭证 / 音色配置 (与 ASR 同一套火山引擎体系)
TTS_APP_ID = os.getenv("DOUBAO_APP_ID", AsrConfig().app_key)
TTS_ACCESS_TOKEN = os.getenv("DOUBAO_ACCESS_TOKEN", AsrConfig().access_key)
TTS_API_KEY = os.getenv("DOUBAO_API_KEY", "")
TTS_RESOURCE_ID = os.getenv("TTS_RESOURCE_ID", "seed-tts-1.0")
TTS_VOICE_TYPE = os.getenv("TTS_VOICE_TYPE", "zh_female_cancan_mars_bigtts")
TTS_ENCODING = os.getenv("TTS_ENCODING", "mp3")


async def tts_http_stream_async(
    text: str,
    voice_type: str = TTS_VOICE_TYPE,
    encoding: str = TTS_ENCODING,
) -> str:
    """
    将文本合成为音频文件, 返回本地文件路径。

    :param text:       待合成文本
    :param voice_type: 音色
    :param encoding:   音频格式 (mp3 / wav / pcm)
    :return: 本地音频文件绝对路径
    :raises ValueError: text 为空
    :raises RuntimeError: TTS 未返回音频数据
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text 不能为空")

    headers = {
        "X-Api-Resource-Id": TTS_RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    # 新版控制台: 只需 X-Api-Key; 旧版控制台: X-Api-App-Key + X-Api-Access-Key
    if TTS_API_KEY:
        headers["X-Api-Key"] = TTS_API_KEY
    else:
        headers["X-Api-App-Key"] = TTS_APP_ID
        headers["X-Api-Access-Key"] = TTS_ACCESS_TOKEN

    websocket = await websockets.connect(
        TTS_WS_URL, additional_headers=headers, max_size=10 * 1024 * 1024
    )

    audio_data = bytearray()
    try:
        # 1. 建立连接
        await start_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
        )

        # 2. 按句号分句, 每句开一个 session
        sentences = [s for s in text.split("。") if s.strip()] or [text]

        for sentence in sentences:
            base_request = {
                "user": {"uid": str(uuid.uuid4())},
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": voice_type,
                    "audio_params": {
                        "format": encoding,
                        "sample_rate": 24000,
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps({"disable_markdown_filter": False}),
                },
            }

            # 3. 开启 session
            session_id = str(uuid.uuid4())
            start_req = copy.deepcopy(base_request)
            start_req["event"] = EventType.StartSession
            await start_session(
                websocket, json.dumps(start_req).encode(), session_id
            )
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            # 4. 逐字发送文本
            async def send_chars(sent=sentence, sid=session_id, req=base_request):
                for char in sent:
                    synth_req = copy.deepcopy(req)
                    synth_req["event"] = EventType.TaskRequest
                    synth_req["req_params"]["text"] = char
                    await task_request(
                        websocket, json.dumps(synth_req).encode(), sid
                    )
                    await asyncio.sleep(0.005)
                await finish_session(websocket, sid)

            send_task = asyncio.create_task(send_chars())

            # 5. 接收音频数据
            while True:
                msg = await receive_message(websocket)

                if msg.type == MsgType.AudioOnlyServer:
                    audio_data.extend(msg.payload)
                elif msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        break
                    elif msg.event == EventType.SessionFailed:
                        logger.error("[TTS] session 失败: %s", msg.payload)
                        break

            await send_task

        # 6. 关闭连接
        await finish_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
        )
    finally:
        await websocket.close()

    if not audio_data:
        raise RuntimeError("TTS 未返回音频数据")

    # 7. 写入本地临时文件并返回路径 (由调用方负责后续清理)
    local_path = os.path.join(
        tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.{encoding}"
    )
    with open(local_path, "wb") as f:
        f.write(audio_data)

    logger.info("[TTS] 音频已生成: %s (%d bytes)", local_path, len(audio_data))
    return local_path
