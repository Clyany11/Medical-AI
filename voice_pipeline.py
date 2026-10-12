"""
voice_pipeline.py — 语音全链路处理模块

功能概述:
 1. ASR (自动语音识别): 接收音频字节流, 调用豆包 ASR WebSocket API,
    实时流式返回识别文本, 最终拼接为完整 prompt
 2. RAG + LLM: 将识别出的 prompt 传入已有的 get_or_compute_async() 流程,
    查询 Milvus 并调用豆包大模型生成回答 (复用 app.py 已有逻辑, 不重复实现)
 3. TTS (文本转语音): 将大模型流式输出的文本实时传入豆包 TTS WebSocket API,
    边生成文本边合成语音, 音频数据以 base64 编码通过 SSE/WebSocket 推送给前端

整体数据流:

 手机端采集音频
     │ (WebSocket 上传 PCM/WAV 字节流)
     ▼
 /voice/asr ──→ AsrWsClient ──→ 实时识别文本 (SSE 流式推送给前端界面)
     │  (识别完成, 得到完整 prompt)
     ▼
 get_or_compute_async() ──→ Redis 缓存 / Milvus 检索 / 豆包 LLM
     │ (LLM 流式输出文本)
     ▼
 /voice/chat_tts ──→ TtsStreamClient ──→ 实时合成语音
     │ (音频 chunk base64 + 文本 chunk 同步推送)
     ▼
 手机端: 文字显示 + 语音播放

与已有代码的关系:
 - 不修改 app.py 中任何已有参数和路由
 - 复用 app.py 中的 get_or_compute_async(), async_doubao_client, milvus_manager
 - 复用 asr_websocket.py 中的 AsrWsClient (直接 import, 不复制代码)
 - 复用 tts_bidirection.py 中的 protocols 包 (直接 import)
 - 新路由通过 router 注册到 app.py, app.py 只需 include_router 一行

API 接口:
 POST /voice/asr      — 上传音频文件, 返回 SSE 流式识别文本
 POST /voice/chat_tts — 输入文本, 返回 SSE 流 (文本 chunk + 音频 base64 chunk)
 POST /voice/full_pipeline — 上传音频, 一次性完成 ASR→RAG→TTS 全链路, SSE 推送
 WS  /voice/asr_ws    — WebSocket 版 ASR (手机端实时推送音频帧)

=====================================================================
[接入说明] 本文件从课堂教材 Agent开发9.pdf 逐行重建 (PDF 中 CID 乱码
字形已按上下文复原)。已完成的接入处理:

 1. `token_billing` 模块 — 已补 token_billing.py 占位实现, import 不再报错。
 2. `from app import get_or_compute_async, stream_doubao_generator` —
    两个函数在 app.py 中已存在 (get_or_compute_async / stream_doubao_generator),
    此处采用延迟导入, 无循环依赖。
 3. ASR 临时文件 — 原教材写死 "/tmp/..." (Windows 下不存在),
    已改为 tempfile.gettempdir(), 兼容 Windows/Linux。
 4. 路由注册 — 已在 app.py 中
    `from voice_pipeline import router as voice_router` + `app.include_router(...)`。

仍需人工处理:
 5. asr_websocket.py 中 Config 的 app_key/access_key 仍是占位符
    (XXXXXXXXXX / ****), 需填入火山引擎豆包真实凭证。
 6. 依赖 websockets / aiohttp / python-multipart 需在运行环境安装。
 7. 文档字符串中提到的 WS /voice/asr_ws 接口本文件并未实现。
=====================================================================
"""

import asyncio
import base64
import copy
import json
import logging
import os
import tempfile
import uuid
from typing import AsyncGenerator, Optional

# Token 计费 (token_billing.py 为本项目补充的占位实现)
from token_billing import record_token_usage, estimate_tokens_from_chars

import aiohttp
import websockets
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── 复用 asr_websocket.py 中的所有类 (不重复实现) ──
from asr_websocket import (
    AsrWsClient,
    Config as AsrConfig,
    RequestBuilder,
    ResponseParser,
    CommonUtils,
    MessageType,
)

# ── 复用 tts_bidirection.py 依赖的 protocols 包 ──
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

logger = logging.getLogger("VoicePipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/voice", tags=["Voice Pipeline"])

# ===================== 配置常量 =====================

# ASR WebSocket 地址
ASR_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"

# TTS WebSocket 地址
TTS_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# TTS 配置 (从环境变量读取, 与 ASR 的 app_key/access_key 同一套体系)
TTS_APP_ID = os.getenv("DOUBAO_APP_ID", AsrConfig().app_key)
TTS_ACCESS_TOKEN = os.getenv("DOUBAO_ACCESS_TOKEN", AsrConfig().access_key)
# 音色必须与 X-Api-Resource-Id 配套:
#  volc.tts_bidirection.default → 普通双向流, 音色用 zh_female_cancan_mars_bigtts
TTS_VOICE_TYPE = os.getenv("TTS_VOICE_TYPE", "zh_female_cancan_mars_bigtts")
TTS_RESOURCE_ID = os.getenv("TTS_RESOURCE_ID", "seed-tts-1.0")

TTS_ENCODING = os.getenv("TTS_ENCODING", "mp3")  # mp3 / wav / pcm


# ===================== Pydantic 请求模型 =====================

class ChatTtsRequest(BaseModel):
    """POST /voice/chat_tts 请求体"""
    text: str                  # 直接输入文本 (跳过 ASR)
    user_id: str = "anonymous"
    session_id: Optional[str] = None  # 可复用已有 session_id

    voice_type: str = TTS_VOICE_TYPE
    encoding: str = TTS_ENCODING


class FullPipelineRequest(BaseModel):
    """POST /voice/full_pipeline 请求体 (JSON 部分, 音频通过 multipart 上传)"""
    user_id: str = "anonymous"
    voice_type: str = TTS_VOICE_TYPE
    encoding: str = TTS_ENCODING


# ===================== ASR 核心封装 =====================

async def asr_stream_from_bytes(
    audio_bytes: bytes,
    segment_duration: int = 200,
) -> AsyncGenerator[str, None]:
    """
    将音频字节流传入豆包 ASR, 实时 yield 识别出的文本片段.

    复用 asr_websocket.py 中的 AsrWsClient, 不重复实现协议细节.
    每次 ASR 返回一个 utterance（句子级别的识别结果），则 yield 一次文本。

    :param audio_bytes: WAV/PCM 格式的音频字节（16kHz, 16bit, mono）
    :param segment_duration: 每个音频分包的时长（毫秒），默认 200ms
    :yield: 识别出的文本片段（实时流式）
    """
    # 将字节写入临时文件（AsrWsClient.execute 接受文件路径）
    # 使用系统临时目录, 兼容 Windows/Linux (原教材写死 /tmp, Windows 下不存在)
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"asr_input_{uuid.uuid4().hex}.wav"
    )
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        async with AsrWsClient(ASR_WS_URL, segment_duration) as client:
            async for response in client.execute(tmp_path):
                if response.code != 0:
                    logger.error(f"ASR 错误码: {response.code}")
                    break

                payload = response.payload_msg
                if not payload:
                    continue

                # 提取 utterances 中的识别文本
                # 豆包 ASR 返回格式:{"result": {"utterances": [{"text": "...", "definite": true}]}}
                result = payload.get("result", {})
                utterances = result.get("utterances", [])
                for utt in utterances:
                    text = utt.get("text", "").strip()
                    if text:
                        # definite=True 表示整句已确认, False 表示中间结果
                        is_definite = utt.get("definite", False)
                        yield json.dumps({
                            "type":   "asr",
                            "text":   text,
                            "definite": is_definite,
                        }, ensure_ascii=False)

                # 最后一包: 返回完整识别结果
                if response.is_last_package:
                    full_text = result.get("text", "")
                    if full_text:
                        yield json.dumps({
                            "type": "asr_final",
                            "text": full_text,
                        }, ensure_ascii=False)
                    break

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def asr_get_full_text(audio_bytes: bytes) -> str:
    """
    对音频字节流执行 ASR, 等待识别完成, 返回完整文本字符串

    用于 full_pipeline 接口: 需完整识别, 再传入 RAG 流程

    :param audio_bytes: WAV/PCM 音频字节
    :return: 完整识别文本
    """
    full_text = ""
    async for chunk_json in asr_stream_from_bytes(audio_bytes):
        chunk = json.loads(chunk_json)
        if chunk["type"] == "asr_final":
            full_text = chunk["text"]
            break
        elif chunk["type"] == "asr" and chunk.get("definite"):
            # 已经确认的句子 (兜底: 如果没有 asr_final)
            full_text += chunk["text"]

    return full_text.strip()


# ===================== TTS 核心封装 =====================

async def tts_stream_from_text(
    text: str,
    voice_type: str = TTS_VOICE_TYPE,
    encoding: str = TTS_ENCODING,
) -> AsyncGenerator[str, None]:
    """
    将文本传入豆包 TTS, 实时 yield 音频 chunk (base64 编码)

    复用 tts_bidirection.py 的协议逻辑 (protocols 包), 不重复实现
    每收到一个音频包, 则 yield 一次 JSON:
      {"type": "audio", "data": "<base64>", "encoding": "mp3"}

    :param text:   待合成的文本
    :param voice_type: 音色 (默认 BV700_V2_streaming)
    :param encoding: 音频格式 (mp3 / wav / pcm)
    :yield: JSON 字符串, 包含 base64 编码的音频 chunk
    """
    headers = {
        "X-Api-App-Key": TTS_APP_ID,
        "X-Api-Access-Key": TTS_ACCESS_TOKEN,
        "X-Api-Resource-Id": TTS_RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    try:
        websocket = await websockets.connect(
            TTS_WS_URL,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,
        )
    except Exception as e:
        logger.error(f"TTS WebSocket 连接失败: {e}")
        yield json.dumps({"type": "error", "message": f"TTS 连接失败: {e}"})
        return

    try:
        # 1. 建立连接
        await start_connection(websocket)

        await wait_for_event(websocket, MsgType.FullServerResponse,
                             EventType.ConnectionStarted)

        # 2. 按句号分句, 每句开一个 session (与 tts_bidirection.py 保持一致)
        sentences = [s for s in text.split("。") if s.strip()]
        if not sentences:
            sentences = [text]  # 无句号时整段处理

        for sentence in sentences:

            if not sentence.strip():
                continue

            base_request = {
                "user": {"uid": str(uuid.uuid4())},
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": voice_type,
                    "audio_params": {
                        "format":       encoding,
                        "sample_rate":  24000,
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps({"disable_markdown_filter": False}),
                },
            }

            # 3. 开启 session
            session_id = str(uuid.uuid4())

            start_req = copy.deepcopy(base_request)
            start_req["event"] = EventType.StartSession
            await start_session(websocket, json.dumps(start_req).encode(),
                                session_id)
            await wait_for_event(websocket, MsgType.FullServerResponse,
                                 EventType.SessionStarted)

            # 4. 逐字发送文本 (与 tts_bidirection.py 保持一致, 5ms 间隔)
            async def send_chars(sent=sentence, sid=session_id, req=base_request):
                for char in sent:
                    synth_req = copy.deepcopy(req)
                    synth_req["event"] = EventType.TaskRequest
                    synth_req["req_params"]["text"] = char
                    await task_request(websocket, json.dumps(synth_req).encode(),
                                       sid)
                    await asyncio.sleep(0.005)
                await finish_session(websocket, sid)

            send_task = asyncio.create_task(send_chars())

            # 5. 接收音频数据, 实时 yield
            while True:
                msg = await receive_message(websocket)

                if msg.type == MsgType.AudioOnlyServer:
                    # 音频 chunk: base64 编码后推送给前端
                    audio_b64 = base64.b64encode(msg.payload).decode("utf-8")
                    yield json.dumps({
                        "type":   "audio",
                        "data":   audio_b64,
                        "encoding": encoding,
                    }, ensure_ascii=False)

                elif msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        break
                    elif msg.event == EventType.SessionFailed:
                        logger.error(f"TTS session 失败: {msg.payload}")
                        break

            await send_task

        # 6. 关闭连接
        await finish_connection(websocket)
        await wait_for_event(websocket, MsgType.FullServerResponse,
                             EventType.ConnectionFinished)

    except Exception as e:
        logger.error(f"TTS 流式处理失败: {e}")
        yield json.dumps({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


async def tts_stream_from_llm_stream(
    llm_text_generator: AsyncGenerator[str, None],
    voice_type: str = TTS_VOICE_TYPE,
    encoding: str = TTS_ENCODING,
) -> AsyncGenerator[str, None]:
    """
    将 LLM 流式输出的文本实时传入 TTS, 同步推送文本 chunk 和音频 chunk

    工作原理:
      - LLM 每输出一个文本 delta, 立即 yield {"type": "text", "delta": "..."}
      - 同时将文本缓存到缓冲区
      - 遇到句号/问号/感叹号时, 将缓冲区整段传入 TTS 合成
      - TTS 每返回一个音频包, yield {"type": "audio", "data": "<base64>"}

    这样前端可以同时:
      - 将 text delta 实时显示在界面上
      - 将 audio chunk 实时播放

    :param llm_text_generator: LLM 流式文本生成器 (yield str)
    :param voice_type:   TTS 音色
    :param encoding:     音频格式
    :yield: JSON 字符串 (type=text 或 type=audio)
    """
    # 句子结束标点 (遇到这些字符时触发 TTS 合成)
    SENTENCE_END_CHARS = {"。", "，", "？", ".", "!", "?", "\n"}

    text_buffer = ""
    full_text = ""

    # 建立 TTS WebSocket 连接 (复用整个会话, 避免每句重新建连)
    headers = {
        "X-Api-App-Key": TTS_APP_ID,
        "X-Api-Access-Key": TTS_ACCESS_TOKEN,
        "X-Api-Resource-Id": TTS_RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    websocket = None
    tts_ready = False
    try:
        ws_conn = websockets.connect(
            TTS_WS_URL,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,
        )
        websocket = await ws_conn if hasattr(ws_conn, "__await__") else ws_conn
        await start_connection(websocket)
        await wait_for_event(websocket, MsgType.FullServerResponse,
                             EventType.ConnectionStarted)
        tts_ready = True

    except Exception as e:
        logger.error(f"TTS 连接失败: {e}")

    async def synthesize_sentence(sentence: str) -> AsyncGenerator[str, None]:
        """对一个句子执行 TTS, yield 音频 chunk"""
        if not sentence.strip():
            return

        base_request = {
            "user": {"uid": str(uuid.uuid4())},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": voice_type,
                "audio_params": {
                    "format":   encoding,
                    "sample_rate": 24000,
                },
            },
        }

        session_id = str(uuid.uuid4())
        start_req = copy.deepcopy(base_request)
        start_req["event"] = EventType.StartSession
        await start_session(websocket, json.dumps(start_req).encode(), session_id)

        await wait_for_event(websocket, MsgType.FullServerResponse,
                             EventType.SessionStarted)

        async def send_chars():
            for char in sentence:
                req = copy.deepcopy(base_request)
                req["event"] = EventType.TaskRequest
                req["req_params"]["text"] = char
                await task_request(websocket, json.dumps(req).encode(), session_id)
                await asyncio.sleep(0.005)
            await finish_session(websocket, session_id)

        send_task = asyncio.create_task(send_chars())

        while True:
            msg = await receive_message(websocket)
            if msg.type == MsgType.AudioOnlyServer:
                audio_b64 = base64.b64encode(msg.payload).decode("utf-8")
                yield json.dumps({
                    "type":   "audio",
                    "data":   audio_b64,
                    "encoding": encoding,
                }, ensure_ascii=False)
            elif msg.type == MsgType.FullServerResponse:
                if msg.event in (EventType.SessionFinished, EventType.SessionFailed):
                    break

        await send_task

    try:
        # 消费 LLM 流式输出
        async for delta in llm_text_generator:
            if not delta:
                continue

            # 推送文本 delta 给前端 (实时显示)
            yield json.dumps({"type": "text", "delta": delta}, ensure_ascii=False)
            full_text += delta
            text_buffer += delta

            # 遇到句子结束符, 触发 TTS 合成 (TTS 不可用时跳过)
            if tts_ready and any(c in text_buffer for c in SENTENCE_END_CHARS):
                sentence_to_synth = text_buffer.strip()
                text_buffer = ""
                if sentence_to_synth:
                    async for audio_chunk in synthesize_sentence(sentence_to_synth):
                        yield audio_chunk

        # 处理缓冲区中剩余的文本 (最后一句可能没有标点)
        if tts_ready and text_buffer.strip():
            async for audio_chunk in synthesize_sentence(text_buffer.strip()):
                yield audio_chunk

        # 推送完成信号 (无论 TTS 是否可用, 始终输出)
        yield json.dumps({"type": "done", "full_text": full_text},
                         ensure_ascii=False)

    except Exception as e:
        logger.error(f"LLM→TTS 流式处理失败: {e}")
        yield json.dumps({"type": "error", "message": str(e)})
    finally:
        if websocket is not None:
            try:
                await finish_connection(websocket)
                await websocket.close()
            except Exception:
                pass


# ===================== SSE 工具函数 =====================

def sse_line(data: str) -> str:
    """将 JSON 字符串包装为 SSE 格式 (data: ...\\n\\n)"""
    return f"data: {data}\n\n"


# ===================== FastAPI 路由 =====================

@router.post("/asr", summary="音频文件 → 流式 ASR 识别文本")
async def voice_asr(
    file: UploadFile = File(..., description="WAV/MP3/M4A 音频文件"),
):
    """
    上传音频文件, 返回 SSE 流式识别文本

    前端接收格式 (SSE):

      data: {"type": "asr", "text": "高血压", "definite": false}
      data: {"type": "asr", "text": "高血压怎么治疗", "definite": true}
      data: {"type": "asr_final", "text": "高血压怎么治疗？"}

    - type=asr:   中间识别结果 (definite=false 为实时预测, true 为已确认)
    - type=asr_final: 最终完整识别文本 (用于后续传入 RAG 流程)

    手机端集成建议:
      - 将 definite=false 的文本实时显示 (灰色), definite=true 的文本固定 (黑色)
      - 收到 asr_final 后, 将 text 作为 prompt 调用 /voice/chat_tts 或 /voice/full_pipeline
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="音频文件为空")

    async def event_generator():
        async for chunk in asr_stream_from_bytes(audio_bytes):
            yield sse_line(chunk)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat_tts", summary="文本 → RAG + LLM + 流式 TTS")
async def voice_chat_tts(body: ChatTtsRequest):
    """
    输入文本, 经过 RAG + LLM 生成答案, 同时流式合成语音, SSE 推送文本和音频

    前端接收格式 (SSE):
      data: {"type": "text", "delta": "高血压患者"}
      data: {"type": "text", "delta": "应注意低盐饮食"}
      data: {"type": "audio", "data": "<base64_mp3>", "encoding": "mp3"}
      data: {"type": "text", "delta": "，每天食盐"}
      data: {"type": "audio", "data": "<base64_mp3>", "encoding": "mp3"}
      data: {"type": "done", "full_text": "高血压患者应注意低盐饮食..."}

    前端处理建议:
      - type=text: 将 delta 追加到界面文本框
      - type=audio: 将 base64 解码后追加到 AudioContext 播放队列
      - type=done: 显示完成状态, 可触发点赞/点踩按钮

    注意: 此接口直接接受文本输入, 通常由 /voice/asr 识别完成后调用
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    # 延迟导入, 避免循环依赖 (voice_pipeline 被 app.py include 后才能访问 app 级别的函数)
    from app import get_or_compute_async, stream_doubao_generator

    session_id = body.session_id or str(uuid.uuid4())

    async def event_generator():
        # 使用 stream_doubao_generator 获取 LLM 流式文本
        llm_gen = stream_doubao_generator(body.text)
        full_text_buf = []

        async for chunk in tts_stream_from_llm_stream(
            llm_gen,
            voice_type=body.voice_type,
            encoding=body.encoding,
        ):
            # 收集文本 chunk 用于 token 估算
            try:
                chunk_data = json.loads(chunk) if isinstance(chunk, str) else {}
                if chunk_data.get("type") == "text":
                    full_text_buf.append(chunk_data.get("delta", ""))
            except Exception:
                pass
            yield sse_line(chunk)

        # Token 计费: 按输出字符数估算语音 tokens
        import asyncio as _asyncio
        _asyncio.create_task(record_token_usage(
            user_id=body.user_id,
            session_id=session_id,
            text_tokens=0,
            voice_tokens=estimate_tokens_from_chars(len("".join(full_text_buf))),
            source="voice_chat_tts",
        ))

        # 推送 session_id, 供前端提交反馈
        yield sse_line(json.dumps({
            "type":    "session_id",
            "session_id": session_id,
        }))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/full_pipeline", summary="音频 → ASR → RAG+LLM → TTS 全链路")
async def voice_full_pipeline(
    file: UploadFile = File(..., description="WAV/MP3/M4A 音频文件"),
    user_id: str = Form(default="anonymous"),
    voice_type: str = Form(default=TTS_VOICE_TYPE),
    encoding: str = Form(default=TTS_ENCODING),
):
    """
    上传音频, 完成 ASR → RAG+LLM → TTS 全链路, SSE 推送所有中间结果

    前端接收格式 (SSE, 按时间):

      # 阶段1: ASR 识别
      data: {"type": "asr", "text": "高血压", "definite": false}
      data: {"type": "asr_final", "text": "高血压怎么治疗？"}

      # 阶段2: LLM 生成 + TTS 合成 (交替推送)
      data: {"type": "text", "delta": "高血压患者"}
      data: {"type": "audio", "data": "<base64>", "encoding": "mp3"}
      data: {"type": "text", "delta": "应注意低盐饮食"}
      data: {"type": "audio", "data": "<base64>", "encoding": "mp3"}

      # 阶段3: 完成
      data: {"type": "done", "full_text": "...", "session_id": "uuid"}

    这是手机端语音交互的主接口, 一次调用完成全部流程
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="音频文件为空")

    # 延迟导入, 避免循环依赖
    from app import stream_doubao_generator

    async def event_generator():
        # ── 阶段1: ASR 识别 ──
        recognized_text = ""
        async for chunk_json in asr_stream_from_bytes(audio_bytes):
            yield sse_line(chunk_json)
            chunk = json.loads(chunk_json)
            if chunk["type"] == "asr_final":
                recognized_text = chunk["text"]

        if not recognized_text:
            yield sse_line(json.dumps({"type": "error", "message": "ASR 识别结果为空"}))
            return

        # ── 阶段2: LLM 流式生成 + TTS 实时合成 ──
        session_id = str(uuid.uuid4())
        llm_gen = stream_doubao_generator(recognized_text)
        full_text_buf = []

        async for chunk in tts_stream_from_llm_stream(
            llm_gen,
            voice_type=voice_type,
            encoding=encoding,
        ):
            try:
                chunk_data = json.loads(chunk) if isinstance(chunk, str) else {}
                if chunk_data.get("type") == "text":
                    full_text_buf.append(chunk_data.get("delta", ""))
            except Exception:
                pass

            yield sse_line(chunk)

        # Token 计费: 按输出字符数估算语音 tokens
        import asyncio as _asyncio
        _asyncio.create_task(record_token_usage(
            user_id=user_id,
            session_id=session_id,
            text_tokens=0,
            voice_tokens=estimate_tokens_from_chars(len("".join(full_text_buf))),
            source="voice_full_pipeline",
        ))
        # ── 阶段3: 完成信号 ──
        yield sse_line(json.dumps({
            "type":    "done",
            "session_id": session_id,
            "asr_text": recognized_text,
        }, ensure_ascii=False))
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/tts_only", summary="纯文本 → 流式 TTS（不经过 LLM）")
async def voice_tts_only(body: ChatTtsRequest):
    """
    直接将输入文本合成语音, 不经过 RAG/LLM,
    用于测试 TTS 功能,或前端已有文本时直接播放
    前端接收格式 (SSE):
      data: {"type": "audio", "data": "<base64_mp3>", "encoding": "mp3"}
      ...（多个 audio chunk）
      data: {"type": "done"}
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    async def event_generator():
        async for chunk in tts_stream_from_text(
            body.text,
            voice_type=body.voice_type,
            encoding=body.encoding,
        ):
            yield sse_line(chunk)
        yield sse_line(json.dumps({"type": "done"}))
    return StreamingResponse(event_generator(), media_type="text/event-stream")
