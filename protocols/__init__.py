"""protocols 包 —— 字节跳动语音 WebSocket 二进制协议。

真实实现放在同目录的 protocols.py (课堂提供的开放源码, 未改动)。
此 __init__ 只做转发, 使 `from protocols import MsgType` 这类写法可用。
"""

from .protocols import (
    CompressionBits,
    EventType,
    HeaderSizeBits,
    Message,
    MsgType,
    MsgTypeFlagBits,
    SerializationBits,
    VersionBits,
    audio_only_client,
    cancel_session,
    finish_connection,
    finish_session,
    full_client_request,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

__all__ = [
    "CompressionBits",
    "EventType",
    "HeaderSizeBits",
    "Message",
    "MsgType",
    "MsgTypeFlagBits",
    "SerializationBits",
    "VersionBits",
    "audio_only_client",
    "cancel_session",
    "finish_connection",
    "finish_session",
    "full_client_request",
    "receive_message",
    "start_connection",
    "start_session",
    "task_request",
    "wait_for_event",
]
