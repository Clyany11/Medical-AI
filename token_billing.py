"""
Token 计费模块 —— 占位实现

背景:
  语音链路 voice_pipeline.py 顶部执行
      from token_billing import record_token_usage, estimate_tokens_from_chars
  但课堂教材/仓库中并未提供 token_billing.py 实体文件 (39agent/app.py
  也只是引用它), 缺失会导致 import voice_pipeline 直接 ImportError。

  此处补一份最小可用实现, 保证语音链路能正常 import 与运行。
  若后续拿到官方/课程版的 token_billing.py, 直接用其覆盖本文件即可。

[缺口标注] 本模块仅打印日志, 不做任何持久化统计。
"""

import logging

logger = logging.getLogger("TokenBilling")

# 中文场景粗略估算: 约 1.5 个字符 ≈ 1 token (仅用于计费统计, 非精确值)
_CHARS_PER_TOKEN = 1.5


def estimate_tokens_from_chars(char_count: int) -> int:
    """按字符数粗略估算 token 数"""
    if char_count <= 0:
        return 0
    return int(char_count / _CHARS_PER_TOKEN)


async def record_token_usage(
    user_id: str,
    session_id: str,
    text_tokens: int = 0,
    voice_tokens: int = 0,
    source: str = "unknown",
) -> None:
    """记录一次 token 用量 (占位实现: 仅输出日志, 不落库)"""
    logger.info(
        "token_usage | user=%s session=%s text=%d voice=%d source=%s",
        user_id, session_id, text_tokens, voice_tokens, source,
    )
