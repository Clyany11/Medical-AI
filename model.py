"""model.py — LLM 客户端工厂 (DeepSeek)

提供:
  - deepseek_api_key / deepseek_base_url / deepseek_model: DeepSeek 配置
  - create_deepseek_client():       创建同步 DeepSeek 客户端 (OpenAI 兼容)
  - create_deepseek_async_client(): 创建异步 DeepSeek 客户端
  - generate_deepseek_answer():     DeepSeek 单轮非流式问答

说明:
  - DeepSeek 提供 OpenAI 兼容的 Chat Completions 接口 (联网搜索另见 web_search.py 用 Responses API)
  - Agent 主链路使用 app.py 内的异步客户端做流式生成, 本模块供脚本与兜底路径使用
  - 模型 ID 可用 DEEPSEEK_MODEL 覆盖 (默认 deepseek-flash; 深度思考用 deepseek-v4-pro)
  - V4 思考能力默认开启, 本模块对 flash 模型统一注入 thinking=disabled 以保持快路径
"""
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

logger = logging.getLogger("ModelFactory")

# ---- DeepSeek 配置 ----
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
# 深度思考模型 (think_type != "disabled" 时使用)
deepseek_reasoner_model = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-v4-pro")

DEFAULT_SYSTEM_PROMPT = "你是一个能力非常强大的助手."


# ==================== DeepSeek V4 思考模式控制 ====================
# V4 系列 (flash / pro) 的思考能力默认为【开启】, 通过请求体 thinking 参数切换:
#   {"thinking": {"type": "disabled"}} → 非思考模式 (等价旧 deepseek-chat, 更快更省)
#   {"thinking": {"type": "enabled"}}  → 思考模式    (等价旧 deepseek-reasoner)
# 本项目的快路径统一用 flash 且关闭思考, 与升级前的 deepseek-chat 行为保持一致;
# 传入非 flash 的模型 (如 deepseek-v4-pro) 时不注入, 保持其"思考默认开启".
def _thinking_extra_body(model: str) -> dict:
    if model and "flash" in model:
        return {"thinking": {"type": "disabled"}}
    return {}


class _ThinkingCompletions:
    """包装 chat.completions, 按 model 自动注入 DeepSeek V4 thinking 参数"""

    def __init__(self, inner):
        self._inner = inner

    async def create(self, *args, **kwargs):
        model = kwargs.get("model") or ""
        extra = dict(kwargs.get("extra_body") or {})
        for key, value in _thinking_extra_body(model).items():
            extra.setdefault(key, value)
        if extra:
            kwargs["extra_body"] = extra
        return await self._inner.create(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ThinkingChat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _ThinkingCompletions(inner.completions)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ThinkingAwareAsyncClient:
    """AsyncOpenAI 包装: 统一注入 V4 thinking 参数, 其余属性透传给底层客户端"""

    def __init__(self, inner: AsyncOpenAI):
        self._inner = inner
        self.chat = _ThinkingChat(inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def create_deepseek_client() -> OpenAI:
    """创建同步 DeepSeek 客户端"""
    return OpenAI(
        api_key=deepseek_api_key,
        base_url=deepseek_base_url,
    )


def create_deepseek_async_client() -> AsyncOpenAI:
    """创建异步 DeepSeek 客户端 (已按模型自动注入 V4 thinking 参数)"""
    return ThinkingAwareAsyncClient(
        AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
        )
    )


def generate_deepseek_answer(client, question: str, model: str = None) -> str:
    """DeepSeek 单轮非流式问答"""
    chosen = model or deepseek_model
    response = client.chat.completions.create(
        model=chosen,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stream=False,
        extra_body=_thinking_extra_body(chosen),
    )
    return response.choices[0].message.content