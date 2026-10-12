"""
middleware.py — 审计中间件 + 服务降级统计 + 扩展模块注册表

被 app.py 使用:
  - `app.add_middleware(AuditMiddleware, skip_paths=[...])`
  - `fallback_manager.record_fallback("full_fallback")`
  - `fallback_manager.get_fallback_stats()`
  - `await extension_registry.health_check_all()`
  - `extension_registry.list_modules()`

[缺口标注] 本模块为接入 app.py 而补充的实现:
  - AuditMiddleware: 记录每个请求的方法/路径/状态码/耗时 (日志审计)。
  - FallbackManager: 内存计数器, 记录服务降级事件次数 (进程内, 未持久化)。
  - ExtensionRegistry: 扩展模块注册与健康检查; 默认空注册表,
    如需纳入具体模块可调用 extension_registry.register(name, checker)。
"""

import asyncio
import inspect
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("Middleware")


# ===================== 审计中间件 =====================

class AuditMiddleware(BaseHTTPMiddleware):
    """记录所有非白名单请求的审计日志 (方法 / 路径 / 状态码 / 耗时)。"""

    def __init__(self, app, skip_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.skip_paths = set(skip_paths or [])

    async def dispatch(self, request, call_next):
        path = request.url.path
        # 白名单路径直接放行, 不记录审计日志
        if path in self.skip_paths:
            return await call_next(request)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "[Audit] %s %s -> 异常 (%.1fms)", request.method, path, elapsed_ms
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[Audit] %s %s -> %s (%.1fms)",
            request.method, path, response.status_code, elapsed_ms,
        )
        return response


# ===================== 服务降级统计 =====================

class FallbackManager:
    """进程内降级事件计数器 (线程安全)。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)

    def record_fallback(self, name: str) -> None:
        """记录一次降级事件, name 例如 "full_fallback"。"""
        with self._lock:
            self._stats[name] += 1
        logger.warning("[Fallback] 触发降级: %s (累计 %d 次)", name, self._stats[name])

    def get_fallback_stats(self) -> Dict[str, Any]:
        """返回降级统计: {"total": n, "by_type": {...}}"""
        with self._lock:
            by_type = dict(self._stats)
        return {"total": sum(by_type.values()), "by_type": by_type}


# ===================== 扩展模块注册表 =====================

class ExtensionRegistry:
    """扩展模块注册与健康检查。"""

    def __init__(self):
        self._modules: Dict[str, Optional[Callable[[], Any]]] = {}

    def register(self, name: str, health_check: Optional[Callable[[], Any]] = None) -> "ExtensionRegistry":
        """
        注册一个扩展模块。

        :param name: 模块名
        :param health_check: 健康检查函数, 可为同步或异步 (无参) Callable; 可为 None
        """
        self._modules[name] = health_check
        return self

    def list_modules(self) -> List[str]:
        """返回已注册模块名列表。"""
        return list(self._modules.keys())

    async def health_check_all(self) -> Dict[str, Any]:
        """
        对所有已注册模块执行健康检查。

        :return: {模块名: 检查结果}; 未提供检查函数时返回 "unknown";
                 检查抛异常时返回 "error: ..."
        """
        results: Dict[str, Any] = {}
        for name, checker in self._modules.items():
            if checker is None:
                results[name] = "unknown"
                continue
            try:
                if inspect.iscoroutinefunction(checker):
                    results[name] = await checker()
                else:
                    results[name] = checker()
            except Exception as e:
                results[name] = f"error: {e}"
        return results


# ===================== 单例 =====================

fallback_manager = FallbackManager()
extension_registry = ExtensionRegistry()
