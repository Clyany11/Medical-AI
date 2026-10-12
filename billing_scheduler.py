"""
billing_scheduler.py — Token 计费定时任务

被 app.py 的 lifespan 以 `asyncio.create_task(start_billing_scheduler())` 启动,
在后台周期性汇总 token 用量并生成账单。

[缺口标注] 本模块为接入 app.py 而补充的占位实现:
  - 真实实现应从 Redis / MySQL 聚合 record_token_usage() 写入的用量数据,
    按用户/会话出账并落库。
  - 当前仅打印日志, 不做任何持久化统计。

依赖: token_billing.record_token_usage 负责写入原始用量, 本模块负责周期汇总。
"""

import asyncio
import logging

logger = logging.getLogger("BillingScheduler")

# 计费周期: 每 24 小时执行一次
BILLING_INTERVAL_SECONDS = 24 * 60 * 60


async def _run_billing_once() -> None:
    """执行一次计费汇总 (占位)。"""
    # [缺口] 真实实现: 从存储中聚合 token 用量 -> 生成账单 -> 落库/告警
    logger.info("[Billing] 执行 Token 计费汇总任务 (占位实现, 未持久化)")


async def start_billing_scheduler(interval_seconds: int = BILLING_INTERVAL_SECONDS) -> None:
    """
    启动后台计费定时任务 (无限循环, 由 asyncio.create_task 调度)。

    :param interval_seconds: 计费周期, 默认 24 小时
    """
    logger.info("Token 计费定时任务已启动, 周期 %d 秒", interval_seconds)
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await _run_billing_once()
            except Exception as e:  # 单次失败不影响下一周期
                logger.error("Token 计费任务执行失败: %s", e)
    except asyncio.CancelledError:
        logger.info("Token 计费定时任务已取消")
        raise
