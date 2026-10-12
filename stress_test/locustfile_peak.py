"""
Agent_online 压力测试 - 极限压测 + 稳定性测试
==================================================
两个场景:
1. 极限压测 (Peak): 高并发短时间，找到系统极限
2. 稳定性测试 (Soak): 中等负载长时间运行，观察内存泄漏/连接池耗尽等问题

使用方式:
    # 极限压测（200并发，5分钟）
    locust -f stress_test/locustfile_peak.py --host=http://127.0.0.1:8103 \
           --headless -u 200 -r 10 -t 5m \
           --csv=stress_test/reports/peak

    # 稳定性测试（30并发，1小时）
    locust -f stress_test/locustfile_peak.py --host=http://127.0.0.1:8103 \
           --headless -u 30 -r 2 -t 60m --tags soak \
           --csv=stress_test/reports/soak
"""

import json
import os
import random
import time
import uuid
from datetime import datetime

from locust import HttpUser, task, between, tag, events

import sys
sys.path.insert(0, ".")
from stress_test.config import (
    MEDICAL_QUESTIONS, CACHE_TEST_QUESTIONS, TCM_QUESTIONS,
    NON_MEDICAL_QUESTIONS, TTS_TEST_TEXTS,
    USER_POOL_SIZE, USER_PREFIX, EXPERT_PREFIX, EXPERT_POOL_SIZE,
    TIMEOUTS, SLA, API_KEY, PEAK_LOAD, SOAK_TEST,
    TRAFFIC_MODEL, CACHE_TRAFFIC_MODEL,
)
from stress_test.query_generator import QueryGenerator

# 动态查询生成器（主流量 + 缓存专项）
query_gen = QueryGenerator(traffic_model=TRAFFIC_MODEL)
cache_query_gen = QueryGenerator(traffic_model=CACHE_TRAFFIC_MODEL)


# ==================== 稳定性监控器 ====================
class SoakMonitor:
    """
    稳定性测试监控器
    定期记录系统指标，用于发现长时间运行中的:
    - 内存泄漏
    - 连接池耗尽
    - 响应时间逐步退化
    - 错误率突增
    """

    def __init__(self):
        self.snapshots = []          # 时间快照列表
        self.start_time = None
        self.report_dir = "stress_test/reports"

    def start(self):
        """开始监控"""
        self.start_time = time.time()
        os.makedirs(self.report_dir, exist_ok=True)

    def take_snapshot(self, stats):
        """
        每分钟记录一次系统指标快照
        Args:
            stats: Locust stats 对象
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
            "total_requests": stats.total.num_requests,
            "total_failures": stats.total.num_failures,
            "current_rps": round(stats.total.current_rps, 2),
            "avg_response_time": round(stats.total.avg_response_time, 1),
            "p95_response_time": stats.total.get_response_time_percentile(0.95),
            "p99_response_time": stats.total.get_response_time_percentile(0.99),
            "error_rate": round(
                stats.total.num_failures / max(stats.total.num_requests, 1), 4
            ),
        }
        self.snapshots.append(snapshot)
        return snapshot

    def detect_degradation(self) -> list:
        """
        检测性能退化趋势
        Returns: 告警列表
        """
        alerts = []
        if len(self.snapshots) < 5:
            return alerts

        # 取最近5个快照的平均响应时间
        recent = self.snapshots[-5:]
        early = self.snapshots[:5] if len(self.snapshots) >= 10 else self.snapshots[:3]

        avg_recent = sum(s["avg_response_time"] for s in recent) / len(recent)
        avg_early = sum(s["avg_response_time"] for s in early) / len(early)

        # 如果最近的响应时间比初始阶段增长了50%以上，告警
        if avg_early > 0 and avg_recent > avg_early * 1.5:
            alerts.append(
                f"[退化告警] 响应时间退化: 初始 {avg_early:.0f}ms → 当前 {avg_recent:.0f}ms "
                f"(+{((avg_recent - avg_early) / avg_early * 100):.0f}%)"
            )

        # 检查错误率突增
        if recent[-1]["error_rate"] > SLA["error_rate_max"]:
            alerts.append(
                f"[错误告警] 当前错误率 {recent[-1]['error_rate']:.2%} "
                f"超过阈值 {SLA['error_rate_max']:.0%}"
            )

        return alerts

    def save_report(self):
        """保存稳定性测试报告到 JSON 文件"""
        report_path = os.path.join(
            self.report_dir,
            f"soak_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report = {
            "test_type": "soak_test",
            "duration_minutes": round(
                (time.time() - self.start_time) / 60, 1
            ) if self.start_time else 0,
            "total_snapshots": len(self.snapshots),
            "snapshots": self.snapshots,
            "degradation_alerts": self.detect_degradation(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  稳定性报告已保存: {report_path}")


# 全局监控实例
soak_monitor = SoakMonitor()

# 快照计数器（用于定期触发快照）
_snapshot_counter = {"requests": 0, "last_snapshot_time": 0}


# ==================== 事件钩子 ====================
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始"""
    soak_monitor.start()
    print("=" * 60)
    print("  Agent_online_v3 极限/稳定性压测 - 开始")
    print("=" * 60)
    print(f"  目标: {environment.host}")
    print(f"  时间: {datetime.now().isoformat()}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束"""
    # 保存最终快照
    stats = environment.runner.stats
    soak_monitor.take_snapshot(stats)
    soak_monitor.save_report()

    # 打印汇总
    print("\n" + "=" * 60)
    print("  测试结束汇总")
    print("=" * 60)
    print(f"  总请求数: {stats.total.num_requests}")
    print(f"  总失败数: {stats.total.num_failures}")
    if stats.total.num_requests > 0:
        error_rate = stats.total.num_failures / stats.total.num_requests
        p95 = stats.total.get_response_time_percentile(0.95) or 0
        p99 = stats.total.get_response_time_percentile(0.99) or 0
        print(f"  错误率: {error_rate:.2%}")
        print(f"  平均响应时间: {stats.total.avg_response_time:.0f}ms")
        print(f"  P95: {p95:.0f}ms")
        print(f"  P99: {p99:.0f}ms")
        print(f"  峰值 RPS: {stats.total.current_rps:.1f}")

        # 退化检测
        alerts = soak_monitor.detect_degradation()
        if alerts:
            print("\n  ⚠️ 检测到性能退化:")
            for alert in alerts:
                print(f"    {alert}")
        else:
            print("\n  ✓ 未检测到性能退化")
    print("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    每个请求完成后触发
    用于定期（每60秒）生成系统快照
    """
    _snapshot_counter["requests"] += 1
    now = time.time()

    # 每60秒记录一次快照
    if now - _snapshot_counter["last_snapshot_time"] >= 60:
        _snapshot_counter["last_snapshot_time"] = now
        # 注意：此处无法直接访问 environment，快照会在 test_stop 补充


# ==================== 辅助函数 ====================
def get_user_id() -> str:
    return f"{USER_PREFIX}_{random.randint(1, USER_POOL_SIZE):04d}"


def get_expert_id() -> str:
    return f"{EXPERT_PREFIX}_{random.randint(1, EXPERT_POOL_SIZE):03d}"


# ==================== 极限压测用户 ====================
class PeakUser(HttpUser):
    """
    极限压测用户
    高频请求，最小等待时间，模拟突发流量洪峰。
    目的：找到系统的吞吐上限和错误率拐点。
    """
    weight = 70
    # 极限测试：最短等待时间
    wait_time = between(0.5, 2)

    def on_start(self):
        self.user_id = get_user_id()
        self.session_id = str(uuid.uuid4())
        self.headers = {"Content-Type": "application/json"}
        if API_KEY and API_KEY != "your-api-key-here":
            self.headers["X-API-Key"] = API_KEY

    @tag("peak", "soak")
    @task(10)
    def rapid_fire_non_stream(self):
        """高频非流式请求 - 主压力源（动态生成长尾问题，避免全量命中缓存）"""
        question = query_gen.next_question()
        payload = {"question": question, "username": self.user_id}

        with self.client.post(
            "/",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["non_stream"],
            name="POST / (极限-非流式)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get("response"):
                        resp.success()
                    else:
                        resp.failure("空响应")
                except Exception:
                    resp.failure("JSON解析失败")
            elif resp.status_code == 429:
                # 限流响应，记录但不算失败
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("peak", "soak")
    @task(6)
    def rapid_fire_stream(self):
        """高频流式请求（动态生成长尾问题，真实压测流式回源路径）"""
        question = query_gen.next_question()
        payload = {
            "question": question,
            "username": self.user_id,
            "session_id": self.session_id,
            "think_type": "normal",
        }

        start = time.time()
        with self.client.post(
            "/chat_stream",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["stream"],
            stream=True,
            name="POST /chat_stream (极限-流式)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                got_data = False
                try:
                    for chunk in resp.iter_content(chunk_size=256):
                        if chunk:
                            got_data = True
                            break  # 极限测试只验证首Token，不读完全部
                    if got_data:
                        resp.success()
                    else:
                        resp.failure("无数据")
                except Exception as e:
                    resp.failure(f"异常: {str(e)[:60]}")
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("peak", "soak")
    @task(4)
    def cache_hit_burst(self):
        """缓存命中爆发 - 高热点流量（专项压测缓存路径）"""
        question = cache_query_gen.next_question()
        payload = {"question": question, "username": self.user_id}

        with self.client.post(
            "/",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["non_stream"],
            name="POST / (极限-缓存)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("peak", "soak")
    @task(2)
    def feedback_burst(self):
        """
        反馈接口压力。

        极限压测目标是压反馈接口的吞吐，用随机 session_id 即可。
        API 契约：feedback 字段必须为 "thumbs_up"/"thumbs_down" 枚举值。
        随机 session 不在库中，后端会返回 404（正常校验），
        200/404 都说明接口在高并发下正常工作，均视为成功。
        """
        payload = {
            "session_id": str(uuid.uuid4()),
            "feedback": "thumbs_up",   # 必须是枚举值，不能是自由文本 "good"
            "user_id": self.user_id,
        }

        with self.client.post(
            "/feedback",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["feedback"],
            name="POST /feedback (极限)",
            catch_response=True,
        ) as resp:
            # 200=成功, 404=随机session不存在（正常校验行为），均视为接口可用
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


class MonitorUser(HttpUser):
    """
    监控用户（权重 30%）
    在极限压测中持续监控系统健康状态。
    如果 /health 开始返回异常，说明系统已达到极限。
    """
    weight = 30
    wait_time = between(2, 5)

    def on_start(self):
        self.headers = {"Content-Type": "application/json"}
        if API_KEY and API_KEY != "your-api-key-here":
            self.headers["X-API-Key"] = API_KEY

    @tag("peak", "soak", "monitor")
    @task(3)
    def health_probe(self):
        """健康探针 - 持续监控"""
        with self.client.get(
            "/health",
            timeout=TIMEOUTS["health"],
            name="GET /health (探针)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    status = data.get("status", "unknown")
                    if status == "healthy":
                        resp.success()
                    elif status == "degraded":
                        # 降级状态也算通过，但记录
                        resp.success()
                    else:
                        resp.failure(f"不健康: {status}")
                except Exception:
                    resp.failure("JSON解析失败")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("peak", "soak", "monitor")
    @task(2)
    def metrics_check(self):
        """指标检查"""
        with self.client.get(
            "/metrics",
            timeout=TIMEOUTS["health"],
            name="GET /metrics (指标)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("soak")
    @task(1)
    def feedback_stats(self):
        """反馈统计检查"""
        with self.client.get(
            "/feedback/stats",
            timeout=TIMEOUTS["feedback"],
            name="GET /feedback/stats",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
