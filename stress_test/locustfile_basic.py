"""

Agent_online 压力测试 - 基础压测脚本
==========================================
使用 Locust 框架, 覆盖所有核心接口的基础压测
包含: 非流式问答,  流式问答,  反馈, 健康检查,  缓存验证

使用方式:
    # Web UI 模式 (推荐, 可实时观察)
    locust -f stress_test/locustfile_basic.py --host=http://127.0.0.1:8103

    # 无头模式 (CI/CD 集成)
    locust -f stress_test/locustfile_basic.py --host=http://127.0.0.1:8103 --headless -u 50 -r 5 -t 5m --csv=stress_test/reports/basic
"""

import json
import random
import time

import uuid
from typing import Optional

from locust import HttpUser, task, between, tag, events
from locust.runners import MasterRunner

# 导入压测配置
import sys
sys.path.insert(0, ".")
from stress_test.config import (

    MEDICAL_QUESTIONS, CACHE_TEST_QUESTIONS, TCM_QUESTIONS,
    NON_MEDICAL_QUESTIONS, TTS_TEST_TEXTS,
    USER_POOL_SIZE, EXPERT_POOL_SIZE, USER_PREFIX, EXPERT_PREFIX,
    TIMEOUTS, SLA, API_KEY, TRAFFIC_MODEL, CACHE_TRAFFIC_MODEL, )
from stress_test.query_generator import QueryGenerator

# 动态查询生成器: 主流量用真实流量模型 (长尾为主, 缓存命中率贴近线上)
query_gen = QueryGenerator(traffic_model=TRAFFIC_MODEL)

# 缓存专项生成器:  高热点占比,  用于验证缓存命中路径
cache_query_gen = QueryGenerator(traffic_model=CACHE_TRAFFIC_MODEL)


# ==================== 全局指标收集器 ====================
class MetricsCollector:
    """收集自定义指标 (TTFT, 缓存命中率等)"""

    def __init__(self):
        self.ttft_samples = []   # 流式首Token延迟

        self.cache_hits = 0      # 缓存命中次数
        self.cache_misses = 0    # 缓存未命中次数
        self.error_count = 0     # 错误计数
        self.total_requests = 0  # 总请求数

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hits + self.cache_misses

        return self.cache_hits / total if total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """错误率"""
        return self.error_count / self.total_requests if self.total_requests > 0 else 0.0

    def reset(self):
        """重置所有指标"""
        self.ttft_samples.clear()
        self.cache_hits = 0
        self.cache_misses = 0

        self.error_count = 0
        self.total_requests = 0


# 全局指标实例
metrics = MetricsCollector()


# ==================== 事件钩子 ====================
@events.test_start.add_listener

def on_test_start(environment, **kwargs):
    """压测开始时的初始化"""
    print("=" * 60)
    print(" Agent_online_v3 基础压力测试 - 开始")
    print("=" * 60)
    print(f" 目标地址: {environment.host}")
    print(f" 流量模型: 长尾{TRAFFIC_MODEL['longtail_disease'] +TRAFFIC_MODEL['longtail_symptom']:.0%} "
            f"热点{TRAFFIC_MODEL['hot']:.0%} 中医{TRAFFIC_MODEL['tcm']:.0%} "

            f"非医学{TRAFFIC_MODEL['non_medical']:.0%}")
    print(f" (长尾问题动态生成走回源,缓存命中率预期 30~45%)")
    print(f" SLA 要求:")
    print(f" - 非流式 P95 < {SLA['non_stream_p95_ms']}ms")
    print(f" - 流式TTFT P95 < {SLA['stream_ttft_p95_ms']}ms")
    print(f" - 缓存命中  P95 < {SLA['cache_hit_p95_ms']}ms")
    print(f" - 错误率 < {SLA['error_rate_max'] * 100}%")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """压测结束时的汇总报告"""
    print("\n" + "=" * 60)
    print(" 压测结束 - 自定义指标汇总")
    print("=" * 60)
    print(f" 总请求数: {metrics.total_requests}")
    print(f" 错误数: {metrics.error_count}")
    print(f" 错误率: {metrics.error_rate:.2%}")

    print(f" 缓存命中率: {metrics.cache_hit_rate:.2%}")
    if metrics.ttft_samples:
        sorted_ttft = sorted(metrics.ttft_samples)
        p50 = sorted_ttft[int(len(sorted_ttft) * 0.5)]
        p95 = sorted_ttft[int(len(sorted_ttft) * 0.95)]
        p99 = sorted_ttft[min(int(len(sorted_ttft) * 0.99), len(sorted_ttft) - 1)]

        print(f" 流式TTFT P50: {p50:.0f}ms")
        print(f" 流式TTFT P95: {p95:.0f}ms")
        print(f" 流式TTFT P99: {p99:.0f}ms")
    # 查询多样性统计(验证长尾流量是否真实模拟)
    gen_stats = query_gen.stats()
    print(f" ── 查询多样性 ──")
    print(f" 主流量生成问题数:  {gen_stats['total_generated']}")
    print(f" 唯一问题数: {gen_stats['unique_questions']}")

    print(f" 唯一率: {gen_stats['unique_ratio']:.1%} (越高说明长尾越丰富)")
    print("=" * 60)


# ==================== 辅助函数 ====================
def get_random_user_id() -> str:
    """生成随机用户 ID"""
    return f"{USER_PREFIX}_{random.randint(1, USER_POOL_SIZE):04d}"


def get_random_expert_id() -> str:
    """生成随机专家 ID"""
    return f"{EXPERT_PREFIX}_{random.randint(1, EXPERT_POOL_SIZE):03d}"


def get_random_question(category: str = "medical") -> str:
    """
    获取随机问题。

    - category="medical"(默认):走动态生成器,按真实流量模型混合
        长尾问题(回源)  + 热点问题(命中缓存),缓存命中率贴近线上真实值。
    - category="cache":走缓存专项生成器(高热点占比),用于验证缓存路径。
    - 其余类别保留固定问题池,用于特定验证。
    """
    if category == "medical":
        return query_gen.next_question()
    if category == "cache":
        return cache_query_gen.next_question()
    pools = {

        "tcm": TCM_QUESTIONS,
        "non_medical": NON_MEDICAL_QUESTIONS, }
    return random.choice(pools.get(category, MEDICAL_QUESTIONS))


#  AI课堂小朱老师独家讲义
# ==================== 用户行为模型 ====================
class MedicalUser(HttpUser):

    """
    医学问答用户  - 主要用户类型(权重 60%)
    模拟真实用户行为:提问 → 阅读 → 反馈 → 再提问
    """
    weight = 60
    # 模拟用户思考时间:2~8秒(真实用户阅读答案需要时间)
    wait_time = between(2, 8)

    def on_start(self):
        """用户初始化:分配用户ID和会话ID"""
        self.user_id = get_random_user_id()
        self.session_id = str(uuid.uuid4())
        self.last_question = ""
        self.last_answer = ""

        self.last_session_id = "" # 保存上次问答后端返回的真实 session_id
        self.headers = {"Content-Type": "application/json"}
        # 如果开启了认证
        if API_KEY and API_KEY != "your-api-key-here":
            self.headers["X-API-Key"] = API_KEY

    @tag("core", "non-stream")
    @task(10)
    def ask_question_non_stream(self):
        """

        非流式问答(核心接口)
        权重最高(10),模拟用户发起医学提问
        """
        question = get_random_question("medical")
        self.last_question = question
        metrics.total_requests += 1

        payload = {
            "question": question,

            "username": self.user_id, }

        with self.client.post(
            "/",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["non_stream"],
            name="POST / (非流式问答)",
            catch_response=True,

        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.last_answer = data.get("response", "")
                    # 保存后端返回的真实   session_id,供后续反馈接口引用
                    # (反馈依赖 feedback_interaction 表中存在该 session,
                    #  用自己生成的随机 session_id 会导致点赞/点踩 404/失败)
                    self.last_session_id = data.get("session_id", "")

                    # 验证响应格式
                    if not self.last_answer:
                        response.failure("响应为空")
                        metrics.error_count += 1
                    else:
                        response.success()

                except json.JSONDecodeError:
                    response.failure("JSON 解析失败")
                    metrics.error_count += 1
            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("core", "stream")

    @task(8)
    def ask_question_stream(self):
        """
        流式问答(SSE)
        记录 TTFT (Time To First Token)
        """
        question = get_random_question("medical")
        metrics.total_requests += 1

        payload = {

            "question": question,
            "username": self.user_id,
            "session_id": self.session_id,
            "think_type": "normal", }

        start_time = time.time()
        ttft_recorded = False


        with self.client.post(
            "/chat_stream",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["stream"],
            stream=True,
            name="POST /chat_stream (流式问答)",
            catch_response=True, ) as response:
            if response.status_code == 200:

                content_received = False
                try:
                    for chunk in response.iter_content(chunk_size=128):
                        if chunk and not ttft_recorded:
                            # 记录首Token延迟
                            ttft_ms = (time.time() - start_time) * 1000
                            metrics.ttft_samples.append(ttft_ms)
                            ttft_recorded = True
                        if chunk:

                            content_received = True
                    if content_received:
                        response.success()
                    else:
                        response.failure("流式响应为空")
                        metrics.error_count += 1

                except Exception as e:
                    response.failure(f"流式读取异常: {str(e)[:100]}")
                    metrics.error_count += 1
            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("feedback")

    @task(3)
    def submit_feedback_thumbs_up(self):
        """
        提交正面反馈(模拟用户点赞)。

        API 契约(feedback_routes.py):
            - feedback 字段必须为 "thumbs_up"(枚举值,非自由文本)
            - 点赞依赖后端返回的真实 session_id(该 session 已在 interaction 表中)
        """
        # 必须先有一次成功问答拿到真实 session_id,否则无法点赞

        if not self.last_session_id:
            return

        metrics.total_requests += 1
        payload = {
            "session_id": self.last_session_id,
            "feedback": "thumbs_up", # 必须是枚举值
            "user_id": self.user_id, }


        with self.client.post(
            "/feedback",
            json=payload,
            headers=self.headers,
            timeout=TIMEOUTS["feedback"],
            name="POST /feedback (点赞)",
            catch_response=True, ) as response:
            # 200=成功;404=session恰好未落库(异步写入延迟),属可接受,不计错误

            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("feedback")
    @task(1)
    def submit_feedback_thumbs_down(self):

        """
        提交负面反馈(模拟用户踩)。

        API 契约(feedback_routes.py):
            - feedback 字段必须为 "thumbs_down"(枚举值)
            - feedbackType 必须为合法枚举: harmful/false_info/not_helpful/other

            - 依赖后端返回的真实   session_id
        """
        if not self.last_session_id:
            return

        metrics.total_requests += 1
        payload = {
            "session_id": self.last_session_id,

            "feedback": "thumbs_down", # 必须是枚举值
            "user_id": self.user_id,
            "feedbackType": "not_helpful", # 必须是合法枚举, 不能是 "thumbs_down"
            "content": self.last_question,
            "answer": self.last_answer, }

        with self.client.post(
            "/feedback",
            json=payload,

            headers=self.headers,
            timeout=TIMEOUTS["feedback"],
            name="POST /feedback (踩)",
            catch_response=True, ) as response:
            # 200=成功;404=session未落库(异步写入延迟),属可接受
            if response.status_code in (200, 404):
                response.success()
            else:

                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("monitor")
    @task(1)
    def check_health(self):
        """健康检查"""
        metrics.total_requests += 1
        with self.client.get(
            "/health",

            timeout=TIMEOUTS["health"],
            name="GET /health (健康检查)",
            catch_response=True, ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ("healthy", "degraded"):
                    response.success()
                else:
                    response.failure(f"不健康状态: {data.get('status')}")

            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1


class CacheTestUser(HttpUser):
    """
    缓存验证用户(权重 25%)
    高频发送相同问题,验证 Redis 缓存命中率和延迟
    """
    weight = 25
    # 缓存用户请求更频繁(模拟热点问题)
    wait_time = between(1, 3)


    def on_start(self):
        """初始化"""
        self.user_id = get_random_user_id()
        self.headers = {"Content-Type": "application/json"}
        if API_KEY and API_KEY != "your-api-key-here":
            self.headers["X-API-Key"] = API_KEY

    @tag("cache")
    @task(1)

    def ask_cached_question(self):
        """
        发送高频重复问题,验证缓存命中
        期望第二次起命中 Redis 缓存,响应时间 < 500ms
        """
        # 固定从缓存测试问题池中选取
        question = get_random_question("cache")
        metrics.total_requests += 1


        payload = {
            "question": question,
            "username": self.user_id, }

        start_time = time.time()
        with self.client.post(
            "/",
            json=payload,
            headers=self.headers,

            timeout=TIMEOUTS["non_stream"],
            name="POST / (缓存验证)",
            catch_response=True, ) as response:
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # 判断是否命中缓存(通过响应时间判断)
                if elapsed_ms < 500:

                    metrics.cache_hits += 1
                else:
                    metrics.cache_misses += 1
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

                metrics.error_count += 1


class ExpertUser(HttpUser):
    """
    专家用户(权重 15%)
    模拟医生审核负面反馈并提交专家答案
    """

    weight = 15
    wait_time = between(5, 15)

    def on_start(self):
        """初始化"""
        self.expert_id = get_random_expert_id()
        self.headers = {"Content-Type": "application/json"}
        if API_KEY and API_KEY != "your-api-key-here":
            self.headers["X-API-Key"] = API_KEY


    @tag("expert")
    @task(3)
    def check_expert_queue(self):
        """查看待审核队列"""
        metrics.total_requests += 1
        with self.client.get(
            "/expert/queue?status=pending&limit=10",
            headers=self.headers,
            timeout=TIMEOUTS["feedback"],

            name="GET /expert/queue (审核队列)",
            catch_response=True, ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("expert")

    @task(1)
    def submit_expert_answer(self):
        """提交专家审核结果"""
        metrics.total_requests += 1
        payload = {
            "task_id": str(uuid.uuid4()),
            "expert_id": self.expert_id,
            "expert_answer": "根据临床指南,建议患者先进行血压监测,记录一周数据后再决定是否用药。",

            "notes": "压测模拟数据", }

        with self.client.post(
            "/expert/submit",
            json=payload,

            headers=self.headers,
            timeout=TIMEOUTS["feedback"],
            name="POST /expert/submit (专家审核)",
            catch_response=True, ) as response:
            # 专家审核可能因为 task_id 不存在而返回 404,属于预期行为
            if response.status_code in (200, 404):
                response.success()

            else:
                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1

    @tag("monitor")
    @task(1)
    def check_feedback_stats(self):
        """查看反馈统计"""
        metrics.total_requests += 1
        with self.client.get(

            "/feedback/stats",
            headers=self.headers,
            timeout=TIMEOUTS["feedback"],
            name="GET /feedback/stats (反馈统计)",
            catch_response=True, ) as response:
            if response.status_code == 200:
                response.success()
            else:

                response.failure(f"状态码: {response.status_code}")
                metrics.error_count += 1
