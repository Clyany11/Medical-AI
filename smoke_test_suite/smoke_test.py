"""
Agent_online 冒烟测试 - 核心测试脚本
==========================================
按 P0/P1/P2 优先级逐个验证接口可用性, 输出彩色报告

使用方式:
    python smoke_test_suite/smoke_test.py

"""

import argparse
import io
import json
import sys
import time
import uuid
import wave
from datetime import datetime


import requests

sys.path.insert(0, ".")
from smoke_test_suite.config import (
    TARGET_HOST, API_KEY, TIMEOUTS,
    SMOKE_QUESTION, SMOKE_TCM_QUESTION, SMOKE_GREETING, SMOKE_TTS_TEXT,
    SMOKE_USER, SMOKE_EXPERT, SMOKE_CRITERIA, REPORT_DIR, )


# ==================== 终端颜色 ====================
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"

    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ==================== 测试结果收集器 ====================
class SmokeResult:
    """冒烟测试结果收集器"""


    def __init__(self):
        self.results = [] # [(level, name, passed, elapsed_ms, message)]

    def add(self, level, name, passed, elapsed_ms, message=""):
        self.results.append({
            "level": level,
            "name": name,
            "passed": passed,
            "elapsed_ms": elapsed_ms,

            "message": message, })

    def summary(self):
        """汇总统计"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        # P0 是否全部通过 (决定是否可上线)
        p0_results = [r for r in self.results if r["level"] == "P0"]

        p0_all_pass = all(r["passed"] for r in p0_results) if p0_results else False
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "p0_all_pass": p0_all_pass, }



# ==================== HTTP 辅助 ====================
def build_headers(json_content=True):
    """构建请求头"""
    headers = {}
    if json_content:
        headers["Content-Type"] = "application/json"

    if API_KEY and API_KEY != "XXXXXXXXXX":
        headers["X-API-Key"] = API_KEY
    return headers


def generate_test_wav(duration=2.0, sample_rate=16000):
    """生成测试用  WAV 音频 (静音)"""
    num_samples = int(sample_rate * duration)

    audio_data = b'\x00\x00' * num_samples
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return buffer.getvalue()



# ==================== 测试用例 ====================
class SmokeTests:
    """冒烟测试用例集合"""

    def __init__(self, host, result: SmokeResult):
        self.host = host.rstrip("/")
        self.result = result
        self.session_id = str(uuid.uuid4())
        self.last_answer = ""


    def _run(self, level, name, func):
        """
        执行单个测试用例, 捕获异常并记录耗时
        """
        start = time.time()
        try:
            passed, message = func()
        except requests.exceptions.Timeout:
            passed, message = False, "请求超时"

        except requests.exceptions.ConnectionError:
            passed, message = False, "连接失败 (服务未启动?)"
        except Exception as e:
            passed, message = False, f"异常: {str(e)[:80]}"
        elapsed_ms = (time.time() - start) * 1000
        self.result.add(level, name, passed, elapsed_ms, message)
        self._print_result(level, name, passed, elapsed_ms, message)
        return passed


    def _print_result(self, level, name, passed, elapsed_ms, message):
        """打印单条结果"""
        status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.RED}FAIL{C.END}"
        level_color = {"P0": C.RED, "P1": C.YELLOW, "P2": C.BLUE}.get(level, "")
        print(f" [{level_color}{level}{C.END}] {name:<28} {status} "
                f"({elapsed_ms:>6.0f}ms) {message}")


    # ========== P0 级别:核心功能 ==========
    def test_health(self):
        """健康检查"""
        def _test():
            resp = requests.get(
                f"{self.host}/health",
                headers=build_headers(False),

                timeout=TIMEOUTS["health"], )
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            data = resp.json()
            status = data.get("status", "unknown")
            if status not in ("healthy", "degraded"):
                return False, f"状态异常: {status}"
            # 检查依赖服务
            deps = []

            if not data.get("redis", True):
                deps.append("Redis不可用")
            if not data.get("milvus", True):
                deps.append("Milvus不可用")
            msg = f"status={status}"
            if deps:
                msg += f" ({', '.join(deps)})"
            return True, msg
        return self._run("P0", "健康检查 /health", _test)


    def test_non_stream_qa(self):
        """非流式问答"""
        def _test():
            resp = requests.post(
                f"{self.host}/",
                json={"question": SMOKE_QUESTION, "username": SMOKE_USER},
                headers=build_headers(),
                timeout=TIMEOUTS["non_stream"], )

            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            data = resp.json()
            answer = data.get("response", "")
            self.last_answer = answer
            if len(answer) < SMOKE_CRITERIA["min_answer_length"]:
                return False, f"回答过短: '{answer[:20]}'"
            return True, f"回答长度={len(answer)}字"
        return self._run("P0", "非流式问答 POST /", _test)


    def test_stream_qa(self):
        """流式问答"""
        def _test():
            resp = requests.post(
                f"{self.host}/chat_stream",
                json={
                    "question": SMOKE_QUESTION,
                    "username": SMOKE_USER,
                    "session_id": self.session_id,
                    "think_type": "normal", }, headers=build_headers(), timeout=TIMEOUTS["stream"],

                stream=True, )
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            chunk_count = 0
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=128):
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)


                # 收到几个块即可确认流正常
                if chunk_count >= 3:
                    break

            if chunk_count == 0:
                return False, "无流式数据返回"
            return True, f"收到{chunk_count}个数据块 ({total_bytes}字节)"
        return self._run("P0", "流式问答 /chat_stream", _test)


    # ========== P1 级别: 重要功能 ==========
    def test_metrics(self):
        """系统指标"""
        def _test():
            resp = requests.get(
                f"{self.host}/metrics",
                headers=build_headers(False),
                timeout=TIMEOUTS["health"], )

            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            data = resp.json()
            return True, f"指标字段数={len(data)}"
        return self._run("P1", "系统指标 /metrics", _test)

    def test_feedback(self):
        """
        用户反馈

        API 契约 (feedback_routes.py FeedbackRequest):
            - feedback 字段必须为 "thumbs_up" 或 "thumbs_down" (枚举值, 非自由文本)
            - 点赞依赖 session_id 在 feedback_interaction 表中存在
        冒烟测试用随机 session_id, 未经真实问答, 点赞会返回 404 (属正常校验),
        因此接受  200/404 均视为接口契约正确 (P1 只验证接口可用性)
        """
        def _test():
            resp = requests.post(
                f"{self.host}/feedback",
                json={
                    "session_id": self.session_id,
                    "feedback": "thumbs_up", # 必须是枚举值, 不能是自由文本
                    "user_id": SMOKE_USER,

                },
                headers=build_headers(),
                timeout=TIMEOUTS["feedback"], )
            # 200=成功, 404=session不存在 (随机session未经问答, 属正常)
            if resp.status_code not in (200, 404):
                return False, f"HTTP {resp.status_code}"
            return True, f"反馈接口正常 (HTTP {resp.status_code})"
        return self._run("P1", "用户反馈 /feedback", _test)


    def test_feedback_stats(self):
        """反馈统计"""
        def _test():
            resp = requests.get(
                f"{self.host}/feedback/stats",
                headers=build_headers(False),
                timeout=TIMEOUTS["feedback"], )
            if resp.status_code != 200:

                return False, f"HTTP {resp.status_code}"
            return True, "统计接口正常"
        return self._run("P1", "反馈统计 /feedback/stats", _test)

    def test_expert_queue(self):
        """专家队列"""
        def _test():
            resp = requests.get(
                f"{self.host}/expert/queue?status=pending&limit=5",
                headers=build_headers(False),

                timeout=TIMEOUTS["feedback"], )
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            data = resp.json()
            tasks = data.get("tasks", [])
            return True, f"待审核任务数={len(tasks)}"
        return self._run("P1", "专家队列 /expert/queue", _test)


    # ========== P2 级别: 增强功能 (语音) ==========
    def test_tts(self):
        """语音合成"""
        def _test():
            resp = requests.post(
                f"{self.host}/voice/tts_only",
                json={"text": SMOKE_TTS_TEXT},
                headers=build_headers(),
                timeout=TIMEOUTS["voice"],
                stream=True, )
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            got_data = False

            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    got_data = True
                    break
            if not got_data:
                return False, "无音频数据返回"
            return True, "TTS 音频流正常"
        return self._run("P2", "语音合成 /voice/tts_only", _test)

    def test_asr(self):

        """语音识别"""
        def _test():
            wav_data = generate_test_wav(2.0)
            headers = build_headers(False) # multipart 不设 Content-Type
            resp = requests.post(
                f"{self.host}/voice/asr",
                files={"file": ("smoke.wav", wav_data, "audio/wav")},
                headers=headers,
                timeout=TIMEOUTS["voice"],

                stream=True, )
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            got_data = False
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    got_data = True
                    break
            # ASR 接口可达即算通过 (静音音频可能无识别结果)

            msg = "ASR 接口响应正常" if got_data else "ASR 接口可达(静音无识别结果)"
            return True, msg
        return self._run("P2", "语音识别 /voice/asr", _test)


# ==================== 主流程 ====================
#  AI课堂小朱老师独家讲义
def run_smoke_tests(host, level_filter=None):
    """
    执行冒烟测试

    Args:
        host: 目标服务地址
        level_filter: 只跑指定级别 (P0/P1/P2), None 表示全部
    """
    result = SmokeResult()

    tests = SmokeTests(host, result)

    print(f"\n{C.BOLD}{'='*68}{C.END}")
    print(f"{C.BOLD} Agent_online_v3 冒烟测试{C.END}")
    print(f"{C.BOLD}{'='*68}{C.END}")
    print(f" 目标: {host}")
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 级别: {level_filter or 'P0+P1+P2 全部'}")

    print(f"{'='*68}\n")

    # P0 核心功能
    if not level_filter or level_filter == "P0":
        print(f"{C.BOLD} ── P0 核心功能 (失败即阻断上线)──{C.END}")
        tests.test_health()
        tests.test_non_stream_qa()
        tests.test_stream_qa()

    # P1 重要功能

    if not level_filter or level_filter == "P1":
        print(f"\n{C.BOLD} ── P1 重要功能 ──{C.END}")
        tests.test_metrics()
        tests.test_feedback()
        tests.test_feedback_stats()
        tests.test_expert_queue()

    # P2 增强功能
    if not level_filter or level_filter == "P2":

        print(f"\n{C.BOLD} ── P2 增强功能 (语音) ──{C.END}")
        tests.test_tts()
        tests.test_asr()

    # 汇总
    s = result.summary()
    print(f"\n{C.BOLD}{'='*68}{C.END}")
    print(f"{C.BOLD} 冒烟测试汇总{C.END}")
    print(f"{'='*68}")
    print(f" 总用例: {s['total']} | "

            f"{C.GREEN}通过: {s['passed']}{C.END} | "
            f"{C.RED}失败: {s['failed']}{C.END}")

    if s["p0_all_pass"]:
        print(f"\n {C.GREEN}{C.BOLD}✓ P0 全部通过 - 核心功能正常, 可继续后续测试{C.END}")
    else:
        print(f"\n {C.RED}{C.BOLD}✗ P0 存在失败 - 核心功能异常, 禁止上线!{C.END}")
    print(f"{'='*68}\n")


    # 保存报告
    save_report(result, host)

    # P0 失败返回非零退出码 (用于 CI 阻断)
    return 0 if s["p0_all_pass"] else 1


def save_report(result: SmokeResult, host):
    """保存 JSON 报告"""
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(
        REPORT_DIR,
        f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    )
    report = {
        "test_type": "smoke_test",
        "host": host,
        "timestamp": datetime.now().isoformat(),
        "summary": result.summary(),
        "results": result.results, }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f" 报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Agent_online_v3 冒烟测试")
    parser.add_argument("--host", default=TARGET_HOST, help="目标服务地址")
    parser.add_argument("--level", choices=["P0", "P1", "P2"], help="只跑指定级别")
    args = parser.parse_args()


    exit_code = run_smoke_tests(args.host, args.level)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
