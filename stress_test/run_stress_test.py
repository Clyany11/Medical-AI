"""
Agent_online 压力测试 - 一键执行脚本

==========================================
提供命令行入口, 支持选择不同压测场景一键执行.
自动创建报告目录, 执行压测, 生成汇总报告

使用方式:
    # 交互式选择场景
    python stress_test/run_stress_test.py

    # 指定场景运行
    python stress_test/run_stress_test.py --scenario basic
    python stress_test/run_stress_test.py --scenario peak
    python stress_test/run_stress_test.py --scenario soak
    python stress_test/run_stress_test.py --scenario voice
    python stress_test/run_stress_test.py --scenario all

=====================================================================
[缺口标注] 本文件从课堂教材 Agent开发9.pdf 逐行重建。
voice 场景引用的 stress_test/locustfile_voice.py 在 PDF 中未给出源码,
当前目录下也不存在该文件 → --scenario voice / all 会因找不到
locust -f 目标文件而失败, 需补齐 locustfile_voice.py 后才能使用。
=====================================================================
"""


import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


# ==================== 场景配置 ====================
SCENARIOS = {

    "basic": {
        "name": "基础压测",
        "description": "50并发, 5分钟, 覆盖核心接口",
        "file": "stress_test/locustfile_basic.py",
        "users": 50,
        "spawn_rate": 5,
        "duration": "5m",
        "csv_prefix": "basic", },

    "peak": {
        "name": "极限压测",
        "description": "200并发, 5分钟, 测试系统极限",
        "file": "stress_test/locustfile_peak.py",
        "users": 200,
        "spawn_rate": 10,
        "duration": "5m",
        "csv_prefix": "peak", }, "soak": {

        "name": "稳定性测试",
        "description": "30并发, 60分钟, 检测内存泄漏/性能退化",
        "file": "stress_test/locustfile_peak.py",
        "users": 30,
        "spawn_rate": 2,
        "duration": "60m",
        "csv_prefix": "soak",
        "extra_args": ["--tags", "soak"], },

    "voice": {
        "name": "语音接口压测",
        "description": "20并发, 5分钟, ASR/TTS/全链路",
        "file": "stress_test/locustfile_voice.py",
        "users": 20,
        "spawn_rate": 2,
        "duration": "5m",
        "csv_prefix": "voice", }, }


def print_banner():
    """打印横幅"""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Agent_online 压力测试执行器                     ║")
    print("║  Multi-Agent RAG 医学问答系统                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_scenarios():
    """打印可用场景列表"""

    print(" 可用压测场景:")
    print(" " + "-" * 50)
    for key, cfg in SCENARIOS.items():
        print(f" {key:>8} | {cfg['name']:8} | {cfg['description']}")
    print(" " + "-" * 50)
    print(f" {'all':>8} | {'全部执行':8} | 按顺序执行: basic → step → peak →voice")
    print()


def ensure_report_dir():
    """确保报告目录存在"""
    report_dir = "stress_test/reports"
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def check_service(host: str) -> bool:
    """

    检查目标服务是否可用
    Args:
        host: 目标服务地址
    Returns:
        是否可用
    """
    import urllib.request
    try:
        url = f"{host}/health"

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print(f" [OK] 目标服务可达: {host}")
                return True
    except Exception as e:
        print(f" [FAIL] 目标服务不可达: {host}")
        print(f"     错误: {e}")
        return False
    return False


def run_scenario(scenario_key: str, host: str) -> bool:
    """

    执行单个压测场景
    Args:
        scenario_key: 场景名称
        host: 目标服务地址
    Returns:
        是否成功
    """
    cfg = SCENARIOS[scenario_key]
    report_dir = ensure_report_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_prefix = f"{report_dir}/{cfg['csv_prefix']}_{timestamp}"

    print(f"\n{'='*60}")
    print(f" 执行场景: {cfg['name']}")
    print(f" 描述: {cfg['description']}")
    print(f" 并发: {cfg['users']} 用户")
    print(f" 时长: {cfg['duration']}")
    print(f" 报告: {csv_prefix}_*.csv")
    print(f"{'='*60}\n")


    # 构建 locust 命令
    cmd = [
        sys.executable, "-m", "locust",
        "-f", cfg["file"],
        f"--host={host}",
        "--headless",
        f"-u", str(cfg["users"]),
        f"-r", str(cfg["spawn_rate"]),
        f"-t", cfg["duration"],

        f"--csv={csv_prefix}",
        "--csv-full-history",
        f"--html={csv_prefix}_report.html", ]

    # 添加额外参数(如 --tags)
    if "extra_args" in cfg:
        cmd.extend(cfg["extra_args"])


    print(f" 命令: {' '.join(cmd)}\n")

    try:
        # 执行压测
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=int(cfg["duration"].rstrip("m")) * 60 + 120, # 额外2分钟缓冲
        )
        success = result.returncode == 0
        if success:
            print(f"\n [OK] {cfg['name']} 完成")
        else:
            print(f"\n [WARN] {cfg['name']} 退出码: {result.returncode}")

        return success
    except subprocess.TimeoutExpired:
        print(f"\n [TIMEOUT] {cfg['name']} 超时")
        return False
    except FileNotFoundError:
        print(f"\n [ERROR] locust 未安装, 请运行: pip install -r stress_test/requirements.txt")
        return False
    except KeyboardInterrupt:
        print(f"\n [中断] 用户取消")

        return False


def run_all(host: str):
    """按顺序执行所有场景 (不含 soak)"""
    order = ["basic", "peak", "voice"]
    results = {}

    for scenario in order:

        success = run_scenario(scenario, host)
        results[scenario] = success
        if not success:
            print(f"\n [WARN] {scenario} 未通过, 继续下一个...")
        # 场景之间等待10秒, 让系统恢复
        print("\n 等待10秒让系统恢复...")
        time.sleep(10)

    # 汇总
    print("\n" + "=" * 60)

    print(" 全部场景执行结果汇总")
    print("=" * 60)
    for scenario, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f" {SCENARIOS[scenario]['name']:12} : {status}")
    print("=" * 60)


def main():
    """主入口"""

    parser = argparse.ArgumentParser(description="Agent_online 压力测试执行器")
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="压测场景名称",

    )
    parser.add_argument(
        "--host",
        default="http://0.0.0.0:8103",
        help="目标服务地址 (默认: http://0.0.0.0:8103)", )
    parser.add_argument(
        "--skip-check",

        action="store_true",
        help="跳过服务可用性检查", )

    args = parser.parse_args()

    print_banner()

    # 检查服务可用性
    if not args.skip_check:

        if not check_service(args.host):
            print("\n 请先启动服务: python app.py")
            print(" 或使用 --skip-check 跳过检查")
            sys.exit(1)

    # 交互式选择场景
    if not args.scenario:
        print_scenarios()
        scenario = input(" 请输入场景名称: ").strip().lower()

        if scenario not in list(SCENARIOS.keys()) + ["all"]:
            print(f" [ERROR] 未知场景: {scenario}")
            sys.exit(1)
    else:
        scenario = args.scenario

    # 执行
    if scenario == "all":
        run_all(args.host)
    else:

        run_scenario(scenario, args.host)


if __name__ == "__main__":
    main()
