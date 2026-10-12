"""
Agent_online 回归测试 - 一键执行器
========================================
封装 pytest 调用, 提供友好的命令行入口和结果汇总。

支持按类别运行, 生成 HTML/JSON 报告

使用方式:
    python regression_test/run_regression.py
"""

import argparse
import os
import subprocess

import sys
import urllib.request
from datetime import datetime


# ==================== 测试类别 ====================
CATEGORIES = {
    "intent": "意图路由正确性",
    "cache": "缓存行为验证",
    "memory": "多轮记忆能力",

    "format": "响应格式规范",
    "flywheel": "数据飞轮闭环",
    "safety": "安全边界处理", }


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Agent_online 回归测试执行器                     ║")

    print("║  功能正确性验证                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def check_service(host):
    """检查服务可用性"""
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=5) as resp:

            if resp.status == 200:
                print(f" [OK] 目标服务可达: {host}\n")
                return True
    except Exception as e:
        print(f" [FAIL] 目标服务不可达: {host}")
        print(f"     错误: {e}\n")
        return False
    return False


#  AI课堂小朱老师独家讲义
def run_regression(host, category=None):
    """
    执行回归测试
    Args:
        host: 目标服务地址
        category: 测试类别, None 表示全部
    """
    report_dir = "regression_test/reports"

    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置目标地址环境变量 (供 config 读取, 也可直接改 config.py)
    env = os.environ.copy()

    # 构建 pytest 命令 (基础参数, 不依赖任何可选插件)
    cmd = [
        sys.executable, "-m", "pytest",

        "regression_test/test_regression.py",
        "-v",
        "--tb=short",
        "--color=yes",
        f"--junitxml={report_dir}/regression_{timestamp}.xml", ]

    # 检测 pytest-html 是否可用, 可用则加 HTML 报告参数
    html_available = _check_pytest_html_available()
    if html_available:

        cmd.extend([
            f"--html={report_dir}/regression_{timestamp}.html",
            "--self-contained-html", ])
    else:
        print(" [INFO] pytest-html 未安装, 跳过 HTML 报告生成")
        print("     安装命令: pip install pytest-html")

    # 按类别过滤

    if category and category != "all":
        cmd.extend(["-m", category])
        print(f" 测试类别: {CATEGORIES.get(category, category)}")
    else:
        print(f" 测试类别: 全部({len(CATEGORIES)}类)")

    print(f" 报告目录: {report_dir}/regression_{timestamp}.*")
    print(f"\n 命令: {' '.join(cmd)}\n")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = subprocess.run(cmd, cwd=project_root, env=env)
        return result.returncode == 0

    except FileNotFoundError:
        print("\n [ERROR] pytest 未安装,请运行:")
        print("      pip install pytest")
        return False
    except KeyboardInterrupt:
        print("\n [中断] 用户取消")
        return False


def _check_pytest_html_available():

    """检测 pytest-html 插件是否已安装"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pytest_html"],
            capture_output=True, timeout=5, )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Agent_online 回归测试执行器")
    parser.add_argument(
        "--category", "-c",
        choices=list(CATEGORIES.keys()) + ["all"],
        default="all",
        help="测试类别(默认全部)", )

    parser.add_argument(
        "--host",
        default="http://127.0.0.1:8103",
        help="目标服务地址", )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过服务可用性检查",

    )
    args = parser.parse_args()

    print_banner()

    # 列出可用类别
    print(" 可用测试类别:")
    for key, desc in CATEGORIES.items():
        print(f" {key:<10} - {desc}")
    print()

    # 检查服务
    if not args.skip_check:
        if not check_service(args.host):

            print(" 请先启动服务: python app.py")
            print(" 或使用 --skip-check 跳过检查")
            sys.exit(1)

    # 执行
    success = run_regression(args.host, args.category)

    print("\n" + "=" * 60)
    if success:
        print(" 回归测试完成: ALL PASS")

    else:
        print(" 回归测试完成: 存在失败用例,   请查看报告")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
