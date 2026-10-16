"""
redis_db 并发安全专项测试
==============================================
不启动后端、不调用 LLM, 直接对 redis_db.RedisClientWrapper 做多线程并发验证:

  1. 分布式锁互斥性   — 同一锁名并发 acquire, 临界区同时只能有一个持有者
  2. 缓存击穿防护     — 同一问题并发 get_or_compute, 正常答案下 compute_func 只执行一次
  3. 缓存穿透占位     — 空结果并发回源次数 (观测项: 暴露 <EMPTY> 未参与双重检查的重复回源风险)
  4. 并发读写一致性   — 多线程反复 set/get, 无异常且值读回正确
  5. 连接池并发压力   — 并发数超过连接池上限 (REDIS_MAX_CONNECTIONS) 时不报连接错误

使用方式 (在项目根目录 Agent_online 下执行):
    python stress_test/test_redis_concurrency.py
    python stress_test/test_redis_concurrency.py -n 50    # 自定义并发线程数
"""

import argparse
import os
import sys
import threading
import time
import uuid

# 保证脚本在任意目录下执行都能 import 到项目根的 redis_db
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from redis_db import redis_manager


# ==================== 终端颜色 ====================
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ==================== 测试 1: 分布式锁互斥性 ====================
def test_lock_mutual_exclusion(n: int, rounds: int = 3):
    """
    并发抢占同一把锁, 在临界区内统计同时持有者数量.
    若锁实现正确, 临界区最大并发数必须恒为 1.
    """
    lock_name = f"test-mutex-{uuid.uuid4()}"
    barrier = threading.Barrier(n)
    measure = threading.Lock()

    in_critical = 0          # 当前处于临界区的线程数
    max_in_critical = 0      # 观测到的临界区最大并发数
    acquired = 0             # 成功获锁次数
    failed = 0               # 获锁失败次数

    def worker():
        nonlocal in_critical, max_in_critical, acquired, failed
        barrier.wait()
        for _ in range(rounds):
            token = redis_manager.acquire_lock(lock_name, acquire_timeout=5.0, lock_timeout=5)
            if not token:
                with measure:
                    failed += 1
                continue
            with measure:
                acquired += 1
                in_critical += 1
                if in_critical > max_in_critical:
                    max_in_critical = in_critical
            time.sleep(0.005)   # 模拟临界区内的业务耗时
            with measure:
                in_critical -= 1
            redis_manager.release_lock(lock_name, token)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    passed = (max_in_critical == 1 and failed == 0)
    msg = (f"临界区最大并发={max_in_critical} (期望 1) | "
           f"成功获锁={acquired} | 获锁失败={failed} | 耗时={elapsed:.2f}s")
    return passed, msg


# ==================== 测试 2: 缓存击穿防护 ====================
def test_cache_breakdown(n: int):
    """
    并发打同一个"全新问题", 正常答案场景下应只回源一次 (compute_func 执行 1 次),
    其余请求等锁后命中缓存. 这是分布式锁 + 双重检查的核心价值.
    """
    question = f"并发击穿测试-{uuid.uuid4()}"
    redis_manager.delete_answer(question)   # 保证首轮 miss

    barrier = threading.Barrier(n)
    count_lock = threading.Lock()
    compute_count = 0
    results = []
    expected = "标准答案-42"

    def compute():
        nonlocal compute_count
        with count_lock:
            compute_count += 1
        time.sleep(0.2)   # 模拟 LLM 生成耗时
        return expected

    def worker():
        barrier.wait()
        results.append(redis_manager.get_or_compute(question, compute))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    cached_hit = sum(1 for r in results if r == expected)
    degraded = sum(1 for r in results if r == "System busy, calculating...")
    passed = (compute_count == 1)
    msg = (f"compute_func 执行次数={compute_count} (期望 1, 即无击穿) | "
           f"返回正确结果={cached_hit} | 降级返回={degraded} | 耗时={elapsed:.2f}s")

    redis_manager.delete_answer(question)
    return passed, msg


# ==================== 测试 3: 缓存穿透占位 ====================
def test_cache_penetration(n: int):
    """
    空结果场景: 期望写 <EMPTY> 占位后不再重复回源 (compute_func 仍只执行 1 次).
    若执行次数 > 1, 说明 <EMPTY> 被 get_answer() 归一成 None, 双重检查无法区分
    "未缓存" 与 "已缓存空值", 导致并发下反复回源 (重复调用 LLM / 重复花钱).
    """
    question = f"并发穿透测试-{uuid.uuid4()}"
    redis_manager.delete_answer(question)

    barrier = threading.Barrier(n)
    count_lock = threading.Lock()
    compute_count = 0

    def compute():
        nonlocal compute_count
        with count_lock:
            compute_count += 1
        time.sleep(0.05)
        return ""   # 空结果 → 应写 <EMPTY> 占位防穿透

    def worker():
        barrier.wait()
        redis_manager.get_or_compute(question, compute)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    passed = (compute_count == 1)
    msg = f"空结果并发回源次数={compute_count} (期望 1)"
    if not passed:
        msg += " → <EMPTY> 未参与双重检查, 存在重复回源风险"

    redis_manager.delete_answer(question)
    return passed, msg


# ==================== 测试 4: 并发读写一致性 ====================
def test_concurrent_read_write(n: int, rounds: int = 5):
    """多线程并发写不同 key 并立即读回, 验证无异常且值一致."""
    test_id = uuid.uuid4().hex[:8]
    barrier = threading.Barrier(n)
    write_lock = threading.Lock()
    errors = []
    written_keys = []

    def worker(idx):
        barrier.wait()
        for r in range(rounds):
            q = f"并发读写测试-{test_id}-{idx}-{r}"
            val = f"val-{idx}-{r}"
            with write_lock:
                written_keys.append(q)
            if not redis_manager.set_answer(q, val, expire_time=120):
                errors.append(f"写入失败: {q}")
                continue
            got = redis_manager.get_answer(q)
            if got != val:
                errors.append(f"读回不一致: {q} 期望={val} 实际={got}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    if written_keys:
        redis_manager.client.delete(*written_keys)   # 清理测试数据

    passed = (len(errors) == 0)
    sample = ("; ".join(errors[:3]) if errors else "无")
    msg = (f"总操作={n * rounds} 次 (写+读) | 异常={len(errors)} | "
           f"示例: {sample} | 耗时={elapsed:.2f}s")
    return passed, msg


# ==================== 测试 5: 连接池并发压力 ====================
def test_connection_pool(n: int):
    """并发数超过连接池上限, 验证不会抛 PoolError / TimeoutError."""
    key = f"pool-test-{uuid.uuid4()}"
    redis_manager.set_answer(key, "ok", expire_time=60)

    barrier = threading.Barrier(n)
    counter_lock = threading.Lock()
    errors = []
    ok = 0

    def worker():
        nonlocal ok
        barrier.wait()
        try:
            for _ in range(3):
                if redis_manager.get_answer(key) == "ok":
                    with counter_lock:
                        ok += 1
        except Exception as e:
            with counter_lock:
                errors.append(f"{type(e).__name__}: {str(e)[:60]}")

    threads = [threading.Thread(target=worker) for _ in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    redis_manager.delete_answer(key)

    passed = (len(errors) == 0)
    sample = ("; ".join(errors[:3]) if errors else "无")
    msg = (f"并发={n} (连接池上限={os.getenv('REDIS_MAX_CONNECTIONS', '100')}) | "
           f"成功读取={ok} | 异常={len(errors)} | 示例: {sample} | 耗时={elapsed:.2f}s")
    return passed, msg


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="redis_db 并发安全专项测试")
    parser.add_argument("-n", "--threads", type=int, default=30, help="并发线程数 (默认 30)")
    args = parser.parse_args()

    n = args.threads

    print("=" * 70)
    print(f"{C.BOLD} redis_db 并发安全专项测试{C.END}   并发线程数: {n}")
    print("=" * 70)

    # 环境自检
    if not redis_manager.is_healthy():
        print(f"{C.RED}[环境错误] Redis 不可用, 请先启动 Redis (默认 127.0.0.1:6379) 并检查 .env{C.END}")
        sys.exit(1)
    print(f"{C.BLUE}[环境] Redis 连接正常{C.END}\n")

    metrics_before = redis_manager.get_metrics()

    cases = [
        ("1. 分布式锁互斥性", test_lock_mutual_exclusion, (n,)),
        ("2. 缓存击穿防护",   test_cache_breakdown,        (n,)),
        ("3. 缓存穿透占位",   test_cache_penetration,      (n,)),
        ("4. 并发读写一致性", test_concurrent_read_write,  (n,)),
        ("5. 连接池并发压力", test_connection_pool,        (n,)),
    ]

    results = []
    for name, func, fargs in cases:
        t0 = time.time()
        try:
            passed, msg = func(*fargs)
        except Exception as e:
            passed, msg = False, f"测试抛异常: {type(e).__name__}: {e}"
        elapsed = time.time() - t0
        results.append((name, passed, msg, elapsed))

        tag = f"{C.GREEN}PASS{C.END}" if passed else f"{C.RED}FAIL{C.END}"
        print(f"[{tag}] {name}  ({elapsed:.2f}s)")
        print(f"        {msg}\n")

    # 汇总
    total = len(results)
    passed_count = sum(1 for _, p, _, _ in results if p)
    print("=" * 70)
    print(f"{C.BOLD} 汇总: {passed_count}/{total} 通过{C.END}")

    metrics_after = redis_manager.get_metrics()
    print(f" ── redis_manager 指标增量 ──")
    for field in ("hit_count", "miss_count", "write_count", "error_count",
                  "lock_acquired", "lock_failed"):
        delta = metrics_after[field] - metrics_before.get(field, 0)
        print(f" {field:<15}: +{delta}")

    info = redis_manager.get_redis_info()
    if info:
        print(f" ── Redis 服务端 ──")
        print(f" 版本: {info.get('redis_version')} | 内存: {info.get('used_memory_human')} | "
              f"当前连接: {info.get('connected_clients')} | "
              f"累计命令: {info.get('total_commands_processed')}")
    print("=" * 70)

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
