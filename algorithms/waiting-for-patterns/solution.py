"""
等待模式问题 — 抛硬币与收集问题
================================

问题 A: 抛硬币，第一次连续抛出 N 个正面，平均需要抛多少次？
问题 B: 投 N 面骰，第一次出现所有面各至少一次，期望次数是多少？

纯 Python 实现，含递推推导、蒙特卡洛验证、与 Coupon Collector 的对比。
"""

import math
import random


# =============================================================================
# 问题 A: 连续 N 次正面的期望等待时间
# =============================================================================

def expected_consecutive_heads_exact(N):
    """
    精确解: E = 2^(N+1) - 2

    推导: 设 E[i] = 已连抛 i 个正面，还需多少次达到 N 个
      E[N] = 0
      E[i] = 1 + 0.5*E[i+1] + 0.5*E[0]   (i = 0,1,...,N-1)

    令 delta_i = E[i] - E[i+1]，得 delta_i = 0.5 * delta_{i+1}
    递推得 delta_i = 2^{i+1}，求和得 E[i] = 2^{N+1} - 2^{i+1}
    因此 E[0] = 2^{N+1} - 2
    """
    return 2 ** (N + 1) - 2


def expected_consecutive_heads_dp(N):
    """
    动态规划验证递推公式。

    递推: E[i] = 1 + 0.5*E[i+1] + 0.5*E[0]
    令 E[i] = a_i * E[0] + b_i，分别递推 a, b，再解 E[0] = b_0 / (1-a_0)
    """
    a = [0.0] * (N + 1)
    b = [0.0] * (N + 1)
    a[N] = 0.0
    b[N] = 0.0
    for i in range(N - 1, -1, -1):
        a[i] = 0.5 * a[i + 1] + 0.5
        b[i] = 1 + 0.5 * b[i + 1]
    return b[0] / (1.0 - a[0])


def simulate_consecutive_heads(N, trials=100000, seed=None):
    """Monte Carlo 模拟连续 N 次正面所需的抛掷次数。"""
    if seed is not None:
        random.seed(seed)
    total = 0
    for _ in range(trials):
        count = 0
        consecutive = 0
        while consecutive < N:
            count += 1
            if random.random() < 0.5:
                consecutive += 1
            else:
                consecutive = 0
        total += count
    return total / trials


# =============================================================================
# 问题 B: N 面骰收集所有面 (Coupon Collector)
# =============================================================================

def expected_coupon_collector(N):
    """
    Coupon Collector 问题。
    已收集 k 个不同面时，下一次获得新面的概率 = (N-k)/N
    期望等待时间 = N/(N-k)
    总期望 = N * (1 + 1/2 + ... + 1/N) = N * H_N
    """
    return N * sum(1.0 / i for i in range(1, N + 1))


def simulate_coupon_collector(N, trials=100000, seed=None):
    """Monte Carlo 模拟收集所有 N 个面所需的投掷次数。"""
    if seed is not None:
        random.seed(seed)
    total = 0
    for _ in range(trials):
        seen = set()
        count = 0
        while len(seen) < N:
            seen.add(random.randint(0, N - 1))
            count += 1
        total += count
    return total / trials


# =============================================================================
# 通用 Pattern 等待时间 (Martingale / Conway Leading Number)
# =============================================================================

def expected_pattern(pattern, p_head=0.5):
    """
    使用 Martingale / Conway Leading Number 方法计算等待任意模式的期望。

    公式: E = sum_{k=1}^{n} (1/p)^k * I(pattern[0:k] == pattern[n-k:n])

    例如 pattern="HHH":
      k=1: "H"=="H", 贡献 2^1 = 2
      k=2: "HH"=="HH", 贡献 2^2 = 4
      k=3: "HHH"=="HHH", 贡献 2^3 = 8
      E = 2 + 4 + 8 = 14 = 2^4 - 2  ✓

    例如 pattern="HTH":
      k=1: "H"=="H", 贡献 2
      k=2: "HT"!="TH", 不贡献
      k=3: "HTH"=="HTH", 贡献 8
      E = 2 + 8 = 10
    """
    n = len(pattern)
    E = 0.0
    for k in range(1, n + 1):
        prefix = pattern[:k]
        suffix = pattern[n - k:]
        if prefix == suffix:
            E += (1.0 / p_head) ** k
    return E


def simulate_pattern(pattern, trials=50000, p_head=0.5, seed=None):
    """模拟等待特定模式出现的期望次数。"""
    if seed is not None:
        random.seed(seed)
    total = 0
    for _ in range(trials):
        seq = ""
        count = 0
        while not seq.endswith(pattern):
            count += 1
            seq += "H" if random.random() < p_head else "T"
        total += count
    return total / trials


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("🪙 等待模式问题 — 抛硬币 & 收集问题")
    print("=" * 70)
    print()

    # ========== 问题 A ==========
    print("【问题 A】连续 N 次正面的期望抛掷次数")
    print("-" * 50)
    print(f"{'N':>4} | {'公式解 2^(N+1)-2':>18} | {'DP 递推':>10} | {'Monte Carlo':>12}")
    print("-" * 50)
    for N in [1, 2, 3, 4, 5, 6, 8, 10]:
        exact = expected_consecutive_heads_exact(N)
        dp = expected_consecutive_heads_dp(N)
        sim = simulate_consecutive_heads(N, 30000, seed=42)
        print(f"{N:>4} | {exact:>18.2f} | {dp:>10.2f} | {sim:>12.2f}")
    print()

    # 递推状态展示
    N = 5
    # E[i] = a_i * E[0] + b_i
    a = [0.0] * (N + 1)
    b = [0.0] * (N + 1)
    a[N] = 0.0
    b[N] = 0.0
    for i in range(N - 1, -1, -1):
        a[i] = 0.5 * a[i + 1] + 0.5
        b[i] = 1 + 0.5 * b[i + 1]
    E0 = b[0] / (1.0 - a[0])
    print(f"递推状态验证 (N={N}):")
    for i in range(N + 1):
        Ei = a[i] * E0 + b[i]
        formula = 2 ** (N + 1) - 2 ** (i + 1)
        print(f"  E[{i}] = 还需 {Ei:>6.2f} 次  (通项公式: {formula})")
    print()

    # ========== 通用 Pattern ==========
    print("【扩展】任意模式的期望等待时间 (Conway Leading Number)")
    print("-" * 55)
    patterns = ["H", "HH", "HHH", "HTH", "HTHT", "HTHH"]
    print(f"{'模式':>8} | {'公式期望':>10} | {'模拟值':>10}")
    print("-" * 35)
    for pat in patterns:
        e = expected_pattern(pat)
        sim = simulate_pattern(pat, 20000, seed=42)
        print(f"{pat:>8} | {e:>10.2f} | {sim:>10.2f}")
    print()

    # ========== 问题 B ==========
    print("【问题 B】N 面骰收集所有面的期望次数 (Coupon Collector)")
    print("-" * 55)
    print(f"{'N':>4} | {'精确值 N*H_N':>14} | {'Monte Carlo':>12} | {'近似 N*ln(N)':>14}")
    print("-" * 50)
    for N in [2, 3, 4, 6, 10, 20, 50, 100]:
        exact = expected_coupon_collector(N)
        sim = simulate_coupon_collector(N, 30000, seed=42)
        approx = N * math.log(N) + 0.5772 * N  # Euler-Mascheroni
        print(f"{N:>4} | {exact:>14.4f} | {sim:>12.4f} | {approx:>14.4f}")
    print()

    # ========== 两问题对比 ==========
    print("【对比】连续正面 vs 收集所有面")
    print("-" * 50)
    print(f"{'N':>4} | {'连续N正面 E':>14} | {'收集N面 E':>14} | {'比值':>8}")
    print("-" * 45)
    for N in [2, 3, 4, 5, 6, 8, 10]:
        e_heads = expected_consecutive_heads_exact(N)
        e_collect = expected_coupon_collector(N)
        ratio = e_heads / e_collect
        print(f"{N:>4} | {e_heads:>14.2f} | {e_collect:>14.2f} | {ratio:>8.2f}")
    print()

    print("关键观察:")
    print("  • 连续正面的期望呈指数增长: E ~ 2^N")
    print("  • Coupon Collector 呈线性对数增长: E ~ N*ln(N)")
    print("  • 当 N=6 时，连续正面需要 126 次，收集骰子只需 14.7 次")
    print("  • 当 N=10 时，连续正面需要 2046 次，收集骰子只需 29.3 次")
    print()
    print("原因: 连续正面要求严格的顺序约束，失败一次全部重来;")
    print("      收集问题只需覆盖集合，顺序无关，可以渐进积累。")


if __name__ == "__main__":
    demo()
