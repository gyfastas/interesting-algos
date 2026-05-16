"""
用 Rand7() 构造 Rand10()
=========================

LeetCode 470: 已有方法 rand7() 可生成 1~7 的均匀随机整数，
利用它构造 rand10() 生成 1~10 的均匀随机整数。

纯 Python 实现，含拒绝采样、通用构造方法、均匀性验证。
"""

import random
import math
from collections import Counter


# =============================================================================
# 基础工具
# =============================================================================

def rand7():
    """已有的均匀随机生成器: 返回 [1, 7] 的整数。"""
    return random.randint(1, 7)


def rand2():
    """用 rand7 构造 rand2: 拒绝 3,4,5,6,7，保留 1,2。"""
    while True:
        x = rand7()
        if x <= 2:
            return x


def rand3():
    """用 rand7 构造 rand3: 拒绝 4,5,6,7，保留 1,2,3。"""
    while True:
        x = rand7()
        if x <= 3:
            return x


# =============================================================================
# 核心解法: 拒绝采样 (Rejection Sampling)
# =============================================================================

def rand10():
    """
    经典解法: 两次 rand7() 生成 1~49，拒绝 41~49。

    原理:
      1. row = rand7()-1 取 0~6，col = rand7() 取 1~7
      2. num = row * 7 + col → 均匀分布 1~49
      3. 若 num <= 40，接受；否则拒绝重试
      4. 返回 (num-1) % 10 + 1 → 1~10

    期望调用 rand7() 次数: 2 * (49/40) = 2.45 次
    """
    while True:
        row = rand7() - 1   # 0~6
        col = rand7()       # 1~7
        num = row * 7 + col  # 1~49
        if num <= 40:
            return (num - 1) % 10 + 1


def rand10_optimized():
    """
    优化版: 三次 rand7() 生成 1~343，拒绝 341~343。

    拒绝率: 3/343 ≈ 0.87% (远低于经典解法的 9/49 ≈ 18.4%)
    期望调用 rand7() 次数: 3 * (343/340) ≈ 3.03 次
    """
    while True:
        num = (rand7() - 1) * 49 + (rand7() - 1) * 7 + rand7()
        # num ∈ [1, 343]
        if num <= 340:
            return (num - 1) % 10 + 1


def rand10_from_rand2():
    """
    另类思路: 先用 rand7 构造 rand2，再用 4 个比特构造 rand10。

    效率较低，仅作演示。
    """
    while True:
        bits = 0
        for _ in range(4):  # 4 bits → 0~15
            bits = (bits << 1) | (rand2() - 1)
        if bits < 10:
            return bits + 1


# =============================================================================
# 通用构造方法: rand(a) → rand(b)
# =============================================================================

def rand_a_to_rand_b(rand_a, a, b):
    """
    通用方法: 利用 rand(a) 构造 rand(b)。

    步骤:
      1. 找最小 k 使得 a^k >= b
      2. 调用 k 次 rand(a)，构造 1~a^k 的均匀分布
      3. 拒绝大于 floor(a^k / b) * b 的数
      4. 取模映射到 1~b

    期望调用次数: k * a^k / (floor(a^k / b) * b)
    """
    if b <= 1:
        return 1

    # 找最小 k
    k = 1
    while a ** k < b:
        k += 1

    max_range = a ** k
    max_accept = (max_range // b) * b

    while True:
        num = 0
        for _ in range(k):
            num = num * a + (rand_a() - 1)
        num += 1  # 转为 1-indexed
        if num <= max_accept:
            return (num - 1) % b + 1


def theoretical_expected_calls(a, b):
    """计算通用方法的期望 rand(a) 调用次数。"""
    k = 1
    while a ** k < b:
        k += 1
    max_range = a ** k
    max_accept = (max_range // b) * b
    return k * max_range / max_accept


# =============================================================================
# 验证工具
# =============================================================================

def chi_square_test(gen_func, n, trials=100000):
    """
    卡方检验验证均匀性。

    自由度 = n-1, 95%% 临界值 ≈ 3.84 (n=2), 9.49 (n=5), 16.92 (n=10)
    """
    counts = Counter()
    for _ in range(trials):
        counts[gen_func()] += 1

    expected = trials / n
    chi2 = sum((counts[i] - expected) ** 2 / expected for i in range(1, n + 1))

    # 计算理论 95% 临界值 (近似)
    # 实际上卡方分布临界值需要查表，这里只返回统计量
    return counts, chi2


def estimate_expected_calls(gen_func, trials=10000):
    """估计实际期望调用 rand7 次数。"""
    # 通过统计总调用次数 / 成功次数
    # 这里简单用调用 gen_func 多次，统计平均
    # 需要 instrumentation，暂时略过
    pass


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("🎲 用 Rand7() 构造 Rand10() — 拒绝采样")
    print("=" * 70)
    print()

    # ========== 核心解法 ==========
    print("【核心解法】拒绝采样")
    print("-" * 50)
    counts, chi2 = chi_square_test(rand10, 10, 100000)
    print(f"经典解法 (2次rand7基础, 拒绝9个数):")
    print(f"  各数字频率: ", end="")
    for i in range(1, 11):
        print(f"{i}:{counts[i]}", end=" ")
    print()
    print(f"  卡方统计量: {chi2:.4f}")
    print(f"  理论期望 rand7 调用: {2 * 49/40:.4f} 次")
    print()

    counts2, chi22 = chi_square_test(rand10_optimized, 10, 100000)
    print(f"优化版 (3次rand7基础, 拒绝3个数):")
    print(f"  各数字频率: ", end="")
    for i in range(1, 11):
        print(f"{i}:{counts2[i]}", end=" ")
    print()
    print(f"  卡方统计量: {chi22:.4f}")
    print(f"  理论期望 rand7 调用: {3 * 343/340:.4f} 次")
    print()

    # ========== 通用方法 ==========
    print("【通用方法】rand(a) → rand(b)")
    print("-" * 50)
    test_cases = [
        (rand7, 7, 10, "rand7→rand10"),
        (rand7, 7, 3, "rand7→rand3"),
        (rand7, 7, 2, "rand7→rand2"),
        (lambda: random.randint(1, 5), 5, 3, "rand5→rand3"),
        (lambda: random.randint(1, 2), 2, 7, "rand2→rand7"),
        (lambda: random.randint(1, 3), 3, 10, "rand3→rand10"),
    ]

    print(f"{'构造':>15} | {'k':>3} | {'期望调用':>10} | {'卡方(n=?)':>12}")
    print("-" * 50)
    for rand_a, a, b, name in test_cases:
        gen = lambda: rand_a_to_rand_b(rand_a, a, b)
        counts, chi2 = chi_square_test(gen, b, 50000)
        k = 1
        while a ** k < b:
            k += 1
        exp_calls = theoretical_expected_calls(a, b)
        print(f"{name:>15} | {k:>3} | {exp_calls:>10.4f} | {chi2:>12.4f}")
    print()

    # ========== 原理讲解 ==========
    print("【原理: 为什么拒绝采样保持均匀性?】")
    print("  设生成 1~49 均匀分布，拒绝 41~49 保留 1~40。")
    print("  P(接受) = 40/49。")
    print("  条件概率: P(num=x | 接受) = P(num=x) / P(接受) = (1/49) / (40/49) = 1/40")
    print("  因此 1~40 在接受条件下仍均匀。")
    print("  再将 1~40 分成 4 组: {1,11,21,31}, {2,12,22,32}, ..., {10,20,30,40}")
    print("  每组映射到 1~10，仍均匀。")
    print()

    print("【效率对比】")
    print(f"  方法                | 期望 rand7 调用 | 拒绝率")
    print(f"  经典 (1~49→1~10)   | {2 * 49/40:>8.4f}       | {9/49*100:.1f}%")
    print(f"  优化 (1~343→1~10)  | {3 * 343/340:>8.4f}       | {3/343*100:.1f}%")
    print(f"  二进制 (4 bits)     | {4 * 16/10:>8.4f}       | {6/16*100:.1f}%")
    print()

    print("关键观察:")
    print("  • 经典解法最简单，期望 2.45 次 rand7 调用")
    print("  • 优化版拒绝率更低，但期望调用次数反而略高 (3.03)")
    print("  • 最优策略取决于 rand7 的调用成本 vs 循环开销")
    print("  • 通用方法适用于任意 rand(a)→rand(b) 的构造")


if __name__ == "__main__":
    demo()
