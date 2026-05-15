"""
优惠券收集问题（Coupon Collector's Problem）
=============================================

问题：有 N 种不同的奖券，每次随机获得一种（均匀分布），
      集齐所有 N 种奖券平均需要多少次抽取？

特例：N = 12（十二星座），平均需要遇到多少人？

数学结论：
  E[T] = N * H_N = N * (1 + 1/2 + 1/3 + ... + 1/N)
  其中 H_N 是第 N 个调和数。
"""

import math
import random


# =============================================================================
# 数学解法
# =============================================================================

def harmonic_number(n: int) -> float:
    """计算第 n 个调和数 H_n = sum_{i=1}^n 1/i。"""
    return sum(1.0 / i for i in range(1, n + 1))


def coupon_collector_expectation(n: int) -> float:
    """集齐 N 种奖券的期望抽取次数。"""
    return n * harmonic_number(n)


def coupon_collector_variance(n: int) -> float:
    """
    集齐 N 种奖券的方差。

    推导：
    T = T_1 + T_2 + ... + T_N
    其中 T_k ~ Geometric(p_k), p_k = (N-k+1)/N
    Var(T_k) = (1-p_k) / p_k^2 = (k-1)/N / ((N-k+1)/N)^2 = N(k-1) / (N-k+1)^2

    令 j = N-k+1，则 k-1 = N-j
    Var(T) = sum_{j=1}^N N(N-j) / j^2
           = N^2 * sum_{j=1}^N 1/j^2 - N * sum_{j=1}^N 1/j
           = N^2 * H_N^{(2)} - N * H_N
    其中 H_N^{(2)} = sum_{j=1}^N 1/j^2
    """
    h_n = harmonic_number(n)
    h_n_2 = sum(1.0 / (i * i) for i in range(1, n + 1))
    return n * n * h_n_2 - n * h_n


def coupon_collector_approximation(n: int) -> float:
    """
    大 N 近似公式：
    E[T] ≈ N ln N + γN + 1/2
    其中 γ ≈ 0.57721566 是 Euler-Mascheroni 常数
    """
    gamma = 0.5772156649015329
    return n * math.log(n) + gamma * n + 0.5


# =============================================================================
# Monte Carlo 模拟
# =============================================================================

def simulate_coupon_collector(n: int) -> int:
    """模拟一次收集过程，返回所需抽取次数。"""
    collected = set()
    draws = 0
    while len(collected) < n:
        collected.add(random.randint(0, n - 1))
        draws += 1
    return draws


def monte_carlo(n: int, rounds: int = 100000) -> tuple[float, float]:
    """
    Monte Carlo 模拟，返回样本均值和样本方差。
    """
    results = [simulate_coupon_collector(n) for _ in range(rounds)]
    mean = sum(results) / len(results)
    variance = sum((x - mean) ** 2 for x in results) / len(results)
    return mean, variance


# =============================================================================
# 演示与验证
# =============================================================================

def demo():
    print("=" * 70)
    print("优惠券收集问题（Coupon Collector's Problem）")
    print("=" * 70)
    print()

    # N = 12（十二星座）
    print("[特例] N = 12（十二星座）")
    print("-" * 50)
    n = 12
    h_n = harmonic_number(n)
    expected = coupon_collector_expectation(n)
    var = coupon_collector_variance(n)
    approx = coupon_collector_approximation(n)

    print(f"  调和数 H_12 = {h_n:.6f}")
    print(f"  精确期望 E[T] = 12 × H_12 = {expected:.4f}")
    print(f"  标准差 σ = {math.sqrt(var):.4f}")
    print(f"  大 N 近似 = {approx:.4f}")
    print()

    # Monte Carlo 验证
    print("  Monte Carlo 验证（10万次模拟）...")
    mc_mean, mc_var = monte_carlo(n, rounds=100000)
    print(f"  模拟均值 = {mc_mean:.4f}")
    print(f"  模拟方差 = {mc_var:.4f}")
    print(f"  与理论值偏差: {abs(mc_mean - expected):.4f} ({abs(mc_mean - expected)/expected * 100:.2f}%)")
    print()

    # 泛化：不同 N 的对比
    print("[泛化] 不同 N 的期望值对比")
    print("-" * 70)
    print(f"{'N':>5} | {'精确 E[T]':>12} | {'近似 E[T]':>12} | {'标准差':>10} | {'相对误差':>10}")
    print("-" * 70)

    for n in [2, 3, 5, 10, 12, 20, 50, 100, 200, 500, 1000]:
        exact = coupon_collector_expectation(n)
        approx = coupon_collector_approximation(n)
        std = math.sqrt(coupon_collector_variance(n))
        rel_err = abs(approx - exact) / exact * 100
        print(f"{n:>5} | {exact:>12.2f} | {approx:>12.2f} | {std:>10.2f} | {rel_err:>9.2f}%")

    print("-" * 70)
    print()

    # 直观解释：每一步的期望
    print("[分步拆解] N = 12 时，收集第 k 个新星座的期望等待时间")
    print("-" * 60)
    print(f"{'k':>3} | {'已有种类':>8} | {'新星座概率':>12} | {'期望等待':>10}")
    print("-" * 60)
    total = 0.0
    for k in range(1, 13):
        p_new = (12 - k + 1) / 12
        wait = 1.0 / p_new
        total += wait
        print(f"{k:>3} | {k-1:>8} | {p_new:>12.4f} | {wait:>10.2f}")
    print("-" * 60)
    print(f"{'总计':>3} | {'':8} | {'':12} | {total:>10.2f}")
    print()

    # 生日问题的对比
    print("[趣味对比] 优惠券收集 vs 生日问题")
    print("-" * 60)
    print("  生日问题：23 人中至少两人生日相同的概率 > 50%")
    print("  优惠券收集：集齐 365 种生日，平均需要 E[T] = 365 × H_365 人")
    n_bd = 365
    e_bd = coupon_collector_expectation(n_bd)
    a_bd = coupon_collector_approximation(n_bd)
    print(f"  精确值: {e_bd:.0f} 人")
    print(f"  近似值: {a_bd:.0f} 人")
    print(f"  → 约 {e_bd / 365:.1f} 倍于一年天数！")


if __name__ == "__main__":
    demo()
