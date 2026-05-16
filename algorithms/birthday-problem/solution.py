"""
生日问题 (Birthday Problem)
============================

在 n 个人的群体中，至少有两个人生日相同的概率是多少？
假设一年 365 天，生日均匀分布，不考虑闰年。
"""

import math
import random


# =============================================================================
# 数学精确解
# =============================================================================

def birthday_probability_exact(n, days=365):
    """
    精确计算：P(至少两人生日相同) = 1 - P(所有人生日都不同)

    P(所有不同) = 365/365 × 364/365 × 363/365 × ... × (365-n+1)/365
                = 365! / ((365-n)! × 365^n)
    """
    if n > days:
        return 1.0  # 鸽巢原理：人数超过天数必然有重复

    p_all_distinct = 1.0
    for i in range(n):
        p_all_distinct *= (days - i) / days

    return 1.0 - p_all_distinct


def birthday_probability_log(n, days=365):
    """
    用对数计算，避免大数 n! 溢出。
    ln(P) = sum_{i=0}^{n-1} ln((days-i)/days)
    """
    if n > days:
        return 1.0
    log_p = 0.0
    for i in range(n):
        log_p += math.log((days - i) / days)
    return 1.0 - math.exp(log_p)


# =============================================================================
# Monte Carlo 模拟
# =============================================================================

def monte_carlo_simulation(n, trials=100_000, days=365, seed=None):
    """
    用随机模拟估计概率。
    返回 (估计概率, 95% 置信区间半宽)
    """
    if seed is not None:
        random.seed(seed)

    count = 0
    for _ in range(trials):
        birthdays = [random.randint(1, days) for _ in range(n)]
        if len(set(birthdays)) < n:
            count += 1

    p_hat = count / trials
    # 95% CI for binomial: p_hat ± 1.96 * sqrt(p_hat*(1-p_hat)/trials)
    std_err = math.sqrt(p_hat * (1 - p_hat) / trials)
    ci_half = 1.96 * std_err
    return p_hat, ci_half


# =============================================================================
# 寻找临界点：概率首次超过 50% 的人数
# =============================================================================

def find_critical_threshold(target_p=0.5, days=365):
    """找到使 P(至少一对) >= target_p 的最小人数 n。"""
    for n in range(1, days + 2):
        p = birthday_probability_exact(n, days)
        if p >= target_p:
            return n, p
    return days + 1, 1.0


# =============================================================================
# 演示与对比
# =============================================================================

def demo():
    print("=" * 65)
    print("🎂 生日问题 — 精确解 vs Monte Carlo 模拟")
    print("=" * 65)
    print()

    # 核心问题：50 人
    n = 50
    p_exact = birthday_probability_exact(n)
    print(f"问题：{n} 人的班级中，至少有两个人同一天生日的概率是多少？")
    print(f"精确答案: {p_exact:.6f}  ≈  {p_exact*100:.2f}%")
    print()

    # Monte Carlo 验证
    print("Monte Carlo 验证:")
    for trials in [1_000, 10_000, 100_000]:
        p_sim, ci = monte_carlo_simulation(n, trials, seed=42)
        print(f"  模拟 {trials:>7,} 次: {p_sim*100:6.2f}%  (95% CI: ±{ci*100:.2f}%)")
    print()

    # 关键节点
    print("概率随人数增长的关键节点:")
    print(f"{'人数':>6} | {'P(至少一对)':>14} | {'P(全不同)':>14}")
    print("-" * 45)
    for n in [5, 10, 15, 20, 23, 25, 30, 40, 50, 60, 70]:
        p = birthday_probability_exact(n)
        print(f"{n:>6} | {p:>14.4f} | {1-p:>14.4f}")
    print()

    # 临界点
    n_50, p_50 = find_critical_threshold(0.5)
    n_99, p_99 = find_critical_threshold(0.99)
    print(f"临界点:")
    print(f"  概率首次超过 50% 的人数: n = {n_50} (P = {p_50*100:.2f}%)")
    print(f"  概率首次超过 99% 的人数: n = {n_99} (P = {p_99*100:.2f}%)")
    print()

    # 对数方法的精度对比（大 n）
    print("大人数验证（对数法避免溢出）:")
    for n in [100, 200, 300, 365]:
        p1 = birthday_probability_exact(n)
        p2 = birthday_probability_log(n)
        print(f"  n={n:3d}: 直接法={p1:.6f}, 对数法={p2:.6f}, diff={abs(p1-p2):.2e}")
    print()

    # 鸽巢原理
    print(f"鸽巢原理验证: n=366 时 P={birthday_probability_exact(366)*100:.0f}%")


if __name__ == "__main__":
    demo()
