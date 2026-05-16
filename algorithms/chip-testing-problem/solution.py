"""
坏芯片检测问题 (Faulty Chip Detection)
=======================================

B 个芯片中恰有 1 个是坏的。
有 T 轮测试机会，N 个检查器。
规则: 每轮可将任意芯片放入任意检查器；若某检查器本轮测的芯片中包含坏芯片，
      该检查器在本轮结束后坏掉(不可再用)。

等价于 LeetCode 458 (Poor Pigs):
  检查器 = 猪, 芯片 = 桶, 坏芯片 = 毒药, T 轮 = minutesToTest/minutesToDie

求: 给定 B, T，至少需要多少个检查器 N 才能一定找出坏芯片?
"""

import math
import random


# =============================================================================
# 核心解法: (T+1)^N >= B
# =============================================================================

def min_checkers(B, T):
    """
    最小 N 满足 (T+1)^N >= B。

    信息论解释:
      每个检查器有 (T+1) 种可能状态:
        - 第 1 轮后坏
        - 第 2 轮后坏
        - ...
        - 第 T 轮后坏
        - 始终没坏
      N 个检查器可编码 (T+1)^N 种不同的状态组合。
      要区分 B 个芯片, 需要 (T+1)^N >= B。
    """
    if B <= 1:
        return 0
    if T == 0:
        return B  # 0 轮测试无法获取任何信息
    return math.ceil(math.log(B) / math.log(T + 1))


# =============================================================================
# 构造性策略: (T+1) 进制编码
# =============================================================================

def encode_strategy(B, N, T):
    """
    构造检测策略。

    方法: 把芯片编号用 (T+1) 进制表示, 需要 N 位。
          检查器 i 在第 t 轮(1-indexed)测所有第 i 位等于 t 的芯片。
          第 i 位为 0 的芯片, 检查器 i 从不测它。

    测试后: 若检查器 i 始终没坏 → 第 i 位 = 0
            若检查器 i 第 t 轮后坏 → 第 i 位 = t
            组合 N 个检查器的结果, 解码出芯片编号。
    """
    base = T + 1
    strategy = []
    for chip_id in range(B):
        # 芯片编号的 (T+1) 进制表示, 补零到 N 位
        digits = []
        x = chip_id
        for _ in range(N):
            digits.append(x % base)
            x //= base
        strategy.append(digits)  # digits[i] = 检查器 i 对应的位数
    return strategy


def simulate_detection(B, T, N, bad_chip, verbose=False):
    """
    模拟 (T+1) 进制编码策略检测过程。

    返回: (是否成功, 实际使用轮数)
    """
    strategy = encode_strategy(B, N, T)
    base = T + 1

    if verbose:
        print(f"  坏芯片 = {bad_chip}, 编码 = {strategy[bad_chip]}")

    # 逐轮测试
    for t in range(1, T + 1):
        if verbose:
            tested = [[] for _ in range(N)]
            for chip in range(B):
                for i in range(N):
                    if strategy[chip][i] == t:
                        tested[i].append(chip)
            print(f"  第{t}轮测试分配: {tested}")

        # 检查哪些检查器会坏
        broken = set()
        for i in range(N):
            if strategy[bad_chip][i] == t:
                broken.add(i)

        if verbose:
            print(f"  本轮坏掉的检查器: {sorted(broken) if broken else '无'}")

    # 解码: 根据每个检查器在第几轮坏, 重构芯片编号
    decoded = 0
    for i in range(N):
        digit = strategy[bad_chip][i]
        decoded += digit * (base ** i)

    success = (decoded == bad_chip)
    if verbose:
        print(f"  解码结果: {decoded}, {'✓ 正确' if success else '✗ 错误'}")
    return success, T


def decode_result(strategy, broken_round, B, N, T):
    """
    根据检查结果解码坏芯片编号。

    broken_round[i] = 检查器 i 在第几轮后坏, 若始终没坏则为 0。
    """
    base = T + 1
    chip_id = 0
    for i in range(N):
        chip_id += broken_round[i] * (base ** i)
    return chip_id if chip_id < B else None


# =============================================================================
# Monte Carlo 模拟验证
# =============================================================================

def monte_carlo_verify(B, T, trials=10000, seed=None):
    """
    随机生成坏芯片, 验证 (T+1) 进制策略是否总能正确找出。
    """
    if seed is not None:
        random.seed(seed)

    N = min_checkers(B, T)
    if N == 0:
        return True

    strategy = encode_strategy(B, N, T)
    base = T + 1

    for _ in range(trials):
        bad = random.randint(0, B - 1)

        # 模拟 T 轮
        broken_round = [0] * N  # 0 表示始终没坏
        for t in range(1, T + 1):
            for i in range(N):
                if broken_round[i] == 0 and strategy[bad][i] == t:
                    broken_round[i] = t

        decoded = decode_result(strategy, broken_round, B, N, T)
        if decoded != bad:
            return False, bad, decoded, broken_round

    return True, None, None, None


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("🔬 坏芯片检测问题 (等价于 LeetCode 458 Poor Pigs)")
    print("=" * 70)
    print()

    print("【核心结论】")
    print("  最小检查器数 N 满足: (T+1)^N >= B")
    print("  即 N = ceil(log(B) / log(T+1))")
    print()

    print("【常用对照表】")
    print(f"{'B':>6} | {'T':>3} | {'最小 N':>6} | {'(T+1)^N':>10}")
    print("-" * 32)
    test_cases = [
        (2, 1), (3, 1), (4, 1), (8, 1), (10, 1),
        (2, 2), (4, 2), (8, 2), (9, 2), (27, 2), (28, 2),
        (2, 3), (8, 3), (16, 3), (64, 3), (65, 3),
        (100, 3), (100, 4), (100, 5),
        (1000, 3), (1000, 5), (1000, 7), (1000, 9), (1000, 15),
    ]
    for B, T in test_cases:
        N = min_checkers(B, T)
        print(f"{B:>6} | {T:>3} | {N:>6} | {(T+1)**N:>10}")
    print()

    # 详细模拟一个例子
    print("【详细模拟】B=9, T=2, N=2")
    print("-" * 50)
    B_ex, T_ex, N_ex = 9, 2, 2
    strategy = encode_strategy(B_ex, N_ex, T_ex)
    print("  (T+1)=3 进制编码:")
    for i in range(B_ex):
        print(f"    芯片{i}: {strategy[i]} (3进制) = {i}")
    print()

    for bad_chip in [0, 4, 8]:
        print(f"  >>> 假设芯片 {bad_chip} 是坏芯片:")
        simulate_detection(B_ex, T_ex, N_ex, bad_chip, verbose=True)
        print()

    # Monte Carlo 验证
    print("【Monte Carlo 验证】")
    verify_cases = [
        (8, 2), (27, 2), (64, 3), (100, 3), (1000, 5),
    ]
    for B, T in verify_cases:
        N = min_checkers(B, T)
        ok, bad, decoded, br = monte_carlo_verify(B, T, trials=5000, seed=42)
        status = "✓ 全部通过" if ok else f"✗ 失败 (bad={bad}, decoded={decoded})"
        print(f"  B={B:4d}, T={T}, N={N}: {status}")
    print()

    # 信息论解释
    print("【信息论解释】")
    print("  每个检查器有 (T+1) 种独立状态:")
    print("    0 = 始终没坏")
    print("    1 = 第 1 轮后坏")
    print("    2 = 第 2 轮后坏")
    print("    ...")
    print("    T = 第 T 轮后坏")
    print("  N 个检查器可编码 (T+1)^N 种独立状态组合。")
    print("  要区分 B 个芯片, 需要状态空间 >= B。")
    print()

    print("【与 LeetCode 458 的对应】")
    print("  检查器  ↔  猪")
    print("  芯片    ↔  水桶")
    print("  坏芯片  ↔  毒药水")
    print("  T 轮    ↔  minutesToTest / minutesToDie")
    print("  检查器坏掉 ↔ 猪死亡")
    print()

    print("【边界情况】")
    print("  T=0: 无法进行任何测试, 需要 N=B (每个检查器保一个芯片)")
    print("  T=1: 需要 N=ceil(log2(B)), 每个检查器测一组, 结果是好/坏 1bit")
    print("  B=2: 任意 T>=1, 都只需要 N=1")


if __name__ == "__main__":
    demo()
