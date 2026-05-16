"""
坏芯片检测问题 (Faulty Chip Detection)
=======================================

B 个芯片中恰有 1 个是坏的。
有 T 次测试机会，N 个检查器。
规则: 检查器插入的芯片组中包含坏芯片时，该检查器坏掉(不可再用)。

求: 给定 B, T，至少需要多少个检查器 N 才能一定找出坏芯片?
"""

import math
import random


# =============================================================================
# 核心解法: 下降阶乘 (Falling Factorial)
# =============================================================================

def falling_factorial(N, T):
    """
    下降阶乘: P(N, T) = N * (N-1) * ... * (N-T+1)
    如果 T > N，则只计算到 N 个因子(因为 N 轮后检查器全部耗尽)。
    """
    effective_T = min(T, N)
    prod = 1
    for i in range(effective_T):
        prod *= (N - i)
    return prod


def min_checkers(B, T):
    """
    求最小 N 使得下降阶乘 P(N, T) >= B。

    信息论下界: T 轮测试, 每轮可用检查器递减,
    最多能区分 P(N, T) = N*(N-1)*...*(N-T+1) 种情况。
    因此需要 P(N, T) >= B。
    """
    if B <= 1:
        return 0
    lo, hi = 1, max(B, T)
    ans = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if falling_factorial(mid, T) >= B:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


# =============================================================================
# 构造性策略: 路径编码 (Path Encoding)
# =============================================================================

def encode_strategy(B, N, T):
    """
    构造检测策略: 给每个芯片分配一条唯一"路径"。

    路径定义: 长度为 T 的元组 (c1, c2, ..., cT)，
    其中 ct 表示第 t 轮该芯片被放入第 ct 个检查器。
    第 t 轮有 (N-t+1) 个可用检查器，因此 ct ∈ [0, N-t]。

    总路径数 = N * (N-1) * ... * (N-T+1) = P(N, T)。
    若 P(N, T) >= B，则可为每个芯片分配唯一路径。

    编码方法: 把 chip_id 视为混合进制数，
    第 t 位的基数为 (N-t+1)。
    """
    paths = []
    for chip_id in range(B):
        path = []
        remaining = chip_id
        for t in range(T):
            base = N - t  # 第 t 轮可用的检查器数
            path.append(remaining % base)
            remaining //= base
        paths.append(tuple(path))
    return paths


def simulate_detection(B, T, N, bad_chip, verbose=False):
    """
    用路径编码策略模拟检测过程，验证能否找到坏芯片。

    返回: (是否成功, 实际使用轮数)
    """
    paths = encode_strategy(B, N, T)
    candidates = list(range(B))
    available = list(range(N))  # 可用检查器编号

    if verbose:
        print(f"  坏芯片 = {bad_chip}, 路径 = {paths[bad_chip]}")

    for t in range(T):
        if len(candidates) <= 1:
            break

        m = len(available)
        # 按路径第 t 位分组
        groups = [[] for _ in range(m)]
        for chip in candidates:
            checker_idx = paths[chip][t]
            groups[checker_idx].append(chip)

        # 坏芯片所在的组
        bad_checker_idx = paths[bad_chip][t]
        if verbose:
            print(f"  第{t+1}轮: {m}个检查器, 分组大小 = {[len(g) for g in groups]}, "
                  f"坏检查器索引 = {bad_checker_idx}")

        # 该检查器坏掉
        available.pop(bad_checker_idx)
        candidates = groups[bad_checker_idx]

    success = (len(candidates) == 1 and candidates[0] == bad_chip)
    return success, min(t + 1, T)


# =============================================================================
# 贪心策略对比 (非最优，用于展示)
# =============================================================================

def simulate_greedy(B, T, N, bad_chip):
    """
    贪心策略: 每轮把候选芯片尽可能均匀分配到当前可用检查器中。
    注意: 贪心策略不保证最优，可能需要的 N 更大。
    """
    candidates = list(range(B))
    available = list(range(N))

    for t in range(T):
        if len(candidates) <= 1:
            break
        m = len(available)
        if m == 0:
            return False  # 没有可用检查器但仍未确定
        # 均匀分组
        groups = [[] for _ in range(m)]
        for i, chip in enumerate(candidates):
            groups[i % m].append(chip)

        # 找到坏芯片所在组
        bad_group = None
        for i, g in enumerate(groups):
            if bad_chip in g:
                bad_group = i
                break

        available.pop(bad_group)
        candidates = groups[bad_group]

    return len(candidates) == 1 and candidates[0] == bad_chip


def min_checkers_greedy(B, T):
    """用贪心策略模拟找最小 N(通常比理论最优大)。"""
    for N in range(1, B + 1):
        ok = True
        for bad in range(B):
            if not simulate_greedy(B, T, N, bad):
                ok = False
                break
        if ok:
            return N
    return B


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("🔬 坏芯片检测问题")
    print("=" * 70)
    print()

    # 核心结论
    print("【核心结论】")
    print("  最小检查器数 N 满足: N * (N-1) * ... * (N-T+1) >= B")
    print("  即下降阶乘 P(N, T) >= B")
    print()

    # 常用对照表
    print("【常用对照表】")
    print(f"{'B':>6} | {'T':>3} | {'最小 N':>6} | {'P(N,T)':>8} | {'贪心 N':>6}")
    print("-" * 45)
    test_cases = [
        (2, 1), (10, 1), (100, 1),
        (4, 2), (6, 2), (10, 2), (12, 2),
        (6, 3), (10, 3), (24, 3), (50, 3), (60, 3),
        (24, 4), (50, 4),
        (100, 2), (100, 3), (100, 4), (100, 5),
        (120, 5),
    ]
    for B, T in test_cases:
        N = min_checkers(B, T)
        p = falling_factorial(N, T)
        N_greedy = min_checkers_greedy(B, T)
        print(f"{B:>6} | {T:>3} | {N:>6} | {p:>8} | {N_greedy:>6}")
    print()

    # 构造性策略验证
    print("【构造性策略验证】(路径编码)")
    verify_cases = [
        (6, 2), (10, 3), (50, 3), (24, 4), (100, 4),
    ]
    for B, T in verify_cases:
        N = min_checkers(B, T)
        ok = True
        for bad in range(B):
            success, rounds = simulate_detection(B, T, N, bad)
            if not success:
                ok = False
                break
        status = "✓ 全部通过" if ok else "✗ 存在失败"
        print(f"  B={B:3d}, T={T}, N={N}: {status}")
    print()

    # 详细模拟一个例子
    print("【详细模拟】B=10, T=3, N=4")
    print("-" * 45)
    B, T, N = 10, 3, 4
    paths = encode_strategy(B, N, T)
    print("  路径编码 (chip_id → path):")
    for i in range(B):
        print(f"    芯片{i}: {paths[i]}")
    print()

    for bad_chip in [0, 7, 9]:
        print(f"  >>> 假设芯片 {bad_chip} 是坏芯片:")
        simulate_detection(B, T, N, bad_chip, verbose=True)
        print()

    # 信息论解释
    print("【信息论解释】")
    print("  每轮测试后，恰有 1 个检查器坏掉(坏芯片所在的那个)。")
    print("  因此第 t 轮有 (N-t+1) 种可能结果(哪个检查器坏)。")
    print("  T 轮总信息量 = N × (N-1) × ... × (N-T+1) 种路径。")
    print("  要区分 B 个芯片，需要路径数 ≥ B，即 P(N,T) ≥ B。")
    print()

    # 边界情况
    print("【边界情况】")
    print(f"  T=1 (只能测1轮): 需要 N ≥ B，每个检查器测1个芯片")
    print(f"  T≥N (轮数充足):  需要 N! ≥ B，N 个检查器各用1次")
    print(f"  B=2, T任意:      只需要 N=2，第1轮即可区分")


if __name__ == "__main__":
    demo()
