"""
手写数值稳定的 Softmax — 纯 Python + PyTorch 实现

重点：naive 版本为什么溢出，-max 技巧为什么有效，以及 Online Softmax。
"""

import math
import torch
import torch.nn as nn


# ============================================================
# 1. Naive Softmax（会溢出）
# ============================================================
def softmax_naive(x: list[float]) -> list[float]:
    """
    直接按定义算：softmax(x_i) = exp(x_i) / Σ exp(x_j)

    问题：当 x_i 很大时 exp(x_i) → inf，当 x_i 很小时 exp(x_i) → 0
    """
    exps = [math.exp(xi) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================
# 2. 数值稳定的 Softmax（减最大值）
# ============================================================
def softmax_stable(x: list[float]) -> list[float]:
    """
    先减去最大值再算 exp：softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)

    数学上完全等价（证明见 README），但 exp 的参数 ≤ 0，不会溢出。
    """
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================
# 3. PyTorch 实现
# ============================================================
def softmax_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """手写的 PyTorch 数值稳定 softmax。"""
    m = x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x - m)
    return exps / exps.sum(dim=dim, keepdim=True)


# ============================================================
# 4. Online Softmax（单次遍历，FlashAttention 的基础）
# ============================================================
def softmax_online(x: list[float]) -> list[float]:
    """
    Online Softmax：只遍历一次就完成 max + exp + sum。

    传统做法需要 3 次遍历（求 max → 求 exp 和 sum → 除以 sum），
    Online 版本在一次遍历中同时维护 max 和 sum，
    遇到新的 max 时修正已有的 sum。这是 FlashAttention 的核心技巧。
    """
    n = len(x)
    m = float('-inf')  # running max
    d = 0.0            # running sum of exp

    # Pass 1: 在一次遍历中同时求 max 和 exp-sum
    for i in range(n):
        if x[i] > m:
            # 新的 max 出现，修正之前的 sum
            d = d * math.exp(m - x[i]) + 1.0
            m = x[i]
        else:
            d += math.exp(x[i] - m)

    # Pass 2: 求每个元素的 softmax 值
    return [math.exp(xi - m) / d for xi in x]


# ============================================================
# 5. Log-Softmax（用于交叉熵损失）
# ============================================================
def log_softmax_stable(x: list[float]) -> list[float]:
    """
    log_softmax(x_i) = x_i - max - log(Σ exp(x_j - max))

    直接算 log(softmax) 会因为 softmax 值太小而下溢（log(0) = -inf），
    用 log-sum-exp 技巧绕过中间的小数值。
    """
    m = max(x)
    log_sum_exp = m + math.log(sum(math.exp(xi - m) for xi in x))
    return [xi - log_sum_exp for xi in x]


# ============================================================
# 数值演示
# ============================================================
def demo():
    print("=" * 60)
    print("Softmax 数值稳定性演示")
    print("=" * 60)

    # Case 1: 正常输入
    x1 = [2.0, 1.0, 0.1]
    print(f"\n--- Case 1: 正常输入 {x1} ---")
    naive = softmax_naive(x1)
    stable = softmax_stable(x1)
    online = softmax_online(x1)
    print(f"Naive:  [{', '.join(f'{v:.6f}' for v in naive)}]  sum={sum(naive):.10f}")
    print(f"Stable: [{', '.join(f'{v:.6f}' for v in stable)}]  sum={sum(stable):.10f}")
    print(f"Online: [{', '.join(f'{v:.6f}' for v in online)}]  sum={sum(online):.10f}")

    # Case 2: 大值输入（naive 会溢出）
    x2 = [1000.0, 1001.0, 1002.0]
    print(f"\n--- Case 2: 大值输入 {x2} ---")
    try:
        naive = softmax_naive(x2)
        print(f"Naive:  [{', '.join(f'{v}' for v in naive)}]")
    except OverflowError:
        print(f"Naive:  OverflowError! exp(1000) 太大了")

    stable = softmax_stable(x2)
    online = softmax_online(x2)
    print(f"Stable: [{', '.join(f'{v:.6f}' for v in stable)}]  sum={sum(stable):.10f}")
    print(f"Online: [{', '.join(f'{v:.6f}' for v in online)}]  sum={sum(online):.10f}")

    # Case 3: 极端差异
    x3 = [100.0, 0.0, -100.0]
    print(f"\n--- Case 3: 极端差异 {x3} ---")
    stable = softmax_stable(x3)
    print(f"Stable: [{', '.join(f'{v:.6e}' for v in stable)}]")
    print(f"→ 第一个元素几乎独占全部概率")

    # Case 4: 全相同
    x4 = [5.0, 5.0, 5.0, 5.0]
    print(f"\n--- Case 4: 全相同 {x4} ---")
    stable = softmax_stable(x4)
    print(f"Stable: [{', '.join(f'{v:.6f}' for v in stable)}]")
    print(f"→ 均匀分布 1/4 = 0.25")

    # Case 5: Log-Softmax
    x5 = [2.0, 1.0, 0.1]
    print(f"\n--- Case 5: Log-Softmax {x5} ---")
    log_sm = log_softmax_stable(x5)
    sm = softmax_stable(x5)
    log_of_sm = [math.log(s) for s in sm]
    print(f"log_softmax:  [{', '.join(f'{v:.6f}' for v in log_sm)}]")
    print(f"log(softmax): [{', '.join(f'{v:.6f}' for v in log_of_sm)}]")
    print(f"差异: [{', '.join(f'{abs(a-b):.2e}' for a, b in zip(log_sm, log_of_sm))}]")

    # 关键数字
    print(f"\n{'='*60}")
    print("float64/float32 溢出边界")
    print("=" * 60)
    print(f"float64 max: {1.7976931348623157e+308:.2e}")
    print(f"exp(709)   = {math.exp(709):.2e}  (OK)")
    try:
        print(f"exp(710)   = {math.exp(710):.2e}")
    except OverflowError:
        print(f"exp(710)   = OverflowError!")
    print(f"float32 max: ~3.4e+38, exp(88) ≈ 1.6e+38 (OK), exp(89) → inf")
    print(f"\n→ -max 技巧保证所有 exp 参数 ≤ 0，最大值处 exp(0)=1，绝不溢出")


if __name__ == "__main__":
    demo()
