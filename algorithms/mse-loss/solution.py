"""
手写 MSE (Mean Squared Error) 损失函数

重点：实现效率、数值稳定性、梯度推导。
"""

import math
import torch
import torch.nn as nn


# ============================================================
# 1. 纯 Python 实现
# ============================================================

def mse_naive(y_true: list[float], y_pred: list[float]) -> float:
    """Naive：两次遍历（一次算差，一次求和）。"""
    n = len(y_true)
    total = 0.0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        total += diff * diff
    return total / n


def mse_oneliner(y_true: list[float], y_pred: list[float]) -> float:
    """Pythonic 一行版：生成器表达式，单次遍历，O(1) 额外空间。"""
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n


def mse_stable(y_true: list[float], y_pred: list[float]) -> float:
    """
    Welford 在线算法风格：增量更新均值，避免大数组求和溢出。

    适用于 n 极大或流式数据场景。
    普通求和在 n=10^8 量级 float32 时会丢失精度（大数吃小数）。
    """
    n = len(y_true)
    mean_se = 0.0
    for i in range(n):
        diff_sq = (y_true[i] - y_pred[i]) ** 2
        mean_se += (diff_sq - mean_se) / (i + 1)  # 增量更新均值
    return mean_se


def mse_kahan(y_true: list[float], y_pred: list[float]) -> float:
    """
    Kahan 补偿求和：用一个补偿变量追踪丢失的低位精度。

    精度接近 float64，开销只多一次加减。
    """
    n = len(y_true)
    total = 0.0
    comp = 0.0  # 补偿项
    for i in range(n):
        diff_sq = (y_true[i] - y_pred[i]) ** 2
        y = diff_sq - comp
        t = total + y
        comp = (t - total) - y  # 丢失的低位
        total = t
    return total / n


# ============================================================
# 2. PyTorch 实现
# ============================================================

def mse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """手写 PyTorch MSE，等价于 F.mse_loss(y_pred, y_true)。"""
    return (y_true - y_pred).pow(2).mean()


def mse_torch_reduction(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """支持 mean/sum/none 三种 reduction，和 nn.MSELoss 行为一致。"""
    se = (y_true - y_pred).pow(2)
    if reduction == "mean":
        return se.mean()
    elif reduction == "sum":
        return se.sum()
    else:  # "none"
        return se


# ============================================================
# 3. 效率对比
# ============================================================

def benchmark():
    """对比不同实现的效率。"""
    import time

    sizes = [100, 10_000, 1_000_000]

    for n in sizes:
        y_true = [float(i) / n for i in range(n)]
        y_pred = [float(i) / n + 0.01 * (i % 7 - 3) for i in range(n)]

        # naive
        t0 = time.perf_counter()
        r1 = mse_naive(y_true, y_pred)
        t1 = time.perf_counter()

        # generator
        t2 = time.perf_counter()
        r2 = mse_oneliner(y_true, y_pred)
        t3 = time.perf_counter()

        # welford
        t4 = time.perf_counter()
        r3 = mse_stable(y_true, y_pred)
        t5 = time.perf_counter()

        print(f"n={n:>10,d}: naive={t1-t0:.4f}s  generator={t3-t2:.4f}s  "
              f"welford={t5-t4:.4f}s  (值:{r1:.8f})")


# ============================================================
# 数值演示
# ============================================================

def demo():
    print("=" * 60)
    print("MSE Loss 手写实现演示")
    print("=" * 60)

    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]

    print(f"\ny_true = {y_true}")
    print(f"y_pred = {y_pred}")
    print(f"diff   = {[t-p for t, p in zip(y_true, y_pred)]}")
    print(f"diff²  = {[(t-p)**2 for t, p in zip(y_true, y_pred)]}")

    result = mse_oneliner(y_true, y_pred)
    print(f"\nMSE = mean({[(t-p)**2 for t, p in zip(y_true, y_pred)]}) = {result}")

    # 手动验证
    manual = ((3-2.5)**2 + (-0.5-0)**2 + (2-2)**2 + (7-8)**2) / 4
    print(f"手算 = (0.25 + 0.25 + 0 + 1) / 4 = {manual}")

    # 梯度
    print(f"\n--- 梯度 ---")
    print(f"∂MSE/∂y_pred_i = 2(y_pred_i - y_true_i) / n")
    n = len(y_true)
    grads = [2 * (p - t) / n for t, p in zip(y_true, y_pred)]
    print(f"梯度 = {[f'{g:.4f}' for g in grads]}")

    # 数值稳定性对比
    print(f"\n--- 数值稳定性 (大数组, float 精度) ---")
    import random
    random.seed(42)
    n = 100_000
    y_t = [random.gauss(1e8, 1) for _ in range(n)]  # 大值 + 小差异
    y_p = [t + random.gauss(0, 0.01) for t in y_t]

    r_naive = mse_naive(y_t, y_p)
    r_stable = mse_stable(y_t, y_p)
    r_kahan = mse_kahan(y_t, y_p)
    print(f"Naive:   {r_naive:.12f}")
    print(f"Welford: {r_stable:.12f}")
    print(f"Kahan:   {r_kahan:.12f}")
    print(f"→ 大数场景下 Welford/Kahan 更可靠")

    # MSE vs MAE vs Huber
    print(f"\n--- MSE vs MAE vs Huber (异常值影响) ---")
    y_t2 = [1.0, 2.0, 3.0, 4.0, 100.0]  # 最后一个是异常值
    y_p2 = [1.1, 2.1, 3.1, 4.1, 5.0]

    mse_val = sum((t-p)**2 for t,p in zip(y_t2, y_p2)) / len(y_t2)
    mae_val = sum(abs(t-p) for t,p in zip(y_t2, y_p2)) / len(y_t2)
    delta = 1.0
    huber_val = sum(
        0.5*(t-p)**2 if abs(t-p) <= delta else delta*(abs(t-p) - 0.5*delta)
        for t,p in zip(y_t2, y_p2)
    ) / len(y_t2)

    print(f"数据: y_true={y_t2}, y_pred={y_p2}")
    print(f"MSE   = {mse_val:.2f}  ← 被异常值 (100-5)² = 9025 主导")
    print(f"MAE   = {mae_val:.2f}  ← 异常值影响线性")
    print(f"Huber = {huber_val:.2f}  ← 异常值影响被截断到线性")

    # 性能
    print(f"\n--- 效率对比 ---")
    benchmark()


if __name__ == "__main__":
    demo()
