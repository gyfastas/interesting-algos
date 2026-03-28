"""
手写 LayerNorm 和 RMSNorm — PyTorch 实现

重点：class 定义 + forward，对比两者的计算差异。
"""

import torch
import torch.nn as nn


# ============================================================
# 1. Layer Normalization
#    Ba et al., 2016 — "Layer Normalization"
#    对每个 token 的特征维度做归一化（减均值、除标准差）
# ============================================================
class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # 可学习缩放
        self.beta = nn.Parameter(torch.zeros(d_model))    # 可学习偏移
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)  — 在最后一维（特征维）归一化
        mean = x.mean(dim=-1, keepdim=True)               # E[x]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Var[x]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 标准化
        return self.gamma * x_norm + self.beta              # 仿射变换


# ============================================================
# 2. RMS Normalization
#    Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization"
#    去掉均值中心化，只用 RMS 做缩放，也去掉 beta 偏移
# ============================================================
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))    # 可学习缩放
        self.eps = eps
        # 注意：没有 beta（偏移项）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # RMS(x)
        x_norm = x / rms                                                   # 只除以 RMS
        return self.gamma * x_norm


# ============================================================
# 纯 Python 数值验证（不依赖 PyTorch）
# ============================================================
def layernorm_numpy(x, gamma, beta, eps=1e-6):
    """纯 Python 实现 LayerNorm，用于验证。"""
    mean = sum(x) / len(x)
    var = sum((xi - mean) ** 2 for xi in x) / len(x)
    return [(g * (xi - mean) / (var + eps) ** 0.5 + b)
            for xi, g, b in zip(x, gamma, beta)]


def rmsnorm_numpy(x, gamma, eps=1e-6):
    """纯 Python 实现 RMSNorm，用于验证。"""
    rms = (sum(xi ** 2 for xi in x) / len(x) + eps) ** 0.5
    return [g * xi / rms for xi, g in zip(x, gamma)]


def verify():
    """数值对比验证。"""
    import random
    random.seed(42)

    D = 8
    x = [random.gauss(0, 2) for _ in range(D)]
    gamma = [1.0] * D
    beta = [0.0] * D

    ln_out = layernorm_numpy(x, gamma, beta)
    rms_out = rmsnorm_numpy(x, gamma)

    print("=" * 50)
    print("数值验证 (D=8, gamma=1, beta=0)")
    print("=" * 50)
    print(f"{'输入 x':>12}  {'LayerNorm':>12}  {'RMSNorm':>12}")
    print("-" * 50)
    for i in range(D):
        print(f"{x[i]:12.4f}  {ln_out[i]:12.4f}  {rms_out[i]:12.4f}")

    # 统计量
    ln_mean = sum(ln_out) / D
    ln_var = sum((v - ln_mean) ** 2 for v in ln_out) / D
    rms_sq_mean = sum(v ** 2 for v in rms_out) / D

    print("-" * 50)
    print(f"LayerNorm 输出: mean={ln_mean:.6f}, var={ln_var:.6f}")
    print(f"  → 均值≈0, 方差≈1 ✓")
    print(f"RMSNorm  输出: E[x²]={rms_sq_mean:.6f}")
    print(f"  → 均方≈1 ✓ (但均值不一定为0)")

    # 计算量对比
    print("\n" + "=" * 50)
    print("计算量对比 (每个 token, D 维特征)")
    print("=" * 50)
    print(f"{'操作':<20} {'LayerNorm':>12} {'RMSNorm':>12}")
    print("-" * 50)
    print(f"{'求均值 E[x]':<20} {'D 次加法':>12} {'—':>12}")
    print(f"{'中心化 x-μ':<20} {'D 次减法':>12} {'—':>12}")
    print(f"{'求方差/RMS':<20} {'D 次乘加':>12} {'D 次乘加':>12}")
    print(f"{'归一化 x/σ':<20} {'D 次除法':>12} {'D 次除法':>12}")
    print(f"{'缩放 γ·x':<20} {'D 次乘法':>12} {'D 次乘法':>12}")
    print(f"{'偏移 +β':<20} {'D 次加法':>12} {'—':>12}")
    print("-" * 50)
    print(f"{'总计':<20} {'~5D':>12} {'~3D':>12}")
    print(f"{'节省':<20} {'':>12} {'~40%':>12}")


if __name__ == "__main__":
    verify()
