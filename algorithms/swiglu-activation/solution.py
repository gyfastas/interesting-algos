"""
手写 SwiGLU 激活函数 + FFN — PyTorch 实现

重点：SiLU、GLU、SwiGLU 的定义，以及 FFN 结构从 2 矩阵到 3 矩阵的变化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 激活函数定义
# ============================================================

def relu(x):
    """ReLU: max(0, x)"""
    return torch.clamp(x, min=0)


def silu(x):
    """SiLU (Swish): x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def gelu(x):
    """GELU: x * Φ(x)，近似实现"""
    return x * 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))


# ============================================================
# 2. 标准 Transformer FFN (GPT-2 / BERT 风格)
#    两个线性层 + ReLU/GELU
# ============================================================
class StandardFFN(nn.Module):
    """
    FFN(x) = W2 · act(W1 · x + b1) + b2

    参数量: D × D_ff + D_ff × D = 2 × D × D_ff (忽略 bias)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)     # up projection
        self.w2 = nn.Linear(d_ff, d_model)      # down projection
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, S, D)
        return self.w2(self.act(self.w1(x)))


# ============================================================
# 3. SwiGLU FFN (Llama / PaLM / DeepSeek 风格)
#    三个线性层：gate + up + down，门控激活
# ============================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN:
      gate = SiLU(W_gate · x)    ← 门控信号
      up   = W_up · x            ← 信息流
      down = W_down · (gate ⊙ up) ← 门控后降维

    参数量: 3 × D × D_ff（比标准 FFN 多 50%）
    所以实际使用时 d_ff 会缩小为 2/3，保持总参数量一致:
      标准 FFN:  2 × D × (4D) = 8D²
      SwiGLU:   3 × D × (8D/3) = 8D²
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # 门控投影
        self.w_up = nn.Linear(d_model, d_ff, bias=False)     # 上投影
        self.w_down = nn.Linear(d_ff, d_model, bias=False)   # 下投影

    def forward(self, x):
        # x: (B, S, D)
        gate = F.silu(self.w_gate(x))   # SiLU 激活的门控信号
        up = self.w_up(x)                # 线性变换的信息流
        return self.w_down(gate * up)    # 门控 + 降维


# ============================================================
# 4. 更紧凑的写法 (Llama 源码风格，gate 和 up 合并)
# ============================================================
class SwiGLUFFNFused(nn.Module):
    """gate 和 up 合并成一个矩阵，推理时少一次 kernel launch。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)  # 合并
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate_up = self.w_gate_up(x)              # (B, S, 2*d_ff)
        gate, up = gate_up.chunk(2, dim=-1)       # 各 (B, S, d_ff)
        return self.w_down(F.silu(gate) * up)


# ============================================================
# 纯 Python 数值演示
# ============================================================
def demo():
    import math

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def silu_py(x):
        return x * sigmoid(x)

    print("=" * 55)
    print("激活函数数值对比")
    print("=" * 55)
    print(f"{'x':>6}  {'ReLU':>8}  {'SiLU':>8}  {'GELU':>8}")
    print("-" * 55)
    for x in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
        r = max(0, x)
        s = silu_py(x)
        # GELU 近似
        g = 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
        print(f"{x:6.1f}  {r:8.4f}  {s:8.4f}  {g:8.4f}")

    print(f"\n{'='*55}")
    print("SwiGLU 门控机制演示 (D=4)")
    print("=" * 55)

    x = [1.0, -0.5, 2.0, -1.0]
    # 模拟 gate 和 up 的线性变换结果
    gate_raw = [0.8, -1.2, 1.5, 0.3]
    up_raw = [0.5, 1.0, -0.3, 0.7]

    gate_activated = [silu_py(g) for g in gate_raw]
    gated_output = [g * u for g, u in zip(gate_activated, up_raw)]

    print(f"{'维度':>4}  {'gate_raw':>10}  {'SiLU(gate)':>10}  {'up':>10}  {'gate⊙up':>10}")
    print("-" * 55)
    for i in range(4):
        print(f"{i:4d}  {gate_raw[i]:10.4f}  {gate_activated[i]:10.4f}  {up_raw[i]:10.4f}  {gated_output[i]:10.4f}")

    print("\n关键观察:")
    print("  gate_raw=-1.2 → SiLU=-0.2689 → 几乎关闭了 up=1.0")
    print("  gate_raw=1.5  → SiLU=1.2802  → 放大了 up=-0.3")
    print("  → 门控让网络选择性地通过/屏蔽信息")

    print(f"\n{'='*55}")
    print("参数量对比 (d_model=4096)")
    print("=" * 55)
    D = 4096

    # 标准 FFN: d_ff = 4D
    d_ff_std = 4 * D
    std_params = 2 * D * d_ff_std
    print(f"标准 FFN (d_ff=4D={d_ff_std}):")
    print(f"  W1: {D}×{d_ff_std} + W2: {d_ff_std}×{D} = {std_params/1e6:.1f}M")

    # SwiGLU FFN: d_ff = 8D/3 (保持同参数量)
    d_ff_swi = int(8 * D / 3)
    # 实际 Llama 会 round 到 256 的倍数
    d_ff_swi = (d_ff_swi + 255) // 256 * 256
    swi_params = 3 * D * d_ff_swi
    print(f"\nSwiGLU FFN (d_ff=8D/3≈{d_ff_swi}):")
    print(f"  W_gate: {D}×{d_ff_swi} + W_up: {D}×{d_ff_swi} + W_down: {d_ff_swi}×{D} = {swi_params/1e6:.1f}M")
    print(f"\n参数量比: {swi_params/std_params:.2f}x (接近 1.0)")


if __name__ == "__main__":
    demo()
