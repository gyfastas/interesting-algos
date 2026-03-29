"""
LoRA (Low-Rank Adaptation) — 手写实现

重点：LoRALinear 层的定义、forward、merge/unmerge、apply 到已有模型。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA Linear 层
# ============================================================
class LoRALinear(nn.Module):
    """
    给一个冻结的 nn.Linear 加上 LoRA 旁路。

    原始: y = Wx
    LoRA: y = Wx + (α/r) · BAx

    其中:
      W: (out, in) — 冻结的原始权重
      A: (r, in)   — 低秩下投影，用高斯初始化
      B: (out, r)  — 低秩上投影，用零初始化（保证训练开始时 ΔW=0）
      r: rank，通常 4~64
      α: 缩放因子，控制 LoRA 的影响幅度
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 旁路参数（只训练这两个）
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # A 用 Kaiming 初始化，B 用零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径（冻结）
        base_out = self.linear(x)

        if self.merged:
            # 已经 merge 了，原始权重已包含 LoRA，直接返回
            return base_out

        # LoRA 旁路: x → A → B → scale
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def merge(self):
        """将 LoRA 权重合并到原始权重中，推理时零额外开销。"""
        if not self.merged:
            # W' = W + (α/r) · B @ A
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge(self):
        """将 LoRA 权重从原始权重中剥离，恢复到训练状态。"""
        if self.merged:
            self.linear.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False

    def extra_repr(self):
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}")


# ============================================================
# 2. 把 LoRA Apply 到已有模型上
# ============================================================
def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    遍历模型，把指定的 nn.Linear 替换��� LoRALinear。

    target_modules: 要加 LoRA 的模块名后缀列表
        例如 ["q_proj", "v_proj"] 只给 attention 的 Q 和 V 加 LoRA
        如果为 None，给所有 Linear 加
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 检查是否是目标模块
        if target_modules is not None:
            if not any(name.endswith(t) for t in target_modules):
                continue

        # 替换为 LoRALinear
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)

        # 在父模块中替换
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            child_name = parts[0]
            parent = model
        setattr(parent, child_name, lora_layer)

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ============================================================
# 3. Merge 所有 LoRA 层（推理部署用）
# ============================================================
def merge_lora(model: nn.Module):
    """合并所有 LoRA 层，推理时恢复到普通 Linear 的速度。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module):
    """反合并所有 LoRA 层，恢复到训练状态。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


# ============================================================
# 演示
# ============================================================
def demo():
    torch.manual_seed(42)

    print("=" * 60)
    print("LoRA 手写实现演示")
    print("=" * 60)

    # 模拟一个小 Transformer 层
    class MiniTransformerLayer(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)
            self.ffn_up = nn.Linear(d, d * 4)
            self.ffn_down = nn.Linear(d * 4, d)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn = F.softmax(q @ k.transpose(-1, -2) / 16, dim=-1) @ v
            x = x + self.o_proj(attn)
            x = x + self.ffn_down(F.silu(self.ffn_up(x)))
            return x

    model = MiniTransformerLayer(d=256)
    t_before, total_before = count_parameters(model)
    print(f"\n原始模型: {total_before:,} 参数 (全部可训练)")

    # Apply LoRA（只给 Q 和 V 加）
    model = apply_lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    t_after, total_after = count_parameters(model)
    print(f"LoRA 后:  {t_after:,} 可训练 / {total_after:,} 总参数")
    print(f"可训练比例: {t_after/total_after*100:.2f}%")
    print(f"新增参数: {total_after - total_before:,} (2 × rank × d × 2层 = {2*8*256*2:,})")

    # Forward 验证
    x = torch.randn(2, 8, 256)

    # 训练时: 基础路径 + LoRA 旁路
    y_train = model(x)
    print(f"\n训练 forward: {y_train.shape}")

    # Merge 后推理
    merge_lora(model)
    y_merged = model(x)
    diff = (y_train - y_merged).abs().max().item()
    print(f"Merge 后 forward: {y_merged.shape}")
    print(f"Merge 前后最大差异: {diff:.2e} (应该 ≈ 0)")

    # Unmerge 恢复
    unmerge_lora(model)
    y_unmerged = model(x)
    diff2 = (y_train - y_unmerged).abs().max().item()
    print(f"Unmerge 后最大差异: {diff2:.2e} (应该 ≈ 0)")

    # 参数量对比
    print(f"\n{'='*60}")
    print("典型 LLM 的 LoRA 参数量")
    print("=" * 60)
    for model_name, d, layers in [("7B", 4096, 32), ("13B", 5120, 40), ("70B", 8192, 80)]:
        full = layers * (4 * d * d + 2 * d * 4 * d)  # attn + ffn
        for r in [8, 16, 64]:
            # 通常只加到 q_proj, v_proj
            lora_params = layers * 2 * (r * d + d * r)  # A + B for 2 modules
            print(f"  {model_name} r={r:2d}: LoRA={lora_params/1e6:.1f}M "
                  f"({lora_params/full*100:.2f}% of {full/1e9:.1f}B)")


if __name__ == "__main__":
    demo()
