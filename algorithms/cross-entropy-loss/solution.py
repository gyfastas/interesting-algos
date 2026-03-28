"""
手写 Cross Entropy Loss — 纯 Python + PyTorch 实现

重点：为什么不分开算 softmax + log + nll，而是用 log-sum-exp 一步到位。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Naive 版本（会出数值问题）
# ============================================================
def cross_entropy_naive(logits: list[float], target: int) -> float:
    """
    先 softmax，再 log，再取 target 位置的负值。

    问题：softmax 可能下溢到 0 → log(0) = -inf
    """
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    probs = [e / total for e in exps]

    # 如果 target 对应的概率极小，log 会趋向 -inf
    return -math.log(probs[target] + 1e-12)  # 加 eps 勉强救一下


# ============================================================
# 2. 数值稳定版本（log-sum-exp 技巧）
# ============================================================
def cross_entropy_stable(logits: list[float], target: int) -> float:
    """
    CE(logits, target) = -log_softmax(logits)[target]
                       = -logits[target] + log(Σ exp(logits))
                       = -logits[target] + max + log(Σ exp(logits - max))

    全程不算 softmax 本身，避开下溢。
    """
    m = max(logits)
    log_sum_exp = m + math.log(sum(math.exp(x - m) for x in logits))
    return -logits[target] + log_sum_exp


# ============================================================
# 3. 支持 batch 的 PyTorch 实现
# ============================================================
def cross_entropy_torch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    手写的 PyTorch cross entropy loss。
    logits: (B, C) — 未经 softmax 的原始输出
    targets: (B,)  — 每个样本的正确类别索引

    等价于 F.cross_entropy(logits, targets)
    """
    # log-sum-exp（数值稳定）
    m = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))

    # 取 target 对应的 logit
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    # CE = -logit[target] + log_sum_exp
    loss = -target_logits + log_sum_exp
    return loss.mean()  # reduction='mean'


# ============================================================
# 4. 带 label smoothing 的版本
# ============================================================
def cross_entropy_label_smoothing(
    logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    """
    Label smoothing: 不用 one-hot，而是 (1-α)·one_hot + α/C·ones

    正确类别的目标概率从 1.0 变成 1-α+α/C，
    其他类别从 0 变成 α/C。
    """
    C = logits.size(-1)

    # log_softmax（数值稳定）
    m = logits.max(dim=-1, keepdim=True).values
    log_probs = logits - m - torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))

    # NLL for target class
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Smooth loss = 均匀分布的 CE = -mean(log_probs)
    smooth_loss = -log_probs.mean(dim=-1)

    # 加权��合
    loss = (1 - smoothing) * nll + smoothing * smooth_loss
    return loss.mean()


# ============================================================
# 5. Focal Loss（处理类别不平衡）
# ============================================================
def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal Loss = -(1 - p_t)^γ · log(p_t)

    当 p_t 接近 1（简单样本）时，(1-p_t)^γ ≈ 0，loss 被压低。
    当 p_t 接近 0（困难样本）时，(1-p_t)^γ ≈ 1，loss 不变。
    效果：自动关注困难样本。
    """
    # 数值稳定的 log_softmax
    m = logits.max(dim=-1, keepdim=True).values
    log_probs = logits - m - torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))

    # p_t 和 log(p_t) for target class
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()

    loss = -((1 - pt) ** gamma) * log_pt
    return loss.mean()


# ============================================================
# 数值演示
# ============================================================
def demo():
    print("=" * 60)
    print("Cross Entropy Loss 手写实现演示")
    print("=" * 60)

    # Case 1: 基础示例
    logits = [2.0, 1.0, 0.1]
    target = 0  # 正确答案是第 0 类
    print(f"\n--- Case 1: logits={logits}, target={target} ---")

    naive = cross_entropy_naive(logits, target)
    stable = cross_entropy_stable(logits, target)
    print(f"Naive CE:  {naive:.6f}")
    print(f"Stable CE: {stable:.6f}")

    # 手动验证
    m = max(logits)
    probs = [math.exp(x - m) for x in logits]
    total = sum(probs)
    probs = [p / total for p in probs]
    print(f"Softmax:   [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"-log(p[{target}]) = -log({probs[target]:.4f}) = {-math.log(probs[target]):.6f} ✓")

    # Case 2: 极端情况——target 概率极小
    logits2 = [10.0, -10.0, -10.0]
    target2 = 1  # 正确答案是概率极小的类
    print(f"\n--- Case 2: logits={logits2}, target={target2} (困难样本) ---")
    stable2 = cross_entropy_stable(logits2, target2)
    print(f"Stable CE: {stable2:.4f}")
    print(f"→ loss 很大，因为模型对正确类的置信度极低")

    # Case 3: 完美预测
    logits3 = [100.0, -100.0, -100.0]
    target3 = 0
    print(f"\n--- Case 3: logits={logits3}, target={target3} (完美预测) ---")
    stable3 = cross_entropy_stable(logits3, target3)
    print(f"Stable CE: {stable3:.6f}")
    print(f"→ loss ≈ 0，模型非常确信正确类")

    # Case 4: 均匀预测
    logits4 = [0.0, 0.0, 0.0]
    target4 = 0
    print(f"\n--- Case 4: logits={logits4}, target={target4} (均匀预测) ---")
    stable4 = cross_entropy_stable(logits4, target4)
    print(f"Stable CE: {stable4:.6f}")
    print(f"理论值 -log(1/3) = {-math.log(1/3):.6f}")
    print(f"→ 完全不确定时 loss = log(C)")

    # CE 的值域
    print(f"\n{'='*60}")
    print("CE Loss 值域参考 (C 个类别)")
    print("=" * 60)
    for C in [2, 10, 100, 1000, 32000, 128000]:
        print(f"  C={C:>6d}:  完美预测 → 0,  均匀预测 → {math.log(C):.2f},  反向预测 → 远大于 log(C)")

    print(f"\n{'='*60}")
    print("LLM 场景: vocab_size=128000")
    print("=" * 60)
    print(f"  随机猜: CE = log(128000) = {math.log(128000):.2f}")
    print(f"  对应 perplexity = exp(CE) = 128000")
    print(f"  好模型: CE ≈ 2.0 → ppl ≈ {math.exp(2.0):.1f}")
    print(f"  优秀模型: CE ≈ 1.5 → ppl ≈ {math.exp(1.5):.1f}")


if __name__ == "__main__":
    demo()
